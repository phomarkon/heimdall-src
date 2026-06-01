"""Bayesian online change-point detection (BOCPD) for regime-shift detection.

Per the depth-play discussed in the 2026-05-10 strategy session: BOCPD
(Adams & MacKay 2007) detects breaks in the data-generating process online
via a posterior over the most recent change-point's run-length r_t.  Pairs
with online ACI (`heimdall_ml.conformal.aci`) to give the verifier two
calibration regimes:

  - *pre-detection*: vanilla ACI with global calibration buffer.
  - *post-detection*: split-CP restricted to scores observed since the most
    recent BOCPD-detected change-point.  Within-regime exchangeability is
    plausible; Theorem 1a's finite-sample guarantee applies *post-detection*.

This module implements the standard BOCPD recursion under a
constant-hazard (geometric run-length prior) and a Gaussian-conjugate
predictive likelihood.  For EPF residuals the predictive Student-t with
Normal-Inverse-Gamma priors is used (closed-form posterior updates).

Reference:
- Adams, R. P. and MacKay, D. J. C. (2007). Bayesian Online Changepoint
  Detection. arXiv:0710.3742.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import t as student_t

__all__ = ["BOCPD", "BOCPDResult"]


@dataclass
class BOCPDResult:
    """Output of a single observation step."""
    map_run_length: int
    posterior: np.ndarray  # (R+1,) over r_t = 0..R
    detected_change: bool
    most_likely_changepoint: int  # index in the source series


@dataclass
class BOCPD:
    """Online BOCPD with a Normal-Inverse-Gamma predictive.

    Hazard model: constant ``hazard = 1 / mean_run_length``.
    Predictive: :math:`p(x_t \\mid r_{t-1}) = \\text{Student-t}` updated
    online via NIG conjugacy.

    Detection rule: report a change at step t iff the posterior MAP
    run-length collapses below ``detection_threshold`` ticks while the
    previous MAP was above ``detection_threshold * 4`` (i.e.\\ a sudden
    jump from "in regime" to "fresh start").
    """

    mean_run_length: float = 200.0       # prior mean — ~2 days at 96 ticks/day
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    detection_threshold: int = 4
    detection_prev_threshold: int = 32

    # Internal state
    _posterior: np.ndarray = field(default=None, repr=False)
    _mu: np.ndarray = field(default=None, repr=False)
    _kappa: np.ndarray = field(default=None, repr=False)
    _alpha: np.ndarray = field(default=None, repr=False)
    _beta: np.ndarray = field(default=None, repr=False)
    _t: int = 0
    _last_map: int = 0

    def __post_init__(self) -> None:
        self._posterior = np.array([1.0])
        self._mu = np.array([self.mu0])
        self._kappa = np.array([self.kappa0])
        self._alpha = np.array([self.alpha0])
        self._beta = np.array([self.beta0])

    @property
    def hazard(self) -> float:
        return 1.0 / max(self.mean_run_length, 1.0)

    def _predictive_logpdf(self, x: float) -> np.ndarray:
        df = 2.0 * self._alpha
        scale_sq = (self._beta * (self._kappa + 1.0)) / (self._alpha * self._kappa)
        scale = np.sqrt(np.maximum(scale_sq, 1e-12))
        return student_t.logpdf(x, df=df, loc=self._mu, scale=scale)

    def step(self, x: float) -> BOCPDResult:
        """Process one observation and return the posterior + detection flag."""
        # Predictive log-likelihood for each existing run-length.
        log_pred = self._predictive_logpdf(x)
        # Growth probabilities: p(r_t = r_{t-1}+1) = posterior * (1-h) * pred
        log_post = np.log(self._posterior + 1e-300) + log_pred
        log_growth = log_post + np.log1p(-self.hazard)
        # Change-point probability mass: sum over previous run-lengths.
        log_cp = float(np.logaddexp.reduce(log_post + np.log(self.hazard)))
        # New posterior (length grows by 1 each step).
        new_log = np.empty(log_growth.size + 1)
        new_log[0] = log_cp
        new_log[1:] = log_growth
        # Normalise.
        new_log -= np.logaddexp.reduce(new_log)
        new_post = np.exp(new_log)

        # Update sufficient statistics.
        new_mu = np.empty_like(new_post)
        new_kappa = np.empty_like(new_post)
        new_alpha = np.empty_like(new_post)
        new_beta = np.empty_like(new_post)
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0
        # Standard NIG update.
        new_mu[1:] = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        new_kappa[1:] = self._kappa + 1.0
        new_alpha[1:] = self._alpha + 0.5
        new_beta[1:] = self._beta + (
            self._kappa * (x - self._mu) ** 2 / (2.0 * (self._kappa + 1.0))
        )

        self._posterior = new_post
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta
        self._t += 1

        map_r = int(np.argmax(self._posterior))
        detected = (
            map_r <= self.detection_threshold
            and self._last_map >= self.detection_prev_threshold
        )
        most_likely_cp = self._t - map_r
        self._last_map = map_r
        return BOCPDResult(
            map_run_length=map_r,
            posterior=self._posterior.copy(),
            detected_change=detected,
            most_likely_changepoint=most_likely_cp,
        )

    def run(self, series: Sequence[float]) -> list[BOCPDResult]:
        return [self.step(x) for x in series]


def detect_changepoints(
    series: Sequence[float],
    *,
    mean_run_length: float = 200.0,
    detection_threshold: int = 4,
    detection_prev_threshold: int = 32,
) -> list[int]:
    """One-shot helper: return indices at which BOCPD flags a change."""
    bocpd = BOCPD(
        mean_run_length=mean_run_length,
        detection_threshold=detection_threshold,
        detection_prev_threshold=detection_prev_threshold,
    )
    out: list[int] = []
    for t, x in enumerate(series):
        r = bocpd.step(x)
        if r.detected_change:
            out.append(t)
    return out
