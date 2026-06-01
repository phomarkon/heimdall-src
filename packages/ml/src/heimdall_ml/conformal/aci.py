"""Online adaptive conformal inference — Theorem 1b, long-run coverage.

Implements Gibbs & Candes (2021, NeurIPS) ACI: at each step t observe whether
the realised value fell inside the current interval, and update the effective
miscoverage rate `alpha_t` by

    alpha_{t+1} = alpha_t + gamma * (alpha - 1{y_t not in [l_t, u_t]})

The interval at step t is the (1 - alpha_t)-empirical quantile of the
calibration scores. Per Theorem 1b (docs/RESEARCH-PROPOSAL.md §4.6), the empirical
miscoverage rate converges to alpha as T -> infinity *without* exchangeability
or stationarity. The post-2025-03-04 regime is precisely the regime where
this matters; ACI is therefore the focal-agent default.

The state is a *single scalar* (`alpha_t`) plus a calibration buffer; this is
intentionally trivial — the heavy lifting is upstream in the forecaster's
nonconformity score.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import ArrayLike


class AdaptiveConformalInference:
    """Stateful online ACI calibrator.

    Parameters
    ----------
    alpha:
        Target long-run miscoverage rate (e.g. 0.1 for 90% intervals).
    gamma:
        Learning rate for the alpha update. Gibbs & Candes report results
        across gamma in [0.005, 0.05]; we default to 0.05.
    window:
        Maximum size of the running calibration buffer. None means unbounded
        (use only on bounded-length test sets).

    Notes
    -----
    Coverage is *long-run* and *marginal*. A finite-T fluctuation analysis
    requires the regret bounds in Bhatnagar et al. 2023 / Angelopoulos et al.
    2024 — cited in §4.6.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.05, window: int | None = 5_000):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self._scores: deque[float] = deque(maxlen=window)
        self._t = 0
        self._misses = 0

    # --- internal -----------------------------------------------------

    def _quantile(self) -> float:
        if not self._scores:
            return float("inf")
        if self.alpha_t <= 0.0:
            return float("inf")
        if self.alpha_t >= 1.0:
            return float("-inf")
        return float(np.quantile(self._scores, 1.0 - self.alpha_t, method="higher"))

    # --- public API ---------------------------------------------------

    def warm_start(self, calibration_scores: ArrayLike) -> None:
        """Seed the calibration buffer; does not change `alpha_t`."""
        for s in np.asarray(calibration_scores, dtype=np.float64).ravel().tolist():
            self._scores.append(float(s))

    def quantile(self) -> float:
        """Current effective (1 - alpha_t) calibration quantile."""
        return self._quantile()

    def predict_in_band(self, score: float) -> bool:
        return float(score) <= self._quantile()

    def update(self, realised_score: float) -> None:
        """Observe a fresh nonconformity score and update alpha_t in-place.

        Call AFTER the verifier has used the current quantile to decide.
        """
        miss = 0.0 if float(realised_score) <= self._quantile() else 1.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - miss)
        # Clip into [0, 1] to keep np.quantile well-defined; the long-run
        # guarantee in Gibbs & Candes is unchanged by clipping.
        self.alpha_t = float(np.clip(self.alpha_t, 0.0, 1.0))
        self._scores.append(float(realised_score))
        self._t += 1
        self._misses += int(miss)

    # --- diagnostics --------------------------------------------------

    @property
    def empirical_miscoverage(self) -> float:
        return self._misses / self._t if self._t > 0 else float("nan")

    @property
    def steps(self) -> int:
        return self._t
