"""Split-conformal prediction — Theorem 1a, finite-sample coverage.

Implements Vovk-Shafer split CP with the standard finite-sample correction:
given i.i.d. (or exchangeable) calibration scores `s_1, ..., s_n` drawn from the
same distribution as the test score `s_{n+1}`, the empirical quantile

    q_hat = ceil((n + 1) * (1 - alpha)) / n   th order statistic of {s_i}

satisfies P(s_{n+1} <= q_hat) >= 1 - alpha exactly (Theorem 1a in
docs/RESEARCH-PROPOSAL.md §4.6). Coverage is *marginal*, not conditional, and
*requires exchangeability*. Pre-/post-2025-03-04 mixing breaks exchangeability;
do not pool calibration sets across the regime break.

Reference: Vovk, Gammerman & Shafer 2005; Lei et al. 2018 JASA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_quantile(scores: ArrayLike, alpha: float) -> float:
    """Compute the finite-sample-corrected (1 - alpha) calibration quantile.

    Per Theorem 1a (docs/RESEARCH-PROPOSAL.md §4.6), this returns the
    `ceil((n+1)*(1-alpha))/n`-th empirical quantile of the calibration scores.
    The +1 / n correction is what gives the *exact* finite-sample coverage
    bound under exchangeability — it is NOT a heuristic.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        raise ValueError("split-CP needs at least one calibration score")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    # Finite-sample correction: rank index k = ceil((n + 1)(1 - alpha)).
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        # If alpha is so small that the correction overshoots, return the supremum
        # (i.e. infinite interval); this is exactly the behaviour split-CP
        # demands and is preferable to silent under-coverage.
        return float(np.inf)
    sorted_s = np.partition(s, k - 1)
    return float(sorted_s[k - 1])


def predict_in_band(score: float, quantile: float) -> bool:
    """True iff `score <= quantile`. Trivial wrapper kept for API symmetry."""
    return float(score) <= float(quantile)


@dataclass(frozen=True)
class SplitConformal:
    """Frozen split-conformal calibrator. Holds the fitted quantile only."""

    alpha: float
    quantile: float
    n_calibration: int

    @classmethod
    def fit(cls, scores: ArrayLike, alpha: float = 0.1) -> SplitConformal:
        s = np.asarray(scores, dtype=np.float64).ravel()
        return cls(alpha=alpha, quantile=fit_quantile(s, alpha), n_calibration=int(s.size))

    def is_in_band(self, score: float) -> bool:
        return predict_in_band(score, self.quantile)

    def interval_from_point(
        self, point_pred: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert an array of point predictions into symmetric +/- q intervals.

        This assumes the nonconformity score is the absolute residual
        |y - y_hat|; for asymmetric scores (e.g. quantile-regression CQR), call
        `fit_quantile` directly on the appropriate score function.
        """
        p = np.asarray(point_pred, dtype=np.float64)
        return p - self.quantile, p + self.quantile
