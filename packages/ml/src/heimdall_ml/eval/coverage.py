"""Coverage diagnostics. Per docs/RESEARCH-PROPOSAL.md §5.3 (metrics)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def marginal_coverage(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """Empirical marginal coverage = (1/T) sum 1{y_t in [l_t, u_t]}."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return float(np.mean((y >= lo) & (y <= hi)))


def conditional_coverage(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    strata: Sequence[int],
) -> dict[int, float]:
    """Coverage stratified by an integer label (e.g. regime, hour-of-day)."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    s = np.asarray(strata, dtype=np.int64)
    out: dict[int, float] = {}
    for k in np.unique(s):
        mask = s == k
        out[int(k)] = float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))
    return out


def interval_width(lower: ArrayLike, upper: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)


def pinball_loss(y: ArrayLike, q_pred: ArrayLike, level: float) -> float:
    """Quantile (pinball) loss at the given level in (0, 1)."""
    if not (0.0 < level < 1.0):
        raise ValueError("level must lie in (0, 1)")
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q_pred, dtype=np.float64)
    diff = y - q
    return float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))
