"""EnbPI — Ensemble batch-prediction Inference. Per docs/RESEARCH-PROPOSAL.md §5.4
ablation A8 (split-CP / ACI / EnbPI three-way comparison).

EnbPI (Xu & Xie, 2021, ICML) is the third member of the §4.6 conformal-paradigm
trio. Where split-CP requires exchangeability and ACI guarantees only long-run
coverage, EnbPI uses out-of-bag residuals from a bagged ensemble to construct
prediction intervals that adapt to local non-stationarity *without* a held-out
calibration split — useful for time-series where every observation is precious.

Algorithm (univariate, fixed horizon):
1. Train ``B`` bootstrap models ``f_b`` on bootstrap subsamples of the train
   set; record, per train index ``i``, the indices of bootstraps that DID NOT
   include ``i`` (the OOB index sets ``S_i``).
2. Aggregate the OOB predictions: ``y_hat_oob[i] = mean_{b in S_i} f_b(x_i)``.
3. Compute residuals ``r_i = |y_i - y_hat_oob[i]|``.
4. At test time t, predict ``y_hat_t = mean_b f_b(x_t)``; the (1-α) interval
   width is the (1-α) empirical quantile of the *most recent* W residuals
   (sliding window — gives local adaptivity).

This module operates *post-hoc* on stored bootstrap predictions. It is wired
into A8 via `experiments/ablations/a8_conformal_variant.py`. Reference:
Xu & Xie 2021, "Conformal prediction interval for dynamic time-series".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EnbPIResult:
    alpha_target: float
    empirical_coverage: float
    mean_width: float
    intervals: NDArray[np.float64]
    n_steps: int
    window: int


def enbpi_intervals(
    point_pred: ArrayLike,
    targets: ArrayLike,
    *,
    alpha: float = 0.1,
    window: int = 200,
    warm: int | None = None,
    bootstrap_oob_residuals: ArrayLike | None = None,
) -> EnbPIResult:
    """Run EnbPI online over a 1-D test series.

    Parameters
    ----------
    point_pred:
        Length-T array of bagged-ensemble point predictions.
    targets:
        Length-T realised values.
    alpha:
        Target miscoverage rate.
    window:
        Sliding-window length used to fit the residual quantile. Xu & Xie use
        T/4 by default; we expose it for the A8 sensitivity sweep.
    warm:
        Number of initial steps used to seed the residual buffer before
        intervals are scored. Defaults to ``min(window, T // 4)``.
    bootstrap_oob_residuals:
        Optional length-T_train array of out-of-bag residuals from training.
        If supplied, the residual buffer is warm-started with these instead
        of the first `warm` test residuals — closer to Xu & Xie's offline
        formulation.

    Notes
    -----
    The standard EnbPI guarantee is *asymptotic* and *marginal*; finite-sample
    coverage is empirical. We do not claim Theorem 1a's exchangeability
    guarantee — A8's purpose is precisely to compare the three paradigms
    side-by-side.
    """
    p = np.asarray(point_pred, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: point_pred {p.shape} vs targets {y.shape}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    T = p.size
    if warm is None:
        warm = max(1, min(window, T // 4))

    abs_resid = np.abs(y - p)

    buf: list[float] = []
    if bootstrap_oob_residuals is not None:
        oob = np.asarray(bootstrap_oob_residuals, dtype=np.float64).ravel()
        buf.extend(oob[-window:].tolist())
    else:
        buf.extend(abs_resid[:warm].tolist())

    intervals = np.empty((T - warm, 2), dtype=np.float64)
    covered = 0
    for j, i in enumerate(range(warm, T)):
        if not buf:
            radius = float("inf")
        else:
            buf_arr = np.asarray(buf, dtype=np.float64)
            n = buf_arr.size
            # Finite-sample-corrected (1 - alpha) quantile, mirroring split-CP.
            k = int(np.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                radius = float(np.max(buf_arr))
            else:
                sorted_buf = np.partition(buf_arr, k - 1)
                radius = float(sorted_buf[k - 1])
        lo, hi = p[i] - radius, p[i] + radius
        intervals[j] = (lo, hi)
        if lo <= y[i] <= hi:
            covered += 1
        # Slide the window: append the freshest residual, drop the oldest.
        buf.append(float(abs_resid[i]))
        if len(buf) > window:
            buf.pop(0)

    n_steps = intervals.shape[0]
    coverage = covered / max(n_steps, 1)
    diffs = intervals[:, 1] - intervals[:, 0]
    finite = np.isfinite(diffs)
    width = float(np.mean(diffs[finite])) if finite.any() else float("nan")

    return EnbPIResult(
        alpha_target=alpha,
        empirical_coverage=coverage,
        mean_width=width,
        intervals=intervals,
        n_steps=n_steps,
        window=window,
    )


__all__ = ["EnbPIResult", "enbpi_intervals"]
