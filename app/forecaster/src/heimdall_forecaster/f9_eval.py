"""F9 (TimesFM-2.0) zero-shot evaluation on DK1 imbalance prices.

Per docs/RESEARCH-PROPOSAL.md §4.2.2 row F9 + §4.4 (Theorem 1a). We:
  1. Slide a rolling forecast window across the val panel,
  2. Take the median (q50) point prediction at horizon step 1,
  3. Compute |y - q50| residuals on a calibration prefix,
  4. Apply split-CP from ``packages/ml/conformal/split_cp.py`` to
     produce a finite-sample (1 - alpha) interval,
  5. Verify empirical coverage on the held-out tail.

This satisfies Theorem 1a (under a stationarity assumption between the
calibration prefix and the tail; the post-2025 window has only mild
non-stationarity over 8 weeks, so we expect coverage ≥ 0.90 at α=0.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from heimdall_forecaster.timesfm_wrapper import TimesFMForecaster
from heimdall_ml.conformal.split_cp import SplitConformal


@dataclass
class F9Result:
    n_points: int
    pinball_q50: float
    cp_alpha: float
    cp_quantile: float
    cp_empirical_coverage: float
    cp_mean_width: float


def evaluate(
    val_panel: Path,
    *,
    target_col: str = "imbalance_price_dkk_mwh_15min",
    seq_len: int = 256,
    horizon: int = 16,
    n_eval_windows: int = 200,
    cal_frac: float = 0.5,
    alpha: float = 0.1,
    backend: str = "gpu",
) -> F9Result:
    df = pl.read_parquet(val_panel).drop_nulls()
    y = df[target_col].to_numpy().astype(np.float64)
    if y.size < seq_len + horizon + n_eval_windows:
        raise ValueError("val panel too short for the requested eval grid")

    f9 = TimesFMForecaster(backend=backend, context_len=seq_len, horizon_len=horizon)
    # Pre-load weights so timing is not skewed.
    f9._load()

    # Roll a window forward `n_eval_windows` times; predict step-1 next-quarter.
    # Skip the first 24 hours of val (potentially constant carry-over from
    # forward-fill at series start).
    burn_in = min(96, y.size // 4)
    starts = np.linspace(seq_len + burn_in, y.size - horizon - 1, n_eval_windows, dtype=int)
    preds_q50 = np.empty(n_eval_windows, dtype=np.float64)
    truths = np.empty(n_eval_windows, dtype=np.float64)
    for i, s in enumerate(starts):
        history = y[s - seq_len : s]
        mean, _ = f9.predict(history)
        preds_q50[i] = float(mean[0])
        truths[i] = y[s]

    abs_resid = np.abs(truths - preds_q50)
    pinball_q50 = float(np.mean(np.maximum(0.5 * (truths - preds_q50), -0.5 * (truths - preds_q50))))

    n_cal = int(round(cal_frac * n_eval_windows))
    cal_scores = abs_resid[:n_cal]
    test_resid = abs_resid[n_cal:]
    cp = SplitConformal.fit(cal_scores, alpha=alpha)
    in_band = test_resid <= cp.quantile
    coverage = float(np.mean(in_band))
    width = 2.0 * float(cp.quantile)
    return F9Result(
        n_points=n_eval_windows,
        pinball_q50=pinball_q50,
        cp_alpha=alpha,
        cp_quantile=float(cp.quantile),
        cp_empirical_coverage=coverage,
        cp_mean_width=width,
    )


__all__ = ["F9Result", "evaluate"]
