"""ACI wrapper around trained F7/F8 quantile predictions.

Implements Theorem 1b coverage: for the *median* (q50) point prediction we
treat the absolute residual as the nonconformity score and update an online
ACI calibrator step-by-step over the val set. We report empirical
miscoverage and width vs the target alpha.

This is a *retrospective* coverage check at sprint day 2. The production
verifier will call this same calibrator online (see ``apps/verifier/conformal``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.eval.coverage import marginal_coverage


@dataclass
class AciResult:
    alpha_target: float
    empirical_coverage: float
    mean_width: float
    n_steps: int
    intervals: np.ndarray  # (n_steps, horizon, 2)


def aci_coverage_from_val(
    val_preds_path: Path,
    *,
    alpha: float = 0.1,
    gamma: float = 0.05,
    horizon_step: int = 0,
) -> AciResult:
    """Run ACI online over the val set saved by ``train_model``.

    Parameters
    ----------
    val_preds_path:
        Path to the ``val_preds.npz`` produced by training (preds shape
        (N, H, Q), targets shape (N, H), both in original units).
    alpha:
        Target long-run miscoverage rate.
    gamma:
        ACI step size; 0.05 follows Gibbs & Candes' default for moderate windows.
    horizon_step:
        Which forecast lead to evaluate (0 = next 15 min).
    """
    z = np.load(val_preds_path)
    preds = z["preds"]  # (N, H, Q)
    targets = z["targets"]  # (N, H)

    # q50 point predictions; nonconformity score = |y − q50|.
    q50 = preds[:, horizon_step, preds.shape[-1] // 2]
    y = targets[:, horizon_step]
    scores = np.abs(y - q50)

    aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)
    # Warm-start with first 100 scores; the rest go through ACI online.
    warm = min(100, scores.size // 4)
    aci.warm_start(scores[:warm])

    intervals = np.empty((scores.size - warm, 2), dtype=np.float64)
    covered = 0
    for i, s in enumerate(scores[warm:]):
        q = aci.quantile()
        lo = q50[warm + i] - q
        hi = q50[warm + i] + q
        intervals[i] = (lo, hi)
        if lo <= y[warm + i] <= hi:
            covered += 1
        aci.update(float(s))

    coverage = covered / max(intervals.shape[0], 1)
    # Width can be inf for the first few ACI ticks if alpha_t hits the
    # boundary; report mean over finite intervals only.
    diffs = intervals[:, 1] - intervals[:, 0]
    finite = np.isfinite(diffs)
    width = float(np.mean(diffs[finite])) if finite.any() else float("nan")
    return AciResult(
        alpha_target=alpha,
        empirical_coverage=coverage,
        mean_width=width,
        n_steps=intervals.shape[0],
        intervals=intervals,
    )


__all__ = ["AciResult", "aci_coverage_from_val"]
