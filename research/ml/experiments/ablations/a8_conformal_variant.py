"""A8 — conformal variant ablation. docs/RESEARCH-PROPOSAL.md §5.4.

Three-way comparison of the conformal-paradigm trio:
  - split-CP (Theorem 1a; exchangeable, finite-sample);
  - online ACI (Theorem 1b; no exchangeability, long-run);
  - EnbPI (sliding-window OOB residuals; asymptotic-marginal).

We report empirical coverage, mean interval width, and coverage stability
under a synthetic 2× residual regime shift injected mid-sequence.

Hypotheses:
  - split-CP under-covers post regime shift (exchangeability fails).
  - ACI recovers coverage but with delay.
  - EnbPI tracks better than split-CP, worse than ACI on width — its sliding
    window absorbs the shift mechanically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.enbpi import enbpi_intervals
from heimdall_ml.conformal.split_cp import SplitConformal

REPO_ROOT = Path(__file__).resolve().parents[3]
F7_VAL = REPO_ROOT / "models/forecaster/f7/seed-42/val_preds.npz"


def _scores() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(F7_VAL)
    preds, targets = z["preds"], z["targets"]
    q50 = preds[:, 0, preds.shape[-1] // 2]
    y = targets[:, 0]
    return np.abs(y - q50), q50  # scores, q50


def _split_cp(scores: np.ndarray) -> dict:
    half = scores.size // 2
    cal, test = scores[:half], scores[half:]
    cp = SplitConformal.fit(cal, alpha=0.1)
    cov = float(np.mean(test <= cp.quantile))
    return {"empirical_coverage": cov, "mean_width": 2 * cp.quantile, "n_test": int(test.size)}


def _aci(scores: np.ndarray) -> dict:
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
    warm = min(200, scores.size // 4)
    aci.warm_start(scores[:warm])
    covs = 0
    n = 0
    widths = []
    for s in scores[warm:]:
        q = aci.quantile()
        if np.isfinite(q):
            widths.append(2 * q)
        if aci.predict_in_band(float(s)):
            covs += 1
        aci.update(float(s))
        n += 1
    return {
        "empirical_coverage": covs / max(n, 1),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "n_test": int(n),
    }


def _enbpi(scores: np.ndarray, q50: np.ndarray, *, window: int = 200) -> dict:
    """EnbPI on absolute residuals.

    EnbPI is naturally formulated on bagged-ensemble OOB predictions; for the
    A8 protocol we substitute the F7 single-seed q50 in place of the bagged
    point predictor. We keep the sliding-window residual buffer that gives
    EnbPI its local-adaptivity property — the substantive claim of A8 (EnbPI
    behaves *between* split-CP and ACI under regime shift) is preserved.
    """
    # Targets are score = |y - q50|, so y = q50 ± score; we feed targets and
    # point preds derived from `scores` only (we do not have per-row targets
    # at this layer). The (point_pred, targets) pair must be reconstructed:
    targets = q50 + scores  # absolute residuals are symmetric; signed irrelevant for |y - p|
    res = enbpi_intervals(point_pred=q50, targets=targets, alpha=0.1, window=window)
    return {
        "empirical_coverage": float(res.empirical_coverage),
        "mean_width": float(res.mean_width),
        "n_test": int(res.n_steps),
        "window": window,
    }


def _regime_shift(scores: np.ndarray, factor: float = 2.0) -> np.ndarray:
    out = scores.copy()
    out[len(out) // 2 :] *= factor
    return out


def main() -> int:
    scores, q50 = _scores()
    shifted = _regime_shift(scores)
    results = {
        "stationary": {
            "split_cp": _split_cp(scores),
            "aci": _aci(scores),
            "enbpi": _enbpi(scores, q50),
        },
        "regime_shift_2x": {
            "split_cp": _split_cp(shifted),
            "aci": _aci(shifted),
            "enbpi": _enbpi(shifted, q50),
        },
    }
    out = REPO_ROOT / "experiments/outputs/a8_conformal_variant.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
