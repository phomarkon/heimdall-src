"""Test-set ACI coverage from cached forecaster predictions (RQ2, Theorem 1b).

Runs the *production* online-ACI calibrator (``heimdall_ml.conformal.aci``,
the same class the verifier's calibrator uses) over the cached held-out TEST
predictions in ``outputs/test_preds/<model>/seed-<s>.npz`` and reports
empirical coverage at the 0.90 target, averaged over the five frozen seeds.

This is the reproducible source for the post-break test-set coverage number.
It uses no GPU and no checkpoint download: it consumes the already-saved
quantile predictions, so it runs in seconds on a laptop CPU.

Note on window: the cached test predictions are the first 10,000 post-break
quarter-hours of the test split (2025-05-03 .. 2025-08-15). Regenerating the
full 12-month test window requires a fresh forecaster inference pass.

Usage:
    python research/ml/experiments/test_set_aci_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_PREDS = SCRIPT_DIR / "outputs/test_preds"
OUT = SCRIPT_DIR / "outputs/test_set_aci_coverage.json"

SEEDS = [13, 42, 137, 1729, 31415]
ALPHA = 0.1  # target coverage 1 - alpha = 0.90
GAMMAS = [0.005, 0.01, 0.05]
MODELS = ["f8", "f7", "f1_lgbm", "f3_ensemble"]


def main() -> int:
    results = []
    for model in MODELS:
        for gamma in GAMMAS:
            covs, widths = [], []
            for seed in SEEDS:
                p = TEST_PREDS / model / f"seed-{seed}.npz"
                if not p.exists():
                    continue
                r = aci_coverage_from_val(p, alpha=ALPHA, gamma=gamma, horizon_step=0)
                covs.append(r.empirical_coverage)
                widths.append(r.mean_width)
            if not covs:
                continue
            covs = np.asarray(covs)
            results.append(
                {
                    "model": model,
                    "gamma": gamma,
                    "alpha": ALPHA,
                    "target_coverage": 1 - ALPHA,
                    "n_seeds": int(covs.size),
                    "coverage_mean": float(covs.mean()),
                    "coverage_std": float(covs.std()),
                    "coverage_per_seed": [float(c) for c in covs],
                    "mean_width": float(np.mean(widths)),
                }
            )
            print(
                f"{model:12s} gamma={gamma:.3f}  "
                f"coverage={covs.mean():.4f} +/- {covs.std():.4f}  (n={covs.size})"
            )

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
