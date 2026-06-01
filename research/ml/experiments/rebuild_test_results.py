"""Rebuild test_set_results.json by re-running the metrics module over the
persisted experiments/outputs/test_preds/<model>/seed-*.npz files.

The single-shot evaluator overwrote results when invoked per-model; this
script consolidates all available NPZ-saved runs into one results file with
the full metric bundle, without re-running any inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heimdall_ml.eval import metrics as M

REPO = Path(__file__).resolve().parents[2]
PREDS_DIR = REPO / "experiments/outputs/test_preds"
OUT = REPO / "experiments/outputs/test_set_results.json"
LEVELS = (0.1, 0.5, 0.9)


def main() -> int:
    results: list[dict] = []
    for model_dir in sorted(PREDS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for seed_npz in sorted(model_dir.glob("seed-*.npz")):
            seed = int(seed_npz.stem.split("-")[1])
            d = np.load(seed_npz, allow_pickle=False)
            preds = d["preds"].astype(np.float64)
            targets = d["targets"].astype(np.float64)
            ts = d["timestamps"] if "timestamps" in d.files else None
            comp = M.collect_all(preds, targets, LEVELS, timestamps=ts)
            results.append({
                "model": model_dir.name,
                "seed": seed,
                "metrics": comp,
                "test_n_windows": int(preds.shape[0]),
                "test_pinball_mean_dkk": comp["pinball_mean"],
            })
            print(f"{model_dir.name} seed={seed} pinball_mean={comp['pinball_mean']:.2f} CRPS={comp['crps_quantile_dkk']:.2f} cov80={comp['interval_10_90_coverage']:.3f}")
    OUT.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nWrote {OUT} with {len(results)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
