"""AF2 — patchTST encoder-depth ablation. Single seed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from heimdall_forecaster.train.run import REPO_ROOT, _load
from heimdall_forecaster.train.trainer import train_model
from heimdall_ml import seeds
from heimdall_ml.eval.leakage import assert_no_test_overlap

CFG_PATH = REPO_ROOT / "apps/forecaster/src/heimdall_forecaster/train/configs/f7.yaml"


def main() -> int:
    seeds.seed_everything(42)
    base = _load(CFG_PATH)
    results = []
    for L in (2, 4, 6, 8):
        cfg = replace(base, n_layers=L, name=f"f7_depth{L}")
        assert_no_test_overlap(cfg.train_panel, role="train")
        assert_no_test_overlap(cfg.val_panel, role="val")
        r = train_model(cfg)
        results.append(
            {"n_layers": L, "val_pinball_mean": r["val_pinball_mean"],
             "val_q10_q90_coverage": r["val_q10_q90_coverage"]}
        )
        print(json.dumps(results[-1], indent=2))
    out = REPO_ROOT / "experiments/outputs/af2_depth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
