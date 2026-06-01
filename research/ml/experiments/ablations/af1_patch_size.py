"""AF1 — patchTST patch-size ablation (forecaster-side).

Re-runs F7 training with patch_len ∈ {4, 8, 16}; everything else held to the
F7 anchor config. Single-seed 42 (per the proposal §5.4 guidance: "single seed
for ablations is fine, 5 seeds for the headline numbers").
"""

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
    for ps in (4, 8, 16):
        cfg = replace(base, patch_len=ps, name=f"f7_patch{ps}")
        # f7 anchor is patch_len 8 — log accordingly.
        assert_no_test_overlap(cfg.train_panel, role="train")
        assert_no_test_overlap(cfg.val_panel, role="val")
        r = train_model(cfg)
        results.append(
            {
                "patch_len": ps,
                "val_pinball_mean": r["val_pinball_mean"],
                "val_q10_q90_coverage": r["val_q10_q90_coverage"],
                "per_quantile": r["per_quantile"],
            }
        )
        print(json.dumps(results[-1], indent=2))
    out = REPO_ROOT / "experiments/outputs/af1_patch_size.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
