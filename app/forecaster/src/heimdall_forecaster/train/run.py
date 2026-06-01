"""CLI runner: ``uv run python -m heimdall_forecaster.train.run --config ...``.

Reads a config YAML, resolves repo-relative panel paths, and invokes
``train_model``. Prints a one-line summary at the end so CI smoke tests can
assert on val pinball.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from heimdall_forecaster.train.trainer import TrainConfig, train_model
from heimdall_ml.eval.leakage import assert_no_test_overlap

REPO_ROOT = Path(__file__).resolve().parents[5]


def _load(cfg_path: Path) -> TrainConfig:
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    raw["quantiles"] = tuple(raw.get("quantiles", (0.1, 0.5, 0.9)))
    raw["train_panel"] = REPO_ROOT / raw.get("train_panel", "data/processed/dk1_panel_train.parquet")
    raw["val_panel"] = REPO_ROOT / raw.get("val_panel", "data/processed/dk1_panel_val.parquet")
    raw["out_dir"] = REPO_ROOT / raw.get("out_dir", "models/forecaster")
    if "anomaly_panel" in raw and raw["anomaly_panel"]:
        raw["anomaly_panel"] = REPO_ROOT / raw["anomaly_panel"]
    # Resolve named feature sets for the F8b/F8c rich-feature variants.
    name = raw.get("name", "")
    base = name
    if "feature_names" not in raw or raw.get("feature_names") is None:
        if base == "f8b" or name == "f8b":
            from heimdall_forecaster.train.dataset import F8B_FEATURES

            raw["feature_names"] = F8B_FEATURES
        elif base == "f8c" or name == "f8c":
            from heimdall_forecaster.train.dataset import F8C_FEATURES

            raw["feature_names"] = F8C_FEATURES
        elif base == "f8d" or name == "f8d":
            from heimdall_forecaster.train.dataset import F8D_FEATURES

            raw["feature_names"] = F8D_FEATURES
        elif base == "f8e" or name == "f8e":
            from heimdall_forecaster.train.dataset import F8E_FEATURES

            raw["feature_names"] = F8E_FEATURES
        elif base == "f13" or name == "f13":
            from heimdall_forecaster.train.dataset import F13_FEATURES

            raw["feature_names"] = F13_FEATURES
        elif base == "f8w_da" or name == "f8w_da":
            from heimdall_forecaster.train.dataset import F13_FEATURES

            raw["feature_names"] = F13_FEATURES
        elif base == "f8w72" or name == "f8w72":
            from heimdall_forecaster.train.dataset import F8WX72_FEATURES

            raw["feature_names"] = F8WX72_FEATURES
    return TrainConfig(**raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    cfg = _load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    # Per docs/RESEARCH-PROPOSAL.md §5.7 — test set is sacred. Refuse to start training
    # if either panel ranges into the held-out window.
    assert_no_test_overlap(cfg.train_panel, role="train")
    assert_no_test_overlap(cfg.val_panel, role="val")
    result = train_model(cfg)
    summary = {
        "name": cfg.name,
        "seed": cfg.seed,
        "val_pinball_mean": float(result["val_pinball_mean"]),
        "val_q10_q90_coverage": float(result["val_q10_q90_coverage"]),
        "per_quantile": {k: float(v) for k, v in result["per_quantile"].items()},
        "ckpt": str(result["ckpt"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
