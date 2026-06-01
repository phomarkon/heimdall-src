"""5-seed sweep on F7/F8/F0/F3 — docs/RESEARCH-PROPOSAL.md §5.3.1.

Trains each config across the frozen seed set [13, 42, 137, 1729, 31415],
collects per-seed metrics, and emits ``models/forecaster/<name>/metrics.json``
plus ``notes/forecaster_leaderboard.md`` (mean ± std per quantile + ACI cov).

Usage:
    uv run python experiments/seed_sweep.py --models f7 f8 f0 f3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heimdall_forecaster.train.f0_ar import F0Config, train_f0
from heimdall_forecaster.train.f3_deepar import F3Config, train_f3
from heimdall_forecaster.train.run import REPO_ROOT, _load
from heimdall_forecaster.train.trainer import train_model
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds
from heimdall_ml.eval.leakage import assert_no_test_overlap

CONFIGS = {
    "f7": REPO_ROOT / "apps/forecaster/src/heimdall_forecaster/train/configs/f7.yaml",
    "f8": REPO_ROOT / "apps/forecaster/src/heimdall_forecaster/train/configs/f8.yaml",
}


def _train_seed(name: str, seed: int) -> dict:
    """Train ``name`` at ``seed`` and return per-seed metric dict."""
    if name in CONFIGS:
        cfg = _load(CONFIGS[name])
        cfg.seed = seed
        # Per docs/RESEARCH-PROPOSAL.md §5.7 — leakage assertion at every entry point.
        assert_no_test_overlap(cfg.train_panel, role="train")
        assert_no_test_overlap(cfg.val_panel, role="val")
        result = train_model(cfg)
    elif name == "f0":
        cfg0 = F0Config(
            seed=seed,
            train_panel=REPO_ROOT / "data/processed/dk1_panel_train.parquet",
            val_panel=REPO_ROOT / "data/processed/dk1_panel_val.parquet",
            out_dir=REPO_ROOT / "models/forecaster",
        )
        assert_no_test_overlap(cfg0.train_panel, role="train")
        assert_no_test_overlap(cfg0.val_panel, role="val")
        result = train_f0(cfg0)
    elif name == "f3":
        cfg3 = F3Config(
            seed=seed,
            train_panel=REPO_ROOT / "data/processed/dk1_panel_train.parquet",
            val_panel=REPO_ROOT / "data/processed/dk1_panel_val.parquet",
            out_dir=REPO_ROOT / "models/forecaster",
        )
        assert_no_test_overlap(cfg3.train_panel, role="train")
        assert_no_test_overlap(cfg3.val_panel, role="val")
        result = train_f3(cfg3)
    else:
        raise ValueError(f"unknown model {name}")

    # ACI wrap on the freshly-saved val_preds.
    out = REPO_ROOT / "models/forecaster" / name / f"seed-{seed}"
    aci = aci_coverage_from_val(out / "val_preds.npz", alpha=0.1, gamma=0.05)
    metrics = {
        "seed": seed,
        "val_pinball_q10": result["per_quantile"]["val_pinball_q10"],
        "val_pinball_q50": result["per_quantile"]["val_pinball_q50"],
        "val_pinball_q90": result["per_quantile"]["val_pinball_q90"],
        "val_pinball_mean": result["val_pinball_mean"],
        "val_q10_q90_coverage": result["val_q10_q90_coverage"],
        "aci_alpha_target": aci.alpha_target,
        "aci_empirical_coverage": aci.empirical_coverage,
        "aci_mean_width": aci.mean_width,
    }
    # Persist into model dir for the leaderboard.
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    # Persist a stand-alone aci_state.json so the caller can resume online.
    with open(out / "aci_state.json", "w") as fh:
        json.dump(
            {
                "alpha": float(aci.alpha_target),
                "gamma": 0.05,
                "warm_start_n": int(aci.n_steps),
                "empirical_coverage": float(aci.empirical_coverage),
            },
            fh,
            indent=2,
        )
    return metrics


def _aggregate(per_seed: list[dict]) -> dict:
    arr = {k: np.array([d[k] for d in per_seed], dtype=np.float64) for k in per_seed[0] if k != "seed"}
    return {
        f"{k}_mean": float(arr[k].mean()) for k in arr
    } | {f"{k}_std": float(arr[k].std(ddof=1)) for k in arr}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["f7", "f8", "f0", "f3"],
        choices=["f7", "f8", "f0", "f3"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(seeds.FROZEN_SEEDS))
    args = parser.parse_args(argv)

    summary: dict[str, dict] = {}
    for name in args.models:
        per_seed = []
        for s in args.seeds:
            print(f"--- training {name} @ seed {s}")
            m = _train_seed(name, s)
            per_seed.append(m)
            print(json.dumps(m, indent=2))
        agg = _aggregate(per_seed)
        out = REPO_ROOT / "models/forecaster" / name / "metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"per_seed": per_seed, "aggregate": agg}, fh, indent=2)
        summary[name] = {"per_seed": per_seed, "aggregate": agg}

    summary_path = REPO_ROOT / "models/forecaster/sweep_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsweep summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
