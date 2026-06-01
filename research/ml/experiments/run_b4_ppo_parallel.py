"""Parallelise B4 PPO training across the five frozen seeds.

PPO with SB3 MlpPolicy is faster on CPU than GPU for our tiny actor-critic
(SB3 issue #1245).  But running five seeds sequentially burns wall time;
running them as concurrent multiprocessing workers lets all cores do work
in parallel and cuts the 200k-steps × 5-seeds wall from ~45 min to
~10–12 min on a many-core box.

Outputs land in ``experiments/outputs/focal_agent_baselines/`` exactly
where the sequential driver wrote them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments.baselines_focal_agent import (
    B4Config,
    P2HSpec,
    b4_train_ppo,
    evaluate_policy,
    load_panel,
    make_b4_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SEEDS = (13, 42, 137, 1729, 31415)
OUT_DIR = REPO_ROOT / "experiments/outputs/focal_agent_baselines"


def _run_one_seed(seed: int, total_timesteps: int) -> dict:
    cfg = B4Config(seed=seed, total_timesteps=total_timesteps)
    train_panel = load_panel("train")
    val_panel = load_panel("val")
    spec = P2HSpec()
    model = b4_train_ppo(cfg, train_panel, spec)
    r = evaluate_policy(make_b4_policy(model), name="b4_ppo", seed=seed,
                        panel=val_panel, spec=spec)
    payload = r.to_json()
    payload["total_timesteps"] = total_timesteps
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"b4_ppo_seed{seed}.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--seeds", type=int, nargs="*", default=list(FROZEN_SEEDS))
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.starmap(
                _run_one_seed,
                [(s, args.total_timesteps) for s in args.seeds],
            )
    else:
        results = [_run_one_seed(s, args.total_timesteps) for s in args.seeds]
    elapsed = time.perf_counter() - t0

    keys = ["total_profit_eur", "profit_per_mwh_eur", "sharpe", "cvar5_daily_eur",
            "participation_rate", "physical_violation_rate"]
    agg = {
        k: {
            "mean": float(np.mean([r[k] for r in results])),
            "std": float(np.std([r[k] for r in results], ddof=1)) if len(results) > 1 else 0.0,
            "values": [r[k] for r in results],
        }
        for k in keys
    }
    summary = {
        "experiment": "b4_ppo_parallel",
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "wall_seconds": round(elapsed, 1),
        "aggregate": agg,
    }
    (OUT_DIR / f"b4_ppo_parallel_summary_{args.total_timesteps}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
