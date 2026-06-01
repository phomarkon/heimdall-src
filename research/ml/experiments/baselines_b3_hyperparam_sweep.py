"""B3 hyperparameter sweep — close the gap with B2.

The default B3 (4-h horizon, S=24, cvar_lambda=1.0) reports
7.94 EUR/MWh vs B2's 17.87.  Reviewer-2 will ask why our
re-implementation underperforms a one-step LP.  This sweep varies
horizon, scenarios, and cvar_lambda to find the configuration that
either matches B2 (suggesting our LP is correctly tuned and the H&H
asset mismatch explains the gap) or stays below B2 across the board
(in which case our LP is structurally limited for the P2H setup).

Outputs:
  experiments/outputs/baselines_b3_hyperparam_sweep.json
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from experiments.baselines_focal_agent import P2HSpec, load_panel
from experiments.baselines_b3_hagstrom_herre import B3Config, evaluate_b3

REPO_ROOT = Path(__file__).resolve().parents[2]


def run() -> dict:
    panel = load_panel("val")
    spec = P2HSpec()
    # Compact grid: 12 cells. Each ~25s → ~5 min total.
    grid = list(product(
        [4, 8, 16, 32],          # horizon_ticks (1h..8h)
        [24],                     # n_scenarios fixed
        [0.0, 0.5, 1.0],          # cvar_lambda
    ))
    print(f"Sweeping {len(grid)} configs...")
    results = []
    for h, s, cl in grid:
        cfg = B3Config(seed=42, horizon_ticks=h, n_scenarios=s, cvar_lambda=cl)
        r = evaluate_b3(panel, spec, cfg).to_json()
        rec = {
            "horizon_ticks": h, "n_scenarios": s, "cvar_lambda": cl,
            "profit_per_mwh_eur": r["profit_per_mwh_eur"],
            "total_profit_eur": r["total_profit_eur"],
            "sharpe": r["sharpe"],
            "cvar5_daily_eur": r["cvar5_daily_eur"],
            "physical_violation_rate": r["physical_violation_rate"],
            "runtime_seconds": r["runtime_seconds"],
        }
        results.append(rec)
        print(f"  H={h:3d} S={s:2d} λ={cl:.1f} → {rec['profit_per_mwh_eur']:6.2f} EUR/MWh "
              f"Sharpe={rec['sharpe']:.1f} viol={rec['physical_violation_rate']:.2f}")

    best = max(results, key=lambda r: r["profit_per_mwh_eur"])
    out = {
        "n_configs": len(grid),
        "best_profit_per_mwh_eur": best["profit_per_mwh_eur"],
        "best_config": best,
        "all_results": results,
        "b2_anchor_profit_per_mwh_eur": 17.87,  # B2 reference from earlier 5-seed sweep
    }
    out_path = REPO_ROOT / "experiments/outputs/baselines_b3_hyperparam_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nBest: {best['profit_per_mwh_eur']:.2f} EUR/MWh "
          f"(H={best['horizon_ticks']}, S={best['n_scenarios']}, λ={best['cvar_lambda']})")
    print(f"B2 anchor: 17.87 EUR/MWh — gap {17.87 - best['profit_per_mwh_eur']:+.2f}")
    return out


if __name__ == "__main__":
    run()
