"""D2 delivery-risk: Monte-Carlo CVaR of delivery-adjusted profit under GROUNDED availability.

Creates the genuine two-sided downside the base simulator lacks (see
project-no-genuine-downside): each filled bid on a variable-output asset (wind/renewables/ev)
may under-deliver; the undelivered MWh is re-settled at the real imbalance price. Realized
availability is drawn around the run's own decision-time expected share a* with the dispersion
GROUNDED in real April forecast error (tools/evaluation/availability_real.py: wind CV 0.29,
renewables 0.26), not an invented number. We Monte-Carlo many availability scenarios and report
the mean delivery-adjusted profit AND its 5% CVaR (the tail a risk-averse desk actually fears).

This makes the risk axis MEASURABLE. Whether a *risk-aware LLM* beats a *risk-neutral* policy on
CVaR is the value test; on existing (non-risk-aware) runs this quantifies each arm's baseline tail
exposure and validates the substrate (CV=0 -> zero penalty, recovering the status quo exactly).

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_delivery_risk.py \
        --glob 'ai-society/runs/llm-value-allarch*/**/' --scenarios 500 \
        --json-out evaluations/delivery_risk_allarch.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.availability_real import grounded_cv_for_archetype
from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids
from tools.evaluation.rescore_runs import (
    _availability_by_archetype,
    _shortfall_settlement,
    _tick_truth,
)

FILLED = {"filled", "partially_filled"}


def _arm(run_dir: Path) -> str:
    try:
        s = json.loads((run_dir / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        s = {}
    if not s.get("llm_enabled", False):
        return "det"
    abl = str(s.get("ablation_strategy") or s.get("chooser_mode") or "llm")
    # surface the recognisable short arm tags used in the allarch matrix run_ids
    name = run_dir.name
    for tag in ("det_rich", "selector_rich", "selector", "cp12", "cp13", "det"):
        if f"-{tag}-" in name:
            return tag
    return abl


def _cvar(x: np.ndarray, q: float = 0.05) -> float:
    if len(x) == 0:
        return float("nan")
    k = max(1, int(len(x) * q))
    return float(np.sort(x)[:k].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    run_dirs = [Path(r) for r in args.runs]
    for g in args.glob:
        run_dirs += [Path(p) for p in glob(g, recursive=True)]
    run_dirs = sorted({d for d in run_dirs if (d / "traces.jsonl").exists() and (d / "summary.json").exists()})
    if not run_dirs:
        raise SystemExit("no runs")

    per_arm: dict[str, dict] = {}
    for d in run_dirs:
        arm = _arm(d)
        traces = _load_traces(d / "traces.jsonl")
        if traces.empty:
            continue
        bids = _score_bids(traces, truth)
        if bids.empty:
            continue
        a_star = _availability_by_archetype(d / "traces.jsonl")  # archetype -> expected share
        run_seed = int(traces["run_id"].iloc[0].__hash__() & 0xFFFF) if "run_id" in traces else 0

        # VARIABLE-ONLY scope: the risk axis lives on availability-gated assets. Including firm P2H
        # upside (which dominates) masks the variable shortfall, so upside is restricted to variable
        # archetypes — the same set that carries shortfall. Firm-asset profit is out of scope here.
        is_var = bids["archetype"].astype(str).str.lower().map(lambda a: grounded_cv_for_archetype(a) > 0.0)
        upside = float(pd.to_numeric(bids.loc[is_var, "realized_profit_eur"], errors="coerce").fillna(0.0).sum())

        # collect filled variable-asset bids with their tick truth
        jobs = []
        for (ts, zone), tick in bids.groupby(["timestamp_utc", "zone"], sort=False):
            tr = truth[(truth["timestamp_utc"] == ts) & (truth["zone"] == zone)]
            direction, volume, spread, settlement, imbalance = _tick_truth(tr)
            if direction == "neutral":
                continue
            for _, b in tick[tick["status"].isin(FILLED)].iterrows():
                arch = str(b.get("archetype")).lower()
                cv = grounded_cv_for_archetype(arch)
                a = float(a_star.get(arch, 1.0))
                cleared = float(pd.to_numeric(b.get("cleared_mwh"), errors="coerce") or 0.0)
                if cv <= 0.0 or a <= 0.0 or a >= 1.0 or cleared <= 0.0:
                    continue
                jobs.append((arch, a, cv, str(b["side"]), cleared, settlement, imbalance))

        # Monte-Carlo availability scenarios
        rng = np.random.default_rng(run_seed)
        totals = np.full(args.scenarios, upside, dtype=float)
        for (arch, a, cv, side, cleared, settlement, imbalance) in jobs:
            kappa = max(1e-3, (1.0 - a) / (a * cv * cv) - 1.0)
            rhos = rng.beta(a * kappa, (1.0 - a) * kappa, size=args.scenarios)
            for s in range(args.scenarios):
                sf, penalty = _shortfall_settlement(side=side, cleared=cleared, a_star=a,
                                                    rho=float(rhos[s]), settlement=settlement, imbalance=imbalance)
                totals[s] -= penalty

        agg = per_arm.setdefault(arm, {"runs": 0, "upside": 0.0, "n_var_bids": 0,
                                       "mean_da_profit": 0.0, "totals": []})
        agg["runs"] += 1
        agg["upside"] += upside
        agg["n_var_bids"] += len(jobs)
        agg["mean_da_profit"] += float(totals.mean())
        agg["totals"].append(totals)

    out = {}
    print(f"\nD2 delivery-risk (grounded availability, {args.scenarios} MC scenarios)\n")
    hdr = f"{'arm':<16}{'runs':>5}{'var_bids':>9}{'upside_eur':>12}{'mean_DA_eur':>12}{'CVaR5_eur':>11}{'shortfall%':>11}"
    print(hdr); print("-" * len(hdr))
    for arm in sorted(per_arm):
        a = per_arm[arm]
        pooled = np.concatenate(a["totals"]) if a["totals"] else np.array([])
        cvar5 = _cvar(pooled)
        mean_da = float(pooled.mean()) if len(pooled) else float("nan")
        shortfall_pct = (a["upside"] - mean_da * a["runs"]) / a["upside"] * 100 if a["upside"] > 0 else 0.0
        print(f"{arm:<16}{a['runs']:>5}{a['n_var_bids']:>9}{a['upside']:>12.0f}"
              f"{mean_da:>12.1f}{cvar5:>11.1f}{shortfall_pct:>10.1f}%")
        out[arm] = {"runs": a["runs"], "n_var_bids": a["n_var_bids"], "upside_eur": round(a["upside"], 1),
                    "mean_da_profit_per_run": round(mean_da, 2), "cvar5_per_run": round(cvar5, 2),
                    "shortfall_pct_of_upside": round(shortfall_pct, 2)}
    print("\nDA = delivery-adjusted profit per run (upside - imbalance-priced shortfall). "
          "CVaR5 = mean of worst 5% scenarios. Variable assets only carry shortfall; firm "
          "(p2h/generator) are inert by construction.\n")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
