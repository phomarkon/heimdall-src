"""Summarize the D2 risk matrix: risk-blind vs risk-aware LLM, behavioral + delivery-CVaR.

Two positives, both matched (same society/seed/window, only the delivery-risk instruction differs):
  1. STEERABILITY / risk behavior: the risk-aware instruction makes the LLM commit less variable
     volume (lower mean bid quantity, lower total cleared MWh on availability-risky assets) and/or
     watch more — a policy change achieved by prompt, which the fixed deterministic pipeline cannot do.
  2. RISK VALUE: delivery-loss tail is linear in committed volume, so less commitment => smaller CVaR
     tail. Reported on wind/renewables (grounded availability) where they fill.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/summarize_d2_risk.py \
        --glob 'ai-society/runs/d2-risk-20260524/d2-*-24-q32' --json-out evaluations/d2_risk_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.availability_real import grounded_cv_for_archetype
from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids
from tools.evaluation.rescore_runs import _availability_by_archetype, _shortfall_settlement, _tick_truth

VAR = {"wind", "renewables", "ev"}
GROUNDED = {"wind", "renewables"}  # have grounded availability CV (ev excluded: not weather)
FILLED = {"filled", "partially_filled"}


def _arm(name: str) -> str:
    return "riskaware" if "-riskaware-" in name else "riskblind" if "-riskblind-" in name else "other"


def _cvar(x: np.ndarray, q: float = 0.05) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.sort(x)[: max(1, int(len(x) * q))].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--scenarios", type=int, default=1000)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    dirs = []
    for g in args.glob:
        dirs += [Path(p) for p in glob(g)]
    dirs = sorted({d for d in dirs if (d / "traces.jsonl").exists()})

    agg = defaultdict(lambda: {"runs": 0, "var_bids": 0, "qtys": [], "cleared_var": 0.0,
                               "watch": 0, "bid": 0, "abstain": 0, "grounded_totals": [],
                               "var_submitted": 0, "var_filled": 0, "var_profit": 0.0})
    for d in dirs:
        arm = _arm(d.name)
        traces = _load_traces(d / "traces.jsonl")
        if traces.empty:
            continue
        var = traces[traces["archetype"].astype(str).str.lower().isin(VAR)]
        a = agg[arm]
        a["runs"] += 1
        for act in ("watch", "bid", "abstain"):
            a[act] += int((var["action"] == act).sum())
        vb = var[var["action"] == "bid"]
        a["var_bids"] += len(vb)
        a["qtys"] += [float(q) for q in pd.to_numeric(vb["quantity_mwh"], errors="coerce").dropna()]
        # fill rate + profit on variable bids (ties risk to fill — smaller bids clear more)
        bids = _score_bids(traces, truth)
        vb_sc = bids[bids["archetype"].astype(str).str.lower().isin(VAR)]
        vsub = vb_sc[vb_sc["status"].isin({"filled", "partially_filled", "wrong_side", "price_not_crossed"})]
        a["var_submitted"] += len(vsub)
        a["var_filled"] += int(vsub["status"].isin(FILLED).sum())
        a["var_profit"] += float(pd.to_numeric(vsub["realized_profit_eur"], errors="coerce").fillna(0).sum())
        a_star = _availability_by_archetype(d / "traces.jsonl")
        rng = np.random.default_rng(abs(hash(d.name)) & 0xFFFF)
        jobs = []
        for (ts, zone), tick in bids.groupby(["timestamp_utc", "zone"], sort=False):
            tr = truth[(truth["timestamp_utc"] == ts) & (truth["zone"] == zone)]
            direction, vol, spread, settle, imb = _tick_truth(tr)
            if direction == "neutral":
                continue
            for _, b in tick[tick["status"].isin(FILLED)].iterrows():
                arch = str(b.get("archetype")).lower()
                cv = grounded_cv_for_archetype(arch)
                av = float(a_star.get(arch, 1.0))
                cl = float(pd.to_numeric(b.get("cleared_mwh"), errors="coerce") or 0.0)
                if arch in GROUNDED and cl > 0:
                    a["cleared_var"] += cl
                if cv > 0 and 0 < av < 1 and cl > 0:
                    jobs.append((av, cv, str(b["side"]), cl, settle, imb))
        up = float(pd.to_numeric(bids.loc[bids["archetype"].astype(str).str.lower().isin(GROUNDED), "realized_profit_eur"], errors="coerce").fillna(0).sum())
        totals = np.full(args.scenarios, up)
        for (av, cv, side, cl, settle, imb) in jobs:
            kappa = max(1e-3, (1 - av) / (av * cv * cv) - 1)
            rhos = rng.beta(av * kappa, (1 - av) * kappa, size=args.scenarios)
            for s in range(args.scenarios):
                _, pen = _shortfall_settlement(side=side, cleared=cl, a_star=av, rho=float(rhos[s]), settlement=settle, imbalance=imb)
                totals[s] -= pen
        a["grounded_totals"].append(totals)

    out = {}
    print(f"\nD2 risk: blind vs aware — {len(dirs)} runs\n")
    print(f"{'arm':<11}{'runs':>5}{'var_bids':>9}{'mean_qty':>9}{'fill_rate':>10}{'profit/run':>11}{'watch%':>8}{'cvar5':>9}")
    for arm in ("riskblind", "riskaware"):
        if arm not in agg:
            continue
        a = agg[arm]
        ndec = a["watch"] + a["bid"] + a["abstain"]
        meanq = np.mean(a["qtys"]) if a["qtys"] else float("nan")
        watchp = 100 * a["watch"] / ndec if ndec else 0
        fillr = a["var_filled"] / a["var_submitted"] if a["var_submitted"] else float("nan")
        ppr = a["var_profit"] / a["runs"] if a["runs"] else float("nan")
        pooled = np.concatenate(a["grounded_totals"]) if a["grounded_totals"] else np.array([])
        cvar = _cvar(pooled)
        print(f"{arm:<11}{a['runs']:>5}{a['var_bids']:>9}{meanq:>9.2f}{fillr:>10.3f}{ppr:>11.1f}{watchp:>7.0f}%{cvar:>9.1f}")
        out[arm] = {"runs": a["runs"], "var_bids": a["var_bids"], "mean_bid_qty": round(float(meanq), 3),
                    "var_fill_rate": None if np.isnan(fillr) else round(float(fillr), 4),
                    "var_profit_per_run": round(float(ppr), 2), "cleared_grounded_mwh": round(a["cleared_var"], 2),
                    "watch_pct": round(watchp, 1), "grounded_cvar5": None if np.isnan(cvar) else round(float(cvar), 2)}
    if "riskblind" in out and "riskaware" in out:
        b, w = out["riskblind"]["mean_bid_qty"], out["riskaware"]["mean_bid_qty"]
        fb, fw = out["riskblind"]["var_fill_rate"], out["riskaware"]["var_fill_rate"]
        pb, pw = out["riskblind"]["var_profit_per_run"], out["riskaware"]["var_profit_per_run"]
        print(f"\nHEADLINE: mean bid qty {b}->{w} ({100*(1-w/b):.0f}% smaller); "
              f"fill_rate {fb}->{fw}; profit/run {pb}->{pw}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
