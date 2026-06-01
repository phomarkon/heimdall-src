"""Summarize the D6 fill-rate matrix: can the LLM get higher fill at similar profit?

Three arms select from the IDENTICAL accepted-candidate menu (fair, menu-only):
  detbest     = deterministic best (max worst-case-profit)  -> low fill, high per-bid margin
  dethighfill = deterministic max clear-probability         -> high fill, low margin
  llmfill     = llm_fill_selector                            -> LLM balances fill vs margin

Positive = llmfill dominates the deterministic frontier: fill rate >> detbest AND total profit not
worse than the better of {detbest, dethighfill}. Reports the full fill/profit tradeoff so the reader
can see where the LLM lands relative to the two deterministic extremes.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/summarize_d6_fill.py \
        --glob 'ai-society/runs/d6-fill-20260524/d6-*-24-q32' --json-out evaluations/d6_fill_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import pandas as pd

from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids

SUB = {"filled", "partially_filled", "wrong_side", "price_not_crossed"}
FILLED = {"filled", "partially_filled"}


def _arm(name: str) -> str:
    for a in ("detbest", "dethighfill", "llmfill"):
        if f"-{a}-" in name:
            return a
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    dirs = []
    for g in args.glob:
        dirs += [Path(p) for p in glob(g)]
    dirs = sorted({d for d in dirs if (d / "traces.jsonl").exists()})

    agg = defaultdict(lambda: {"runs": 0, "sub": 0, "fill": 0, "profit": 0.0, "cleared": 0.0})
    for d in dirs:
        arm = _arm(d.name)
        traces = _load_traces(d / "traces.jsonl")
        if traces.empty:
            continue
        bids = _score_bids(traces, truth)
        sub = bids[bids["status"].isin(SUB)]
        if sub.empty:
            continue
        a = agg[arm]
        a["runs"] += 1
        a["sub"] += len(sub)
        a["fill"] += int(sub["status"].isin(FILLED).sum())
        a["profit"] += float(pd.to_numeric(sub["realized_profit_eur"], errors="coerce").fillna(0).sum())
        a["cleared"] += float(pd.to_numeric(bids["cleared_mwh"], errors="coerce").fillna(0).sum())

    out = {}
    print(f"\nD6 fill-rate: deterministic frontier vs LLM — {len(dirs)} runs\n")
    print(f"{'arm':<13}{'runs':>5}{'submitted':>10}{'fill_rate':>10}{'profit/run':>11}{'profit/sub_bid':>15}{'profit/fill':>12}")
    for arm in ("detbest", "dethighfill", "llmfill"):
        if arm not in agg:
            continue
        a = agg[arm]
        fr = a["fill"] / a["sub"] if a["sub"] else float("nan")
        ppr = a["profit"] / a["runs"] if a["runs"] else float("nan")
        pps = a["profit"] / a["sub"] if a["sub"] else float("nan")
        ppf = a["profit"] / a["fill"] if a["fill"] else float("nan")
        print(f"{arm:<13}{a['runs']:>5}{a['sub']:>10}{fr:>10.3f}{ppr:>11.1f}{pps:>15.2f}{ppf:>12.2f}")
        out[arm] = {"runs": a["runs"], "submitted": a["sub"], "fill_rate": round(fr, 4),
                    "profit_per_run": round(ppr, 2), "profit_per_sub_bid": round(pps, 3)}
    if {"detbest", "llmfill"} <= out.keys():
        db, lf = out["detbest"], out["llmfill"]
        print(f"\nHEADLINE: fill_rate detbest {db['fill_rate']} -> llmfill {lf['fill_rate']}; "
              f"profit/run {db['profit_per_run']} -> {lf['profit_per_run']}")
        if "dethighfill" in out:
            print(f"  (det_high_fill control: fill {out['dethighfill']['fill_rate']}, profit/run {out['dethighfill']['profit_per_run']})")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
