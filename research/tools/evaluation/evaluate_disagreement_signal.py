"""Cross-agent disagreement as an emergent risk signal (society value).

A single deterministic policy emits one decision per tick; a heterogeneous LLM society emits many.
If the *spread* of those decisions (side-split among bidders, action entropy) predicts when the tick
goes badly (wrong-side, no-fill, loss), then the society produces a calibrated risk/uncertainty signal
that no single policy can — a concrete positive for the multi-agent design.

Test: bin ticks by cross-agent disagreement, report realized outcomes per bin. A monotone "more
disagreement → worse outcome" is the positive. Clean control: a SINGLE-agent (or deterministic) run
has no disagreement signal by construction. Realized-outcome join via _score_bids over activation truth.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_disagreement_signal.py \
        --glob 'ai-society/runs/**/' --min-agents 4 --json-out evaluations/disagreement_signal.json
"""

from __future__ import annotations

import argparse
import json
import math
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids

SUBMITTED = {"filled", "partially_filled", "wrong_side", "price_not_crossed"}
FILLED = {"filled", "partially_filled"}


def _entropy(counts: list[int]) -> float:
    n = sum(counts)
    if n == 0:
        return 0.0
    ps = [c / n for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps)) if len(ps) > 1 else 0.0  # normalised 0..1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--min-agents", type=int, default=4, help="min agents deciding at a tick to score it")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    run_dirs = [Path(r) for r in args.runs]
    for g in args.glob:
        run_dirs += [Path(p) for p in glob(g, recursive=True)]
    run_dirs = sorted({d for d in run_dirs if (d / "traces.jsonl").exists()})

    ticks = []  # one row per (run, step): disagreement + realized outcome
    for d in run_dirs:
        traces = _load_traces(d / "traces.jsonl")
        if traces.empty or "action" not in traces.columns:
            continue
        bids = _score_bids(traces, truth)
        if bids.empty:
            continue
        # disagreement is computed over the DECISIONS (traces); outcomes over the scored bids
        for (run_id, step), tdec in traces.groupby(["run_id", "step"]):
            actions = tdec["action"].astype(str).tolist()
            n_dec = len(actions)
            if n_dec < args.min_agents:
                continue
            bidders = tdec[tdec["action"] == "bid"]
            n_bid = len(bidders)
            # side-split among bidders (0 = consensus, 1 = even split)
            up = int((bidders["side"] == "up").sum()) if "side" in bidders else 0
            down = int((bidders["side"] == "down").sum()) if "side" in bidders else 0
            side_split = _entropy([up, down])
            # action entropy across bid/watch/abstain
            act_split = _entropy([actions.count("bid"), actions.count("watch"), actions.count("abstain")])
            # realized outcome for this tick's scored bids
            sb = bids[(bids["run_id"] == run_id) & (bids["step"] == step)]
            sub = sb[sb["status"].isin(SUBMITTED)]
            if sub.empty:
                continue
            wrong = float((sub["status"] == "wrong_side").mean())
            fill = float(sub["status"].isin(FILLED).mean())
            prof = float((pd.to_numeric(sub["realized_profit_eur"], errors="coerce") > 0).mean())
            ticks.append({"side_split": side_split, "act_split": act_split, "n_bid": n_bid,
                          "wrong_side_rate": wrong, "fill_rate": fill, "p_profitable": prof,
                          "n_submitted": len(sub)})

    if not ticks:
        raise SystemExit("no multi-agent ticks scored")
    df = pd.DataFrame(ticks)
    print(f"\nDisagreement signal — {len(df)} multi-agent ticks (>= {args.min_agents} agents)\n")

    out = {"n_ticks": len(df), "bins": {}, "correlations": {}}
    for sig in ("side_split", "act_split"):
        # bin into consensus / mixed / high-disagreement
        try:
            df["_bin"] = pd.qcut(df[sig], q=[0, 0.5, 0.8, 1.0], labels=["low", "mid", "high"], duplicates="drop")
        except ValueError:
            df["_bin"] = pd.cut(df[sig], bins=[-0.01, 0.01, 0.5, 1.01], labels=["low", "mid", "high"])
        print(f"== signal: {sig} ==")
        print(f"  {'bin':<6}{'n':>6}{'wrong_side':>12}{'fill_rate':>11}{'P(profit)':>11}")
        g = df.groupby("_bin", observed=True)
        for b, grp in g:
            print(f"  {str(b):<6}{len(grp):>6}{grp['wrong_side_rate'].mean():>12.3f}"
                  f"{grp['fill_rate'].mean():>11.3f}{grp['p_profitable'].mean():>11.3f}")
        # Spearman of signal vs wrong-side (positive = disagreement predicts wrong-side)
        rho = df[sig].corr(df["wrong_side_rate"], method="spearman")
        print(f"  Spearman({sig}, wrong_side_rate) = {rho:+.3f}\n")
        out["correlations"][sig] = {"spearman_wrong_side": None if pd.isna(rho) else round(float(rho), 4)}
        out["bins"][sig] = {str(b): {"n": int(len(grp)), "wrong_side": round(float(grp["wrong_side_rate"].mean()), 4),
                                     "fill_rate": round(float(grp["fill_rate"].mean()), 4),
                                     "p_profitable": round(float(grp["p_profitable"].mean()), 4)}
                            for b, grp in g}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
