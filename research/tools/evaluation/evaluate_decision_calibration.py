"""D4 decision calibration — are the LLM's self-reported confidence / risk_label /
priority_score calibrated against realized outcomes?

This is the "LLM as risk-monitor / operator co-pilot" value axis. The deterministic pipeline
emits no calibrated risk signal; if the LLM's labels predict which bids lose money or fill, that
is value the deterministic core cannot provide. If they don't (e.g. confidence pinned at ~0.85
regardless of outcome), that is an honest negative.

Reuses evaluate_society_run's realized-outcome join (_score_bids over activation truth), then
merges the decision fields from the trace on (run_id, agent_id, step).

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_decision_calibration.py \
        --truth-dir data/cache/evaluation_truth/april_2026 \
        ai-society/runs/llm-value-allarch-20260523/* ai-society/runs/llm-value-allarch-rich-20260523/*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids

SUBMITTED = {"filled", "partially_filled", "wrong_side", "price_not_crossed"}
FILLED = {"filled", "partially_filled"}
CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def _decisions(traces: pd.DataFrame) -> pd.DataFrame:
    # _load_traces flattens the LLMBidDecision into top-level columns.
    cols = ["run_id", "agent_id", "step", "confidence", "risk_label", "uncertainty_label", "priority_score"]
    have = [c for c in cols if c in traces.columns]
    return traces[have].copy()


def _arm(run_dir: Path) -> str:
    parts = run_dir.name.split("-")
    return parts[1] if len(parts) > 1 else run_dir.name


def _ece(conf: np.ndarray, outcome: np.ndarray) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(outcome[m].mean() - conf[m].mean())
    return ece


def _cvar5(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    k = max(1, int(len(x) * 0.05))
    return float(np.sort(x)[:k].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    run_dirs = [Path(r) for r in args.runs if (Path(r) / "traces.jsonl").exists()]

    per_arm: dict[str, list[pd.DataFrame]] = {}
    for d in sorted(run_dirs):
        traces = _load_traces(d / "traces.jsonl")
        bids = _score_bids(traces, truth)
        dec = _decisions(traces)
        merged = bids.merge(dec, on=["run_id", "agent_id", "step"], how="left", suffixes=("", "_dec"))
        per_arm.setdefault(_arm(d), []).append(merged)

    out = {}
    print("\nD4 decision calibration vs realized outcome\n")
    for arm in sorted(per_arm):
        df = pd.concat(per_arm[arm], ignore_index=True)
        sub = df[df["status"].isin(SUBMITTED)].copy()
        n_sub = len(sub)
        if n_sub == 0:
            print(f"### {arm}: no submitted bids\n")
            continue
        sub["profitable"] = (sub["realized_profit_eur"] > 0).astype(float)
        sub["filled"] = sub["status"].isin(FILLED).astype(float)
        conf = pd.to_numeric(sub["confidence"], errors="coerce").to_numpy()
        prof = sub["profitable"].to_numpy()
        valid = ~np.isnan(conf)
        ece = _ece(conf[valid], prof[valid])

        print(f"### {arm}  (submitted={n_sub}, filled={int(sub['filled'].sum())}, "
              f"P(profitable)={prof.mean():.2f}, conf mean={np.nanmean(conf):.2f} std={np.nanstd(conf):.2f})")
        print(f"    confidence ECE vs P(profitable): {ece:.3f}  "
              f"({'pinned/uninformative' if np.nanstd(conf) < 0.05 else 'has spread'})")

        # risk_label discrimination
        rk = sub.groupby("risk_label")["realized_profit_eur"].agg(["count", "mean"])
        rk["cvar5"] = sub.groupby("risk_label")["realized_profit_eur"].apply(lambda x: _cvar5(x.to_numpy()))
        rk["P_profit"] = sub.groupby("risk_label")["profitable"].mean()
        if len(rk):
            print("    risk_label   n   mean€   cvar5€   P(profit)")
            for lbl, r in rk.iterrows():
                print(f"      {str(lbl):<9}{int(r['count']):>4}{r['mean']:>8.1f}{r['cvar5']:>9.1f}{r['P_profit']:>10.2f}")
        # does risk=high actually carry more downside than risk!=high?
        hi = sub[sub["risk_label"] == "high"]["realized_profit_eur"].to_numpy()
        lo = sub[sub["risk_label"] != "high"]["realized_profit_eur"].to_numpy()
        if len(hi) and len(lo):
            verdict = "HIGHER downside (calibrated)" if _cvar5(hi) < _cvar5(lo) else "NOT higher downside (miscalibrated)"
            print(f"    risk=high cvar5 {_cvar5(hi):.1f} vs risk!=high cvar5 {_cvar5(lo):.1f} -> {verdict}")

        # priority_score: do loss/false-accept bids get higher priority? (operator-triage value)
        ps = pd.to_numeric(sub["priority_score"], errors="coerce")
        loss = sub["realized_profit_eur"] < 0
        if ps.notna().any() and loss.any() and (~loss).any():
            print(f"    priority_score: loss bids mean={ps[loss].mean():.2f} vs profit bids mean={ps[~loss].mean():.2f} "
                  f"({'flags losses' if ps[loss].mean() > ps[~loss].mean() else 'does NOT flag losses'})")

        # side prediction accuracy among submitted with a side
        sided = sub[sub["side"].notna()]
        if len(sided):
            acc = (sided["status"] != "wrong_side").mean()
            print(f"    side accuracy (submitted, non-wrong-side): {acc:.2f} on n={len(sided)}")
        print()
        out[arm] = {"submitted": int(n_sub), "conf_ece": ece, "conf_std": float(np.nanstd(conf)),
                    "P_profitable": float(prof.mean())}

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
