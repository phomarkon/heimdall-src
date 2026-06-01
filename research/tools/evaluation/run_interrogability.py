"""Interrogability metric — open-ended audit Q&A over a decision record, with a deterministic-query
control (the honest "where is the LLM actually needed" capability matrix).

Five question categories per run, each with ground truth computed from the trace:
  retrieval   : counts / specific values            (a deterministic query answers these EXACTLY)
  aggregate   : filters / max / "how many ..."       (a deterministic query answers these EXACTLY)
  causal      : "why did it watch / choose up at T?"  (needs NL reasoning over the rationale/record)
  counterfact : "would it have filled at limit L-30?" (needs reasoning; bespoke code only)
  unanswerable: not in the record                     (must DECLINE, not confabulate)

Scored objectively (number match / yes-no / reason-keyword / decline). Two systems:
  LLM           : Qwen3-32B answers from the serialized record.
  det-query     : a deterministic function answers retrieval/aggregate EXACTLY, and returns
                  NO_NL_CAPABILITY for causal/counterfact/unanswerable (a query engine cannot
                  produce a natural-language causal answer, a counterfactual, or a graceful decline
                  without pre-coding every question).

The result is a capability matrix: the LLM's unique value is the categories a query engine cannot
serve without anticipating each question in advance.

Usage:
  python tools/evaluation/run_interrogability.py --glob 'ai-society/runs/d3-faithfulness-20260524/d3-grounded-apr02-*-24-q32' \
      --truth-dir data/cache/evaluation_truth/april_2026 --eval-root evaluations \
      --base-url http://127.0.0.1:8003/v1 --out evaluations/interrogability.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import requests

_THINK = re.compile(r"<think>.*?</think>", re.S)


def ask(base, key, model, system, user):
    r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0, "max_tokens": 300,
                            "chat_template_kwargs": {"enable_thinking": False},
                            "messages": [{"role": "system", "content": system + " /no_think"},
                                         {"role": "user", "content": user}]}, timeout=120)
    r.raise_for_status()
    return _THINK.sub("", r.json()["choices"][0]["message"]["content"]).strip()


def has_num(text, n):
    return re.search(rf"(?<![\d.]){n}(?![\d.])", text) is not None


def ensure_eval(run_dir: Path, truth_dir: Path, eval_root: Path, context_dir: Path) -> Path:
    out = eval_root / ("io_" + run_dir.name)
    if not (out / "bid_evaluations.parquet").exists():
        import os
        e = dict(os.environ); e["PYTHONPATH"] = ".:packages/data/src:ai-society/src"
        r = subprocess.run(["uv", "run", "python", "tools/evaluation/evaluate_society_run.py",
                            "--run-dir", str(run_dir), "--context-dir", str(context_dir),
                            "--truth-dir", str(truth_dir), "--output-dir", str(out)],
                           env=e, capture_output=True, text=True)
        if not (out / "bid_evaluations.parquet").exists():
            raise SystemExit(f"eval failed for {run_dir}: {r.stderr[-400:]}")
    return out


def build(run_dir: Path, truth_dir: Path, eval_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "traces.jsonl").read_text().splitlines() if l.strip()]
    b = pd.read_parquet(eval_dir / "bid_evaluations.parquet")
    truth = pd.read_parquet(truth_dir / "activation_truth.parquet")
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    b["timestamp_utc"] = pd.to_datetime(b["timestamp_utc"], utc=True)
    b = b.merge(truth[["timestamp_utc", "zone", "activation_direction", "settlement_price_eur_mwh", "spot_price_eur_mwh"]],
                on=["timestamp_utc", "zone"], how="left")
    foc = [r for r in recs if r["agent_id"] == "agent-000"]
    # serialized record for the LLM
    lines = []
    bstat = {(r.agent_id, r.step): r.status for r in b.itertuples()}
    for r in foc:
        d = r.get("decision") or {}
        lines.append(f"tick {r['step']}: action={d.get('action')} side={d.get('side')} qty={d.get('quantity_mwh')} "
                     f"limit={d.get('limit_price_eur_mwh')} outcome={bstat.get((r['agent_id'], r['step']), d.get('action'))} "
                     f"| {(d.get('rationale') or '')[:150]}")
    record = "FOCAL P2H AGENT DECISION RECORD:\n" + "\n".join(lines)

    fb = b[b.agent_id == "agent-000"]
    n_bids = int((fb.status != "watch").sum() & 0) if False else int(fb["side"].notna().sum())
    bids_df = fb[fb["side"].notna()]
    n_bids = len(bids_df)
    sides = sorted(set(bids_df["side"].dropna()))
    n_watch_all = sum(1 for r in recs if (r.get("decision") or {}).get("action") == "watch")
    n_filled = int((fb["status"].isin(["filled", "partially_filled"])).sum())
    max_qty = float(b["quantity_mwh"].max()) if b["quantity_mwh"].notna().any() else 0.0
    # a watch tick + a bid tick for causal/counterfactual
    watch_ticks = [r["step"] for r in foc if (r.get("decision") or {}).get("action") == "watch"]
    # prefer a price_not_crossed tick (side matched activation, limit didn't cross) — most informative
    # for counterfactuals (lowering the limit flips it to clear); fall back to filled, then wrong_side.
    cf = pd.concat([bids_df[bids_df["status"] == s] for s in ("price_not_crossed", "filled", "wrong_side")]).head(1)
    cf_row = cf.iloc[0].to_dict() if len(cf) else None

    qs = []
    qs.append(dict(cat="retrieval", q="How many bids (not watches) did the focal agent submit? Answer with a number.",
                   check=lambda t, n=n_bids: has_num(t, n), det=str(n_bids)))
    qs.append(dict(cat="retrieval", q="What side(s) were the focal agent's bids on (up and/or down)?",
                   check=lambda t, s=sides: all(x in t.lower() for x in s) and not (("down" in t.lower()) and ("down" not in s)),
                   det=",".join(sides)))
    n_watch_foc = sum(1 for r in foc if (r.get("decision") or {}).get("action") == "watch")
    qs.append(dict(cat="aggregate", q="How many times did the focal agent choose to WATCH instead of bidding? Answer a number.",
                   check=lambda t, n=n_watch_foc: has_num(t, n), det=str(n_watch_foc)))
    qs.append(dict(cat="aggregate", q="How many of the focal agent's bids actually filled? Answer a number.",
                   check=lambda t, n=n_filled: has_num(t, n), det=str(n_filled)))
    if watch_ticks:
        wt = watch_ticks[len(watch_ticks) // 2]
        qs.append(dict(cat="causal", q=f"Why did the focal agent WATCH instead of bidding at tick {wt}? Explain briefly.",
                       check=lambda t: any(w in t.lower() for w in ("reject", "below threshold", "conformal", "weak", "no accepted", "uncertain", "watch", "not accepted", "negative")),
                       det="NO_NL_CAPABILITY"))
    if cf_row:
        qs.append(dict(cat="causal", q=f"At tick {int(cf_row['step'])} the focal agent bid {cf_row['side']}. What was the main driver of choosing that side? Explain briefly.",
                       check=lambda t: any(w in t.lower() for w in ("edge", "forecast", "above spot", "below spot", "up-", "spot", "interval", "regime", "profit")),
                       det="NO_NL_CAPABILITY"))
        L = float(cf_row["limit_price_eur_mwh"]); S = cf_row["settlement_price_eur_mwh"]
        if pd.notna(S):
            S = float(S); fills = (L - 30) <= S and cf_row["activation_direction"] == cf_row["side"]
            ans = "yes" if fills else "no"
            qs.append(dict(cat="counterfact",
                           q=(f"At tick {int(cf_row['step'])} the bid was {cf_row['side']} at limit {L:.1f} EUR/MWh; the market "
                              f"activated '{cf_row['activation_direction']}' and settled at {S:.1f} EUR/MWh. A bid clears only if its "
                              f"side matches the activated direction AND its limit <= settlement. Would it have cleared at a limit of "
                              f"{L-30:.1f} instead? Answer yes or no."),
                           check=lambda t, a=ans: re.search(rf"\b{a}\b", t.lower()) is not None and not re.search(rf"\b{'no' if a=='yes' else 'yes'}\b", t.lower()),
                           det="NO_NL_CAPABILITY"))
            # CF2: lower the limit to clearly below settlement; clears iff the side matched the activation
            fills2 = cf_row["activation_direction"] == cf_row["side"]  # (S-5 <= S always true)
            ans2 = "yes" if fills2 else "no"
            qs.append(dict(cat="counterfact",
                           q=(f"At tick {int(cf_row['step'])} the bid was {cf_row['side']}; the market activated "
                              f"'{cf_row['activation_direction']}' and settled at {S:.1f} EUR/MWh. Same clearing rule (side must "
                              f"match activation AND limit <= settlement). Would the bid have cleared at a limit of {S-5:.1f} "
                              f"(below settlement)? Answer yes or no."),
                           check=lambda t, a=ans2: re.search(rf"\b{a}\b", t.lower()) is not None and not re.search(rf"\b{'no' if a=='yes' else 'yes'}\b", t.lower()),
                           det="NO_NL_CAPABILITY"))
    qs.append(dict(cat="unanswerable", q="What was the measured DK1 wind-forecast error (in MW) during this run?",
                   check=lambda t: "not in record" in t.lower() or "cannot" in t.lower() or "no information" in t.lower() or "does not" in t.lower(),
                   det="NO_NL_CAPABILITY"))
    qs.append(dict(cat="unanswerable", q="What will the imbalance price be next week?",
                   check=lambda t: "not in record" in t.lower() or "cannot" in t.lower() or "future" in t.lower() or "no information" in t.lower() or "unable" in t.lower(),
                   det="NO_NL_CAPABILITY"))
    return record, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--truth-dir", type=Path, required=True)
    ap.add_argument("--context-dir", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, default=Path("evaluations"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    ap.add_argument("--key", default="heimdall-local")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    SYS = ("You are an audit assistant. Answer ONLY from the provided decision record. If the record does "
           "not contain the answer, reply exactly 'NOT IN RECORD'. Be concise.")
    runs = [Path(p) for p in glob(args.glob) if (Path(p) / "traces.jsonl").exists()]
    cats = ["retrieval", "aggregate", "causal", "counterfact", "unanswerable"]
    agg = {c: {"llm_ok": 0, "det_ok": 0, "det_na": 0, "n": 0} for c in cats}
    for run in sorted(runs):
        ev = ensure_eval(run, args.truth_dir, args.eval_root, args.context_dir)
        record, qs = build(run, args.truth_dir, ev)
        for q in qs:
            c = q["cat"]; agg[c]["n"] += 1
            # LLM
            try:
                a = ask(args.base_url, args.key, args.model, SYS, record + "\n\nQUESTION: " + q["q"])
            except Exception as e:  # noqa
                a = f"[err {e}]"
            agg[c]["llm_ok"] += int(bool(q["check"](a)))
            # deterministic-query control
            if q["det"] == "NO_NL_CAPABILITY":
                agg[c]["det_na"] += 1
            else:
                agg[c]["det_ok"] += int(bool(q["check"](q["det"])))
    print(f"\nInterrogability — {len(runs)} runs, Qwen3-32B vs deterministic-query control\n")
    print(f"{'category':<13}{'n':>4}{'LLM correct':>14}{'det-query':>22}")
    print("-" * 53)
    out = {}
    for c in cats:
        a = agg[c]; n = a["n"]
        det = f"{a['det_ok']}/{n} exact" if a["det_na"] == 0 else f"NO NL CAPABILITY ({a['det_na']}/{n})"
        llm_cell = f"{a['llm_ok']}/{n} ({100 * a['llm_ok'] / n:.0f}%)" if n else "-"
        print(f"{c:<13}{n:>4}{llm_cell:>16}{det:>24}")
        out[c] = {**a, "llm_pct": round(100 * a["llm_ok"] / n, 1) if n else None}
    args.out.write_text(json.dumps(out, indent=2))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
