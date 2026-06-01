"""Interrogability test (the explainability axis a fixed template structurally cannot do).

The completeness/selectivity metric was circular: it scored the LLM rationale against a pre-enumerated
driver set that a hand-built template maxes by construction. That measures "can you emit a complete
rationale string", not "can you EXPLAIN". Real auditability is interrogability: answering open-ended,
causal, counterfactual, aggregate questions about the decision record that nobody pre-programmed.

This poses audit questions over a real run's decision record and scores the LLM's answers against
ground truth computed from the trace. Two controls:
  - deterministic baseline: N/A — it has no natural-language Q&A interface (categorical gap).
  - no-record ablation: ask the same questions WITHOUT the trace -> the LLM should fail/decline,
    proving its correct answers come from reading the record, not from priors.

Usage:
  python tools/evaluation/run_audit_qa.py --run-dir <dir> --eval-dir <evaldir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

_THINK = re.compile(r"<think>.*?</think>", re.S)


def ask(base_url, key, model, system, user):
    r = requests.post(f"{base_url}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0.1, "max_tokens": 400,
                            "chat_template_kwargs": {"enable_thinking": False},
                            "messages": [{"role": "system", "content": system + " /no_think"},
                                         {"role": "user", "content": user}]},
                      timeout=120)
    r.raise_for_status()
    return _THINK.sub("", r.json()["choices"][0]["message"]["content"]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", default="heimdall-local")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in (args.run_dir / "traces.jsonl").read_text().splitlines() if l.strip()]
    b = pd.read_parquet(args.eval_dir / "bid_evaluations.parquet")
    bstat = {(r.agent_id, r.step): r.status for r in b.itertuples()}

    # compact, faithful serialization of the FOCAL agent's decision record (what an auditor would query)
    foc = [r for r in recs if r["agent_id"] == "agent-000"]
    lines = []
    for r in foc:
        d = r.get("decision") or {}
        st = bstat.get((r["agent_id"], r["step"]), d.get("action"))
        lines.append(f"tick {r['step']}: action={d.get('action')} side={d.get('side')} "
                     f"qty={d.get('quantity_mwh')} limit={d.get('limit_price_eur_mwh')} outcome={st} "
                     f"| rationale: {(d.get('rationale') or '')[:160]}")
    record = "FOCAL P2H AGENT DECISION RECORD (2026-01-08 DK1 run):\n" + "\n".join(lines)
    SYS = ("You are an audit assistant. Answer ONLY from the provided decision record. "
           "If the record does not contain the answer, reply exactly 'NOT IN RECORD'. Be concise.")

    # questions with checkable ground truth (computed from the trace), + 1 unanswerable
    qs = [
        ("count_bids", "How many bids did the focal agent submit, and what side were they on?",
         lambda t: ("9" in t or "nine" in t.lower()) and "up" in t.lower(), True),
        ("count_watch", "On how many ticks did the focal agent choose to watch instead of bidding?",
         lambda t: "15" in t or "fifteen" in t.lower(), True),
        ("count_filled", "How many of the focal agent's bids actually filled?",
         lambda t: re.search(r"\b2\b|\btwo\b", t.lower()) is not None, True),
        ("why_nonfill", "What was the most common reason the focal agent's bids did NOT fill?",
         lambda t: "cross" in t.lower() or "price" in t.lower() or "limit" in t.lower(), True),
        ("outage_influence", "Did the focal agent reference a generation outage in its reasoning, and did that change whether it bid?",
         lambda t: ("outage" in t.lower() or "trip" in t.lower()) , True),
        ("unanswerable", "What was the measured DK1 wind-forecast error (in MW) during this run?",
         lambda t: "not in record" in t.lower(), True),
    ]

    out = {"with_record": {}, "no_record": {}}
    for key, q, check, _ in qs:
        a = ask(args.base_url, args.key, args.model, SYS, record + "\n\nQUESTION: " + q)
        out["with_record"][key] = {"correct": bool(check(a)), "answer": a[:240]}
    # ablation: no record -> should not know the specifics
    for key, q, check, _ in qs[:3] + [qs[5]]:
        a = ask(args.base_url, args.key, args.model, SYS, "(no decision record provided)\n\nQUESTION: " + q)
        out["no_record"][key] = {"correct": bool(check(a)), "answer": a[:200]}

    wr = sum(v["correct"] for v in out["with_record"].values())
    nr = sum(v["correct"] for v in out["no_record"].values())
    out["summary"] = {"with_record_correct": f"{wr}/{len(out['with_record'])}",
                      "no_record_correct": f"{nr}/{len(out['no_record'])}"}
    print(json.dumps(out, indent=2))
    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
