"""Interrogability WITH a leak-safe RAG query tool (2026-05-25) — tests the capstone's recommendation.

The interrogability capstone found the LLM strong on causal/refusal (100%) but weak on retrieval
(3/6) and aggregate (4/6) — "the LLM miscounts ±1; pair it with a query tool". This script tests that
fix directly: it re-uses the EXACT questions, ground truth, and checks from run_interrogability.build(),
and adds a third arm —

  LLM            : answers from the serialized record (the capstone baseline), re-measured here.
  LLM + RAG      : same model, but a per-run RAG (retrieve-then-read) over a corpus of per-tick decision
                   records + a precomputed run-statistics card is prepended to each question.
  det-query      : deterministic control (exact on retrieval/aggregate, NO_NL_CAPABILITY otherwise).

The per-run corpus is built from the run's own trace/eval (no external facts), reusing the
RAGRetriever from heimdall_ai_society.rag (as_of=None — post-hoc audit, no temporal cutoff needed).

Usage (point at runs with a focal agent-000; pin to the idle :8003):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:ai-society/src uv run python tools/evaluation/run_interrogability_rag.py \
    --glob 'ai-society/runs/ff-rag-20260525/ff-rag-20260525-det-*-24-q32' \
    --truth-dir data/cache/evaluation_truth/april_2026 \
    --context-dir data/cache/real_context/april_2026 --out evaluations/interrogability_rag.json
"""
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd

from heimdall_ai_society.rag import RAGRetriever, RagDocument
from tools.evaluation.run_interrogability import ask, build, ensure_eval


def run_facts(run_dir: Path, eval_dir: Path) -> dict:
    """Recompute the focal-agent run statistics (same formulas as run_interrogability.build)."""
    recs = [json.loads(l) for l in (run_dir / "traces.jsonl").read_text().splitlines() if l.strip()]
    foc = [r for r in recs if r["agent_id"] == "agent-000"]
    b = pd.read_parquet(eval_dir / "bid_evaluations.parquet")
    fb = b[b.agent_id == "agent-000"]
    bids_df = fb[fb["side"].notna()]
    n_bids = len(bids_df)
    sides = sorted(set(bids_df["side"].dropna()))
    n_filled = int((fb["status"].isin(["filled", "partially_filled"])).sum())
    n_watch_foc = sum(1 for r in foc if (r.get("decision") or {}).get("action") == "watch")
    max_qty = float(b["quantity_mwh"].max()) if b["quantity_mwh"].notna().any() else 0.0
    return {"n_bids": n_bids, "sides": sides, "n_filled": n_filled,
            "n_watch": n_watch_foc, "max_qty": max_qty}


def build_run_corpus(run_dir: Path, record: str, facts: dict) -> RAGRetriever:
    docs: list[RagDocument] = []
    # one precomputed run-statistics card (the standard RAG fix for counting questions)
    docs.append(RagDocument(
        doc_id="stats",
        text=(f"Focal agent run statistics: submitted {facts['n_bids']} bids; chose WATCH "
              f"{facts['n_watch']} times; {facts['n_filled']} bids filled or partially filled; "
              f"bid on side(s) {', '.join(facts['sides']) or 'none'}; maximum candidate quantity "
              f"{facts['max_qty']:.2f} MWh."),
        source=run_dir.name, kind="prior_run_lesson", market_as_of=None,
    ))
    # per-tick decision records (support causal questions)
    for line in record.splitlines():
        line = line.strip()
        if not line.lower().startswith("tick "):
            continue
        tick_id = line.split(":", 1)[0].replace("tick ", "tick-").strip()
        docs.append(RagDocument(doc_id=f"{tick_id}", text=line, source=run_dir.name,
                                kind="historical_stats", market_as_of=None))
    return RAGRetriever.build_from_documents(docs, backend="dense", device="cpu")


SYS = ("You are an audit assistant. Answer ONLY from the provided decision record and retrieved facts. "
       "If they do not contain the answer, reply exactly 'NOT IN RECORD'. Be concise.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--truth-dir", type=Path, required=True)
    ap.add_argument("--context-dir", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, default=Path("evaluations"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    ap.add_argument("--key", default="heimdall-local")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = [Path(p) for p in glob(args.glob) if (Path(p) / "traces.jsonl").exists()]
    cats = ["retrieval", "aggregate", "causal", "counterfact", "unanswerable"]
    agg = {c: {"llm_ok": 0, "rag_ok": 0, "det_ok": 0, "det_na": 0, "n": 0} for c in cats}
    for run in sorted(runs):
        ev = ensure_eval(run, args.truth_dir, args.eval_root, args.context_dir)
        record, qs = build(run, args.truth_dir, ev)
        retr = build_run_corpus(run, record, run_facts(run, ev))
        for q in qs:
            c = q["cat"]; agg[c]["n"] += 1
            # arm 1: LLM from record only (baseline)
            try:
                a_llm = ask(args.base_url, args.key, args.model, SYS, record + "\n\nQUESTION: " + q["q"])
            except Exception as e:  # noqa: BLE001
                a_llm = f"[err {e}]"
            agg[c]["llm_ok"] += int(bool(q["check"](a_llm)))
            # arm 2: LLM + RAG (retrieve-then-read)
            hits = retr.retrieve(q["q"], as_of=None, k=args.top_k)
            facts_block = "RETRIEVED FACTS:\n" + "\n".join(f"- {h['text']}" for h in hits)
            try:
                a_rag = ask(args.base_url, args.key, args.model, SYS,
                            facts_block + "\n\n" + record + "\n\nQUESTION: " + q["q"])
            except Exception as e:  # noqa: BLE001
                a_rag = f"[err {e}]"
            agg[c]["rag_ok"] += int(bool(q["check"](a_rag)))
            # arm 3: deterministic-query control
            if q["det"] == "NO_NL_CAPABILITY":
                agg[c]["det_na"] += 1
            else:
                agg[c]["det_ok"] += int(bool(q["check"](q["det"])))

    print(f"\nInterrogability + RAG — {len(runs)} runs, Qwen3-32B\n")
    print(f"{'category':<13}{'n':>4}{'LLM':>10}{'LLM+RAG':>12}{'det-query':>22}")
    print("-" * 61)
    out = {}
    for c in cats:
        a = agg[c]; n = a["n"]
        det = f"{a['det_ok']}/{n} exact" if a["det_na"] == 0 else f"NO NL CAP ({a['det_na']}/{n})"
        lc = f"{a['llm_ok']}/{n}" if n else "-"
        rc = f"{a['rag_ok']}/{n}" if n else "-"
        print(f"{c:<13}{n:>4}{lc:>10}{rc:>12}{det:>22}")
        out[c] = {**a,
                  "llm_pct": round(100 * a["llm_ok"] / n, 1) if n else None,
                  "rag_pct": round(100 * a["rag_ok"] / n, 1) if n else None}
    args.out.write_text(json.dumps(out, indent=2))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
