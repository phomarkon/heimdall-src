"""Blinded explanation-preference study with an independent (non-Qwen) judge via OpenRouter.

Validates "richer = better explanation" beyond the syntactic rubric: an independent model judges,
for the SAME bid decision, which of two rationales better lets an operator audit the trade. Two
comparisons:
  vs_det:      LLM rationale  vs  the deployed deterministic rationale (constant procedural string)
  vs_template: LLM rationale  vs  a rich hand-written deterministic template (the ceiling)

Each pair is judged in BOTH orders (LLM-as-A and LLM-as-B). A preference counts only when the judge
is order-consistent (picks the same content both ways); order-flips are reported as position bias and
not counted as a preference. This is the rigorous, still-cheap design. Independent judge avoids the
self-preference bias of a Qwen-judges-Qwen setup.

Usage:
  python tools/evaluation/run_preference_judge.py --n 24 --model google/gemini-2.0-flash-lite-001 \
      --out evaluations/preference_judge.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from glob import glob
from pathlib import Path

import requests

GROUNDED = "ai-society/runs/d1-infoedge-20260524/d1-grounded-jan08-seed42-24-q32/traces.jsonl"
TEMPLATE = "/tmp/d1-template-arm/traces.jsonl"
DET_STRING = "deterministic_best_accepted selected the highest-ranked exact simulator-accepted candidate"

SYS = ("You audit a Nordic mFRR balancing-market trading agent. You are shown TWO explanations, A and B, "
       "of the SAME bid decision. Pick the one that better lets a market operator understand and audit "
       "WHY the bid was made (specific evidence, the drivers of the decision, alternatives considered). "
       "Answer with exactly one character: A or B. No other text.")


def _key() -> str:
    for l in Path(".env").read_text().splitlines():
        if l.startswith("OPENROUTER_API_KEY"):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY in .env")


def judge(key: str, model: str, a: str, b: str) -> str | None:
    body = {"model": model, "max_tokens": 3, "temperature": 0,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": f"Explanation A:\n{a}\n\nExplanation B:\n{b}\n\nWhich is better, A or B?"}]}
    for _ in range(3):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 200:
            t = r.json()["choices"][0]["message"]["content"].strip().upper()
            m = re.search(r"[AB]", t)
            return m.group(0) if m else None
        time.sleep(2)
    return None


def pref_pair(key, model, llm_text, other_text):
    """Judge both orders; return 'llm' / 'other' if consistent, else 'bias'."""
    o1 = judge(key, model, llm_text, other_text)   # LLM is A
    o2 = judge(key, model, other_text, llm_text)   # LLM is B
    if o1 == "A" and o2 == "B":
        return "llm"
    if o1 == "B" and o2 == "A":
        return "other"
    return "bias"


def _collect_pairs(glob_pat: str, per_run: int) -> list[tuple[str, str, str]]:
    """Across runs in glob: (llm_rationale, template_rationale, det_string) for sampled bids."""
    import subprocess
    triples = []
    for run in sorted(glob(glob_pat)):
        run = Path(run)
        if not (run / "traces.jsonl").exists():
            continue
        tmpl_dir = Path("/tmp/pref-tmpl") / run.name
        if not (tmpl_dir / "traces.jsonl").exists():
            subprocess.run(["python3", "tools/evaluation/_template_rationale_control.py", str(run), str(tmpl_dir)],
                           capture_output=True)
        g = [json.loads(l) for l in (run / "traces.jsonl").read_text().splitlines() if l.strip()]
        tm = {(r["agent_id"], r["step"]): (r.get("decision") or {}).get("rationale")
              for r in (json.loads(l) for l in (tmpl_dir / "traces.jsonl").read_text().splitlines() if l.strip())}
        bids = [r for r in g if (r.get("decision") or {}).get("action") == "bid"]
        random.shuffle(bids)
        for r in bids[:per_run]:
            llm = (r.get("decision") or {}).get("rationale")
            tpl = tm.get((r["agent_id"], r["step"]))
            if llm and tpl:
                triples.append((llm, tpl, DET_STRING))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", help="LLM run dirs to draw pairs from (multi-seed)")
    ap.add_argument("--per-run", type=int, default=8)
    ap.add_argument("--n", type=int, default=24, help="single-run fallback count")
    ap.add_argument("--models", default="google/gemini-2.0-flash-lite-001",
                    help="comma-separated OpenRouter judges (independent of Qwen)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = _key()
    random.seed(42)

    if args.glob:
        triples = _collect_pairs(args.glob, args.per_run)
    else:
        g = [json.loads(l) for l in Path(GROUNDED).read_text().splitlines() if l.strip()]
        t = {(r["agent_id"], r["step"]): (r.get("decision") or {}).get("rationale")
             for r in (json.loads(l) for l in Path(TEMPLATE).read_text().splitlines() if l.strip())}
        bids = [r for r in g if (r.get("decision") or {}).get("action") == "bid"]
        random.shuffle(bids)
        triples = [((r.get("decision") or {}).get("rationale"), t.get((r["agent_id"], r["step"])), DET_STRING)
                   for r in bids[:args.n]]
        triples = [x for x in triples if x[0] and x[1]]

    out = {"n_pairs": len(triples), "judges": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        res = {"vs_det": {"llm": 0, "other": 0, "bias": 0}, "vs_template": {"llm": 0, "other": 0, "bias": 0}}
        for llm_text, tpl_text, det in triples:
            res["vs_det"][pref_pair(key, model, llm_text, det)] += 1
            res["vs_template"][pref_pair(key, model, llm_text, tpl_text)] += 1
        for comp in ("vs_det", "vs_template"):
            n = sum(res[comp].values())
            res[comp]["llm_pct"] = round(100 * res[comp]["llm"] / n, 1) if n else None
            print(f"[{model}] {comp}: LLM {res[comp]['llm']} / other {res[comp]['other']} / "
                  f"bias {res[comp]['bias']}  -> LLM-preferred {res[comp]['llm_pct']}%")
        out["judges"][model] = res
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  (n_pairs={len(triples)})")


if __name__ == "__main__":
    main()
