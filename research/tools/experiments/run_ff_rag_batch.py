"""Sequential launcher + scorer for the ff-rag-20260525 experiment.

Runs each config with `heimdall_ai_society run`, scores it with
`evaluate_society_run.py`, and writes incremental results to a PRIVATE summary
under the run dir (so it never clobbers the shared ablation-batch-summary.json
that the 3-GPU matrix writes). Pin GPU + endpoint by launching with:

  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    PYTHONPATH=.:ai-society/src uv run python tools/experiments/run_ff_rag_batch.py \
    --list ai-society/configs/ff-rag-20260525/full.txt

All configs already pin llm.base_urls to :8003; CUDA_VISIBLE_DEVICES=3 keeps the
in-process forecaster on GPU3. GPUs 0-2 are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

CONTEXT_ROOT = Path("data/cache/real_context")
TRUTH_ROOT = Path("data/cache/evaluation_truth")
RESULTS = Path("ai-society/runs/ff-rag-20260525/results.jsonl")
ARM_RE = re.compile(r"ff-rag-20260525-(det|cp12-norag|cp12-rag)-([a-z0-9-]+?)-seed(\d+)-(\d+)-q32")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    return env


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, env=_env(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def _run_json(cmd: list[str]) -> dict:
    out = _run(cmd)
    i = out.find("{")
    if i < 0:
        raise RuntimeError(f"no JSON from {' '.join(cmd)}:\n{out[-1500:]}")
    return json.loads(out[i:])


def run_one(config: Path) -> dict:
    cfg = _run_json(["uv", "run", "python", "-m", "heimdall_ai_society", "validate-config", str(config)])
    ctx_name = Path(cfg["context_dataset_dir"]).name
    context_dir = CONTEXT_ROOT / ctx_name
    truth_dir = TRUTH_ROOT / ctx_name
    out = _run(["uv", "run", "python", "-m", "heimdall_ai_society", "run", "--config", str(config)])
    run_dir = None
    for line in out.splitlines():
        if line.startswith("wrote society run:"):
            run_dir = Path(line.split(":", 1)[1].strip())
    if run_dir is None:
        raise RuntimeError(f"no run dir parsed for {config}\n{out[-1500:]}")
    eval_payload = _run_json([
        "uv", "run", "python", "tools/evaluation/evaluate_society_run.py",
        "--run-dir", str(run_dir), "--context-dir", str(context_dir),
        "--truth-dir", str(truth_dir), "--output-dir", str(Path("evaluations") / run_dir.name),
    ])
    rs = eval_payload["run_summary"]
    summary = json.loads((run_dir / "summary.json").read_text())
    rag = summary.get("rag") or {}
    m = ARM_RE.search(config.stem)
    return {
        "config": str(config),
        "run_id": run_dir.name,
        "arm": m.group(1) if m else "?",
        "window": m.group(2) if m else "?",
        "seed": int(m.group(3)) if m else None,
        "opportunity_capture": rs.get("opportunity_capture"),
        "cumulative_pnl_eur": rs.get("cumulative_pnl_eur"),
        "oracle_feasible_profit_eur": rs.get("oracle_feasible_profit_eur"),
        "fill_rate": rs.get("fill_rate"),
        "bid_count": rs.get("bid_count"),
        "filled_count": rs.get("filled_count"),
        "profit_per_mwh": rs.get("profit_per_mwh"),
        "accepted": summary.get("accepted"),
        "abstained": summary.get("abstained"),
        "watched": summary.get("watched"),
        "rag_enabled": summary.get("rag_enabled"),
        "rag_query_count": rag.get("query_count"),
    }


def aggregate(results: list[dict]) -> None:
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        if "error" in r:
            continue
        by_arm.setdefault(r["arm"], []).append(r)
    print("\n=== ff-rag-20260525 capture by arm (mean over windows/seeds) ===")
    print(f"{'arm':12s} {'n':>3s} {'capture':>9s} {'pnl_eur':>11s} {'fill':>6s} {'rag_q':>6s}")
    for arm in ["det", "cp12-norag", "cp12-rag"]:
        rs = by_arm.get(arm, [])
        caps = [x["opportunity_capture"] for x in rs if x["opportunity_capture"] is not None]
        pnls = [x["cumulative_pnl_eur"] for x in rs if x["cumulative_pnl_eur"] is not None]
        fills = [x["fill_rate"] for x in rs if x["fill_rate"] is not None]
        ragq = [x["rag_query_count"] for x in rs if x["rag_query_count"] is not None]
        mc = sum(caps) / len(caps) if caps else float("nan")
        mp = sum(pnls) / len(pnls) if pnls else float("nan")
        mf = sum(fills) / len(fills) if fills else float("nan")
        mq = sum(ragq) / len(ragq) if ragq else 0
        print(f"{arm:12s} {len(rs):>3d} {mc:>9.4f} {mp:>11.0f} {mf:>6.3f} {mq:>6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    ap.add_argument("--continue-on-failure", action="store_true", default=True)
    args = ap.parse_args()
    configs = [Path(x) for x in args.list.read_text().split() if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if args.append and args.out.exists():
        results = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
    with args.out.open("a" if args.append else "w") as fh:
        for i, config in enumerate(configs, 1):
            t0 = time.time()
            try:
                rec = run_one(config)
                rec["elapsed_seconds"] = round(time.time() - t0, 1)
            except Exception as exc:  # noqa: BLE001
                rec = {"config": str(config), "error": str(exc)[:500]}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            cap = rec.get("opportunity_capture")
            print(f"[{i}/{len(configs)}] {rec.get('arm','?'):11s} {rec.get('window','?')} "
                  f"seed{rec.get('seed','?')} capture={cap} rag_q={rec.get('rag_query_count')} "
                  f"{'ERR='+rec['error'] if 'error' in rec else ''}", flush=True)
    aggregate(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
