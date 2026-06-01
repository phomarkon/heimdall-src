"""Attribute LLM tokens / latency / cost to individual society runs.

Differences the vLLM cumulative counters (from vllm_metrics.py) over each run's wall-clock
window (summary.json created_at_utc + runtime_seconds), summed across all endpoints, to get
per-run: prompt tokens, generation tokens, requests, mean end-to-end latency, generation
throughput, and (optional, with explicit rates) token-equivalent API cost and GPU-time cost.
Aggregated by arm / society + overall. The LLM-side footprint evidence for the thesis.

Cost is OFF unless you pass rates (no hidden assumptions):
  --in-usd-per-mtok / --out-usd-per-mtok  -> token-equivalent cost as if billed by a hosted API
  --gpu-hr-usd                            -> compute cost = runtime_h * n_gpu * rate (self-hosted reality)

Usage:
  uv run python tools/observability/llm_cost_report.py \
    --metrics ai-society/runs/<batch>/vllm_metrics.csv \
    --runs-glob 'ai-society/runs/<batch>/*' --out ai-society/runs/<batch>/llm_cost_report.json \
    [--in-usd-per-mtok 0.15 --out-usd-per-mtok 0.60 --gpu-hr-usd 3.0 --n-gpu 4]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

COUNTERS = ["prompt_tokens_total", "generation_tokens_total", "requests_total",
            "latency_sum_s", "latency_count"]


def _series(metrics: Path) -> dict[str, dict[str, np.ndarray]]:
    df = pd.read_csv(metrics)
    df["t"] = pd.to_datetime(df["ts_utc"], utc=True).astype("int64") / 1e9
    out: dict[str, dict[str, np.ndarray]] = {}
    for ep, g in df.groupby("endpoint"):
        g = g.sort_values("t")
        out[str(ep)] = {"t": g["t"].to_numpy(),
                        **{c: pd.to_numeric(g[c], errors="coerce").to_numpy() for c in COUNTERS}}
    return out


def _delta(series: dict, start: float, end: float) -> dict[str, float]:
    """Sum, across endpoints, the counter increase over [start, end] (interp on monotonic counters)."""
    tot = dict.fromkeys(COUNTERS, 0.0)
    covered = False
    for ep, s in series.items():
        if s["t"].min() > start or s["t"].max() < end:
            continue  # window not fully covered by this endpoint's samples
        covered = True
        for c in COUNTERS:
            tot[c] += float(np.interp(end, s["t"], s[c]) - np.interp(start, s["t"], s[c]))
    tot["_covered"] = covered
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--runs-glob", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--in-usd-per-mtok", type=float, default=None)
    ap.add_argument("--out-usd-per-mtok", type=float, default=None)
    ap.add_argument("--gpu-hr-usd", type=float, default=None)
    ap.add_argument("--n-gpu", type=int, default=4)
    args = ap.parse_args()

    series = _series(args.metrics)
    runs = []
    for d in sorted(glob(args.runs_glob)):
        sp = Path(d) / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if not s.get("llm_enabled") or "created_at_utc" not in s:
            continue  # deterministic runs make no LLM calls
        end = pd.Timestamp(s["created_at_utc"])
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        runtime = float(s["runtime_seconds"])
        start_ts, end_ts = (end.value / 1e9) - runtime, end.value / 1e9
        dt = _delta(series, start_ts, end_ts)
        if not dt["_covered"]:
            runs.append({"run_id": Path(d).name, "covered": False, "runtime_s": round(runtime, 1)})
            continue
        prompt, gen = dt["prompt_tokens_total"], dt["generation_tokens_total"]
        reqs, lsum, lcnt = dt["requests_total"], dt["latency_sum_s"], dt["latency_count"]
        rec = {
            "run_id": Path(d).name, "covered": True,
            "agent_count": s.get("agent_count"), "ticks": s.get("ticks"),
            "rag_enabled": s.get("rag_enabled"), "runtime_s": round(runtime, 1),
            "prompt_tokens": int(prompt), "generation_tokens": int(gen),
            "total_tokens": int(prompt + gen), "requests": int(round(reqs)),
            "mean_latency_s": round(lsum / lcnt, 2) if lcnt > 0 else None,
            "gen_throughput_tok_s": round(gen / runtime, 1) if runtime > 0 else None,
            "tokens_per_request": int((prompt + gen) / reqs) if reqs > 0 else None,
        }
        if args.in_usd_per_mtok is not None and args.out_usd_per_mtok is not None:
            rec["token_equiv_usd"] = round(prompt / 1e6 * args.in_usd_per_mtok
                                           + gen / 1e6 * args.out_usd_per_mtok, 4)
        if args.gpu_hr_usd is not None:
            rec["gpu_time_usd"] = round(runtime / 3600 * args.n_gpu * args.gpu_hr_usd, 4)
        runs.append(rec)

    scored = [r for r in runs if r.get("covered")]
    tot_prompt = sum(r["prompt_tokens"] for r in scored)
    tot_gen = sum(r["generation_tokens"] for r in scored)
    summary = {
        "n_runs": len(runs), "n_llm_runs_covered": len(scored),
        "total_prompt_tokens": tot_prompt, "total_generation_tokens": tot_gen,
        "total_tokens": tot_prompt + tot_gen,
        "total_requests": int(sum(r["requests"] for r in scored)),
    }
    if scored and "token_equiv_usd" in scored[0]:
        summary["total_token_equiv_usd"] = round(sum(r["token_equiv_usd"] for r in scored), 2)
    if scored and "gpu_time_usd" in scored[0]:
        summary["total_gpu_time_usd"] = round(sum(r["gpu_time_usd"] for r in scored), 2)

    agg = defaultdict(lambda: {"n": 0, "gen": 0, "prompt": 0, "lat": [], "thru": []})
    for r in scored:
        k = f"{'rag' if r['rag_enabled'] else 'llm'}-{r['agent_count']}ag"
        a = agg[k]
        a["n"] += 1; a["gen"] += r["generation_tokens"]; a["prompt"] += r["prompt_tokens"]
        if r["mean_latency_s"]: a["lat"].append(r["mean_latency_s"])
        if r["gen_throughput_tok_s"]: a["thru"].append(r["gen_throughput_tok_s"])
    per_arm = {k: {"n": v["n"], "mean_gen_tokens": int(v["gen"] / v["n"]),
                   "mean_prompt_tokens": int(v["prompt"] / v["n"]),
                   "mean_latency_s": round(float(np.mean(v["lat"])), 2) if v["lat"] else None,
                   "mean_throughput_tok_s": round(float(np.mean(v["thru"])), 1) if v["thru"] else None}
               for k, v in agg.items()}

    out = {"summary": summary, "per_arm": per_arm, "runs": runs}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nLLM cost/token report ({args.metrics.name})")
    print(f"  LLM runs covered: {len(scored)}   total tokens: {summary['total_tokens']:,} "
          f"(gen {tot_gen:,} / prompt {tot_prompt:,})   requests: {summary['total_requests']:,}")
    if "total_token_equiv_usd" in summary:
        print(f"  token-equivalent API cost: ${summary['total_token_equiv_usd']:,.2f}")
    if "total_gpu_time_usd" in summary:
        print(f"  GPU-time cost: ${summary['total_gpu_time_usd']:,.2f}")
    print(f"\n  {'arm':14s}{'n':>3s}{'gen tok':>10s}{'prompt tok':>12s}{'lat s':>8s}{'tok/s':>8s}")
    for k, v in sorted(per_arm.items()):
        print(f"  {k:14s}{v['n']:>3d}{v['mean_gen_tokens']:>10,}{v['mean_prompt_tokens']:>12,}"
              f"{(v['mean_latency_s'] or 0):>8.1f}{(v['mean_throughput_tok_s'] or 0):>8.0f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
