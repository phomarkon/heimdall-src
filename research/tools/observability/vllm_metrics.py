"""Scrape vLLM Prometheus /metrics (token + latency counters) at a fixed interval.

Records the cumulative counters each vLLM endpoint exposes so a reporter can difference
them over a run's time window and attribute tokens / requests / latency per run (the LLM
analogue of gpu_telemetry.py). Read-only HTTP GET; negligible overhead.

Captured per endpoint per sample (cumulative since server start):
  prompt_tokens_total, generation_tokens_total, request_success_total (all reasons),
  e2e_request_latency_seconds_sum / _count.

Usage:
  uv run python tools/observability/vllm_metrics.py --out <csv> \
    [--endpoints http://127.0.0.1:8000 ...] [--interval 5] [--duration-s N]
"""
from __future__ import annotations

import argparse
import csv
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

WANTED = {
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success_total": "requests_total",
    "vllm:e2e_request_latency_seconds_sum": "latency_sum_s",
    "vllm:e2e_request_latency_seconds_count": "latency_count",
}
HEADER = ["ts_utc", "endpoint", "prompt_tokens_total", "generation_tokens_total",
          "requests_total", "latency_sum_s", "latency_count"]


def _scrape(endpoint: str) -> dict[str, float]:
    url = endpoint.rstrip("/") + "/metrics"
    acc = dict.fromkeys(WANTED.values(), 0.0)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"[vllm-metrics] scrape error {endpoint}: {exc}")
        return acc
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in WANTED:
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        acc[WANTED[name]] += val  # sum across label sets (engines / finished_reasons)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--endpoints", nargs="+",
                    default=[f"http://127.0.0.1:{p}" for p in (8000, 8001, 8002, 8003)])
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    started = time.time()
    n = 0
    with args.out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HEADER)
            fh.flush()
        try:
            while True:
                ts = datetime.now(tz=UTC).isoformat()
                for ep in args.endpoints:
                    m = _scrape(ep)
                    w.writerow([ts, ep, m["prompt_tokens_total"], m["generation_tokens_total"],
                                m["requests_total"], m["latency_sum_s"], m["latency_count"]])
                fh.flush()
                n += 1
                if args.duration_s and (time.time() - started) >= args.duration_s:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[vllm-metrics] wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()
