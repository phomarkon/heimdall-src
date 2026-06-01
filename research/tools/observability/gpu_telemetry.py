"""Lightweight GPU resource sampler (power / util / memory / temp / clocks).

Polls nvidia-smi at a fixed interval and appends one CSV row per GPU per sample,
each stamped with a real UTC timestamp. Read-only (query only); negligible overhead
(one nvidia-smi call per interval). Pair with resource_report.py to attribute measured
energy to individual runs and produce the energy-footprint evidence for the thesis /
Applied Energy paper (reproducibility contract: log compute/energy from day one).

Usage:
  uv run python tools/observability/gpu_telemetry.py --out <csv> [--interval 5] [--duration-s N]
Stop with Ctrl-C / SIGTERM (it flushes on every sample, so partial data is always valid).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

FIELDS = [
    "index", "power.draw", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu",
    "clocks.sm", "clocks.mem",
]
HEADER = ["ts_utc", "gpu", "power_w", "util_gpu_pct", "util_mem_pct",
          "mem_used_mib", "mem_total_mib", "temp_c", "clock_sm_mhz", "clock_mem_mhz"]


def _sample() -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=20,
    ).stdout.strip()
    ts = datetime.now(tz=UTC).isoformat()
    rows = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        rows.append([ts, *vals])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = run until killed")
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
                try:
                    for row in _sample():
                        w.writerow(row)
                    fh.flush()
                    n += 1
                except Exception as exc:  # noqa: BLE001 - a transient nvidia-smi hiccup must not kill sampling
                    print(f"[telemetry] sample error: {exc}")
                if args.duration_s and (time.time() - started) >= args.duration_s:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[telemetry] wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()
