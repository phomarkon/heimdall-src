"""Attribute measured GPU energy + runtime to individual society runs.

Joins a gpu_telemetry.csv time series with each run's wall-clock window (from
summary.json: created_at_utc and runtime_seconds) and integrates total board power
over that window to get energy (kWh) per run, plus mean/peak power, mean util, peak
memory. Aggregates by arm / society and overall. This is the energy-footprint evidence
for the thesis + Applied Energy paper (real measured power, not estimated).

Energy is the trapezoidal integral of summed board power (all GPUs) over the run window.
Idle GPUs still draw a baseline (~200 W on B200), so total energy includes that; the
report also prints the idle baseline so the marginal compute energy can be derived.

Usage:
  uv run python tools/observability/resource_report.py \
    --telemetry ai-society/runs/<batch>/gpu_telemetry.csv \
    --runs-glob 'ai-society/runs/<batch>/*' \
    --out ai-society/runs/<batch>/resource_report.json [--co2-kg-per-kwh 0.122]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _load_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    for c in ["power_w", "util_gpu_pct", "mem_used_mib"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # total board power summed across GPUs at each timestamp
    total = df.groupby("ts_utc").agg(
        total_power_w=("power_w", "sum"),
        mean_util_pct=("util_gpu_pct", "mean"),
        max_mem_mib=("mem_used_mib", "max"),
        n_gpu=("gpu", "nunique"),
    ).reset_index().sort_values("ts_utc")
    return total


def _idle_baseline_w(total: pd.DataFrame) -> float:
    """Lowest-decile total power = system idle baseline."""
    return float(np.quantile(total["total_power_w"].to_numpy(), 0.10))


def _energy_kwh(total: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    win = total[(total["ts_utc"] >= start) & (total["ts_utc"] <= end)]
    if len(win) < 2:
        return {"energy_kwh": None, "mean_power_w": None, "peak_power_w": None,
                "mean_util_pct": None, "peak_mem_mib": None, "n_samples": int(len(win))}
    t = win["ts_utc"].astype("int64").to_numpy() / 1e9  # seconds
    p = win["total_power_w"].to_numpy()
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x renamed trapz
    energy_wh = float(_trap(p, t) / 3600.0)
    return {
        "energy_kwh": round(energy_wh / 1000.0, 5),
        "mean_power_w": round(float(p.mean()), 1),
        "peak_power_w": round(float(p.max()), 1),
        "mean_util_pct": round(float(win["mean_util_pct"].mean()), 1),
        "peak_mem_mib": int(win["max_mem_mib"].max()),
        "n_samples": int(len(win)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True, type=Path)
    ap.add_argument("--runs-glob", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--co2-kg-per-kwh", type=float, default=None,
                    help="optional grid carbon intensity to also report kgCO2e")
    args = ap.parse_args()

    total = _load_telemetry(args.telemetry)
    idle_w = _idle_baseline_w(total)

    runs = []
    for d in sorted(glob(args.runs_glob)):
        sp = Path(d) / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if "created_at_utc" not in s or "runtime_seconds" not in s:
            continue
        end = pd.Timestamp(s["created_at_utc"])
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        runtime = float(s["runtime_seconds"])
        start = end - pd.Timedelta(seconds=runtime)
        e = _energy_kwh(total, start, end)
        runs.append({
            "run_id": Path(d).name,
            "agent_count": s.get("agent_count"),
            "ticks": s.get("ticks"),
            "chooser_mode": s.get("chooser_mode"),
            "rag_enabled": s.get("rag_enabled"),
            "runtime_s": round(runtime, 1),
            "runtime_per_tick_s": round(runtime / max(int(s.get("ticks", 1)), 1), 2),
            **e,
        })

    scored = [r for r in runs if r["energy_kwh"] is not None]
    total_kwh = round(sum(r["energy_kwh"] for r in scored), 4)
    total_runtime_h = round(sum(r["runtime_s"] for r in runs) / 3600.0, 3)
    summary = {
        "n_runs": len(runs),
        "n_runs_with_energy": len(scored),
        "idle_baseline_total_w": round(idle_w, 1),
        "total_gpu_energy_kwh": total_kwh,
        "total_runtime_hours": total_runtime_h,
        "mean_kwh_per_run": round(total_kwh / len(scored), 5) if scored else None,
    }
    if args.co2_kg_per_kwh is not None:
        summary["co2_kg_per_kwh_assumed"] = args.co2_kg_per_kwh
        summary["total_co2e_kg"] = round(total_kwh * args.co2_kg_per_kwh, 4)

    # per (chooser, rag) aggregate
    agg = defaultdict(lambda: {"n": 0, "kwh": 0.0, "runtime_s": 0.0})
    for r in scored:
        key = f"{r['chooser_mode']}{'+rag' if r['rag_enabled'] else ''}"
        a = agg[key]
        a["n"] += 1
        a["kwh"] += r["energy_kwh"]
        a["runtime_s"] += r["runtime_s"]
    per_arm = {k: {"n": v["n"], "mean_kwh": round(v["kwh"] / v["n"], 5),
                   "mean_runtime_s": round(v["runtime_s"] / v["n"], 1)}
               for k, v in agg.items()}

    out = {"summary": summary, "per_arm": per_arm, "runs": runs}
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nGPU resource report ({args.telemetry.name})")
    print(f"  runs scored: {len(scored)}/{len(runs)}   idle baseline: {idle_w:.0f} W (all GPUs)")
    print(f"  total GPU energy: {total_kwh:.3f} kWh   total runtime: {total_runtime_h:.2f} h")
    if args.co2_kg_per_kwh is not None:
        print(f"  est. CO2e: {summary['total_co2e_kg']:.3f} kg  (@ {args.co2_kg_per_kwh} kg/kWh)")
    print(f"\n  {'arm':16s}{'n':>3s}{'mean kWh':>10s}{'mean runtime':>14s}")
    for k, v in sorted(per_arm.items()):
        print(f"  {k:16s}{v['n']:>3d}{v['mean_kwh']:>10.4f}{v['mean_runtime_s']:>12.0f} s")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
