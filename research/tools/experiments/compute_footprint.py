"""Compute / energy footprint aggregator (proposal §10).

Scans every MLflow run under known tracking dirs for the
``heimdall.<name>.{wall_seconds,gpu_kwh,co2e_grams,gpu_avg_w}`` metric trio
that ``packages/ml/tracking.py`` logs via ``track_compute`` /
``track_experiment_compute``.

Aggregates per (run_name → model family), reports total wall-hours,
total kWh, total kg CO₂e.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/footprint"
SEARCH = [
    REPO / "mlruns",
    Path("/tmp"),
]
NAME_PAT = re.compile(r"^heimdall\.(?P<task>[^.]+)\.(?P<metric>wall_seconds|gpu_kwh|co2e_grams|gpu_avg_w)$")


def _scan_mlruns_dir(root: Path) -> list[dict]:
    """Yield dicts with {experiment_id, run_id, name, params, metrics}."""
    if not root.exists():
        return []
    rows: list[dict] = []
    # MLflow file backend: <root>/<exp_id>/<run_id>/{meta.yaml, metrics/*, params/*}
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in (".trash", "models"):
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta = run_dir / "meta.yaml"
            metrics_dir = run_dir / "metrics"
            if not meta.exists() or not metrics_dir.exists():
                continue
            name = None
            for line in meta.read_text().splitlines():
                if line.startswith("run_name"):
                    name = line.split(":", 1)[1].strip().strip("'\"")
                    break
            metrics = {}
            for mf in metrics_dir.iterdir():
                try:
                    val_line = mf.read_text().splitlines()[-1]
                    metrics[mf.name] = float(val_line.split()[1])
                except Exception:
                    continue
            rows.append({"experiment_id": exp_dir.name,
                         "run_id": run_dir.name, "name": name or run_dir.name,
                         "metrics": metrics})
    return rows


def main(argv: list[str] | None = None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict] = []
    for r in SEARCH:
        if r == Path("/tmp"):
            # Glob just the canonical MLflow dirs we know we created.
            for sub in r.glob("mlflow_*"):
                all_runs.extend(_scan_mlruns_dir(sub))
        else:
            all_runs.extend(_scan_mlruns_dir(r))

    rows = []
    for r in all_runs:
        wall = kwh = co2 = 0.0
        for k, v in r["metrics"].items():
            m = NAME_PAT.match(k)
            if not m:
                continue
            metric = m["metric"]
            if metric == "wall_seconds":
                wall = max(wall, v)
            elif metric == "gpu_kwh":
                kwh += v
            elif metric == "co2e_grams":
                co2 += v
        if wall or kwh:
            rows.append({"run_name": r["name"],
                         "experiment_id": r["experiment_id"],
                         "wall_seconds": wall, "gpu_kwh": kwh,
                         "co2e_grams": co2})

    # Aggregate per model family.
    fam_re = re.compile(r"^(?P<fam>canonical_[a-z0-9_]+_(?:price|activation)|"
                        r"[a-z0-9_]+?)(?:-?seed)?")
    by_fam: dict[str, dict] = {}
    for r in rows:
        m = fam_re.match(r["run_name"] or "")
        fam = (m["fam"] if m else r["run_name"]) or "unknown"
        d = by_fam.setdefault(fam, {"n_runs": 0, "wall_seconds": 0.0,
                                     "gpu_kwh": 0.0, "co2e_grams": 0.0})
        d["n_runs"] += 1
        d["wall_seconds"] += r["wall_seconds"]
        d["gpu_kwh"] += r["gpu_kwh"]
        d["co2e_grams"] += r["co2e_grams"]
    total = {"n_runs": sum(d["n_runs"] for d in by_fam.values()),
             "wall_hours": sum(d["wall_seconds"] for d in by_fam.values()) / 3600.0,
             "gpu_kwh": sum(d["gpu_kwh"] for d in by_fam.values()),
             "kg_co2e": sum(d["co2e_grams"] for d in by_fam.values()) / 1000.0}
    leaderboard = sorted([{"family": f, **d} for f, d in by_fam.items()],
                         key=lambda x: -x["gpu_kwh"])
    (OUT / "by_run.json").write_text(json.dumps(rows, indent=2))
    (OUT / "by_family.json").write_text(json.dumps(leaderboard, indent=2))
    (OUT / "totals.json").write_text(json.dumps(total, indent=2))
    print(f"scanned {len(rows)} runs across {len(by_fam)} families")
    print(f"total: {total['wall_hours']:.1f} wall-h | "
          f"{total['gpu_kwh']:.3f} kWh | {total['kg_co2e']:.4f} kg CO2e")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
