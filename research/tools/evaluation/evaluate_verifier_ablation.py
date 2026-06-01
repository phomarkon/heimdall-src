"""Score + aggregate the verifier-ablation matrix into a robust verifier-vs-verifierless result.

Pipeline per run: ensure `evaluations/<run_id>/run_summary.json` exists (score via
evaluate_society_run if missing), then aggregate by (variant, window) across the 5 frozen seeds.

Metrics:
  - realized_profit_eur            (mean +/- std over seeds)        -> from run_summary
  - verifier_subfloor_rate         (shadow: share of submitted bids with worst-case < 0)
  - would_have_been_blocked_count  (shadow: count the verifier would reject)
  - bid / watch / abstain counts   -> from the society summary.json
The clean verifier contribution is guarded vs shadow-toolvisible (tools held constant, gate toggled);
shadow-contextonly shows the value of the gate when grounding degrades; deterministic is the
always-safe reference.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_verifier_ablation.py \
        --json-out evaluations/verifier_ablation_20260524.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

RUN_ROOT = Path("ai-society/runs/verifier-ablation-20260524")
CONTEXT_DIR = Path("data/cache/real_context/april_2026")
TRUTH_DIR = Path("data/cache/evaluation_truth/april_2026")
VARIANTS = ["deterministic", "guarded", "shadow-toolvisible", "shadow-contextonly"]


def _parse_run_id(run_id: str) -> tuple[str, str, str] | None:
    # vab-s06-actioncore-<variant>-<window>-seed<seed>-24-q32
    for variant in VARIANTS:
        tag = f"-{variant}-"
        if tag in run_id:
            rest = run_id.split(tag, 1)[1]  # <window>-seed<seed>-24-q32
            window = rest.split("-seed", 1)[0]
            seed = rest.split("-seed", 1)[1].split("-", 1)[0]
            return variant, window, seed
    return None


def _ensure_scored(run_dir: Path) -> Path | None:
    out = Path("evaluations") / run_dir.name / "run_summary.json"
    if out.exists():
        return out
    cmd = [
        sys.executable, "tools/evaluation/evaluate_society_run.py",
        "--run-dir", str(run_dir),
        "--context-dir", str(CONTEXT_DIR),
        "--truth-dir", str(TRUTH_DIR),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"  score FAIL {run_dir.name}: {r.stderr.strip().splitlines()[-1:] }")
        return None
    return out if out.exists() else None


def _shadow_metrics(traces_path: Path) -> dict | None:
    shadows = []
    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for call in rec.get("tool_calls") or []:
            if call.get("name") == "shadow_required_simulation":
                shadows.append(call.get("result") or {})
    if not shadows:
        return None

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    rejected = [s for s in shadows if s.get("shadow_accepted") is not True]
    neg = [s for s in shadows if (_f(s.get("shadow_worst_case_profit_eur")) or 0) < 0]
    return {
        "shadow_bid_count": len(shadows),
        "would_have_been_blocked_count": len(rejected),
        "subfloor_rate": round(len(neg) / len(shadows), 6),
        "reject_rate": round(len(rejected) / len(shadows), 6),
        "min_worst_case_eur": min((_f(s.get("shadow_worst_case_profit_eur")) for s in shadows
                                   if _f(s.get("shadow_worst_case_profit_eur")) is not None), default=None),
    }


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(mean(xs), 2), "std": round(pstdev(xs), 2) if len(xs) > 1 else 0.0, "n": len(xs)}


def _aggregate(cells: dict[tuple[str, str], dict[str, list]]) -> tuple[dict, dict]:
    by_variant: dict[str, dict] = {}
    by_cell: dict[str, dict] = {}
    for variant in VARIANTS:
        prof, sub, wb, bids, wat, ab = [], [], [], [], [], []
        for window in sorted({w for (v, w) in cells if v == variant}):
            c = cells[(variant, window)]
            prof += c["profit"]
            sub += c["subfloor"]
            wb += c["would_block"]
            bids += c["bids"]
            wat += c["watch"]
            ab += c["abstain"]
            by_cell[f"{variant}|{window}"] = {
                "n_seeds": len(c["profit"]),
                "realized_profit_eur": _stats(c["profit"]),
                "verifier_subfloor_rate": _stats(c["subfloor"]),
                "would_block": _stats([float(x) for x in c["would_block"]]),
            }
        by_variant[variant] = {
            "realized_profit_eur": _stats(prof),
            "verifier_subfloor_rate": _stats(sub),
            "would_block_count": _stats([float(x) for x in wb]),
            "bids_made": _stats([float(x) for x in bids]),
            "watch": _stats([float(x) for x in wat]),
            "abstain": _stats([float(x) for x in ab]),
        }
    return by_variant, by_cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--f8-metrics", type=Path, default=Path("models/forecaster/f8/metrics.json"))
    args = ap.parse_args()

    # per (variant, window) -> lists across seeds
    cells: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"profit": [], "subfloor": [], "would_block": [], "bids": [], "watch": [], "abstain": []}
    )
    run_dirs = sorted(
        d for d in RUN_ROOT.glob("vab-*")
        if (d / "summary.json").exists() and (d / "traces.jsonl").exists()
    )
    print(f"scoring {len(run_dirs)} runs...")
    for d in run_dirs:
        parsed = _parse_run_id(d.name)
        if not parsed:
            continue
        variant, window, _seed = parsed
        summ = json.loads((d / "summary.json").read_text())
        scored = _ensure_scored(d)
        profit = None
        if scored:
            rs = json.loads(scored.read_text())
            profit = rs.get("realized_profit_eur")
        c = cells[(variant, window)]
        c["profit"].append(profit)
        c["watch"].append(summ.get("watched", 0))
        c["abstain"].append(summ.get("abstained", 0))
        if variant.startswith("shadow"):
            sm = _shadow_metrics(d / "traces.jsonl")
            if sm:
                c["subfloor"].append(sm["subfloor_rate"])
                c["would_block"].append(sm["would_have_been_blocked_count"])
                c["bids"].append(sm["shadow_bid_count"])  # shadow bids aren't labelled "accepted"
        else:
            c["bids"].append(summ.get("accepted", 0))  # verifier-enforced -> accepted == bids made

    by_variant, by_cell = _aggregate(cells)

    # coverage premise (Thm 1b) from F8
    coverage = None
    if args.f8_metrics.exists():
        d = json.loads(args.f8_metrics.read_text())
        raw = [s.get("val_q10_q90_coverage") for s in d.get("per_seed", [])]
        aci = [s.get("aci_empirical_coverage") for s in d.get("per_seed", [])]
        coverage = {"raw_mean": round(mean(raw), 4), "aci_mean": round(mean(aci), 4), "target": 0.90, "n_seeds": len(aci)}

    out = {
        "matrix": "verifier-ablation-20260524",
        "coverage_premise_thm1b": coverage,
        "by_variant": by_variant,
        "by_variant_window": by_cell,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
    _print_report(out)
    if args.json_out:
        print(f"\nwrote {args.json_out}")


def _print_report(out: dict) -> None:
    coverage = out["coverage_premise_thm1b"]
    by_variant = out["by_variant"]
    if coverage:
        print(f"\nCoverage premise (Thm 1b): raw {coverage['raw_mean']:.3f} -> ACI {coverage['aci_mean']:.3f} (target 0.90)")
    print(f"\n{'variant':<20}{'profit mean±std':>22}{'subfloor%':>14}{'would_block':>13}{'bids':>8}{'watch':>8}")
    for v in VARIANTS:
        s = by_variant[v]
        p, sf, wb = s["realized_profit_eur"], s["verifier_subfloor_rate"], s["would_block_count"]
        pstr = f"{p['mean']}±{p['std']}" if p["mean"] is not None else "n/a"
        sfstr = f"{100*sf['mean']:.1f}±{100*sf['std']:.1f}" if sf["mean"] is not None else "n/a"
        wbstr = f"{wb['mean']:.1f}" if wb["mean"] is not None else "n/a"
        print(f"{v:<20}{pstr:>22}{sfstr:>14}{wbstr:>13}{s['bids_made']['mean']:>8}{s['watch']['mean']:>8}")


if __name__ == "__main__":
    main()
