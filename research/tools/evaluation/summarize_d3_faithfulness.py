"""Summarize the D3 faithfulness matrix: grounded vs ungrounded, pooled + per frozen seed.

Headline metric: evidence-unfounded rate — the fraction of outage/grid claims the agent made with
NO supporting tool call (confabulated evidence). The value claim is that grounding the context
(seed_outage_context) drives this to ~0 while the ungrounded default confabulates, giving a faithful,
auditable rationale the deterministic core cannot produce. Reports robustness across the 5 frozen
seeds (a single-seed result is not publishable per feedback-value-metrics-need-clean-controls).

Usage:
    PYTHONPATH=. uv run python tools/evaluation/summarize_d3_faithfulness.py \
        --glob 'ai-society/runs/d3-faithfulness-20260524/d3-*-24-q32' \
        --json-out evaluations/d3_faithfulness_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path

from tools.evaluation.evaluate_rationale_faithfulness import ArmTally, score_run

FIELDS = ("n_records", "n_bids", "nums_total", "nums_grounded", "accept_claims", "accept_supported",
          "accept_contradicted", "regime_claims", "regime_supported", "regime_contradicted",
          "regime_unfounded", "evidence_claims", "evidence_unfounded", "autonomous_sim_records")


def _add(agg: ArmTally, t: ArmTally) -> None:
    for f in FIELDS:
        setattr(agg, f, getattr(agg, f) + getattr(t, f))


def _arm_seed(name: str) -> tuple[str, str]:
    arm = "grounded" if "-grounded-" in name else "ungrounded" if "-ungrounded-" in name else "other"
    m = re.search(r"seed(\d+)", name)
    return arm, (m.group(1) if m else "?")


def _rate(num: int, den: int) -> str:
    return f"{100*num/den:.0f}% ({num}/{den})" if den else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    dirs = []
    for g in args.glob:
        dirs += [Path(p) for p in glob(g)]
    dirs = sorted({d for d in dirs if (d / "traces.jsonl").exists()})
    if not dirs:
        raise SystemExit("no runs")

    pooled: dict[str, ArmTally] = {}
    per_seed: dict[tuple[str, str], ArmTally] = {}
    for d in dirs:
        arm, seed = _arm_seed(d.name)
        t = score_run(d)
        _add(pooled.setdefault(arm, ArmTally()), t)
        _add(per_seed.setdefault((arm, seed), ArmTally()), t)

    out = {"pooled": {}, "per_seed": {}}
    print(f"\nD3 faithfulness — {len(dirs)} runs\n")
    print("POOLED (across all seeds/windows):")
    print(f"  {'arm':<12}{'evid-unfounded':>18}{'regime-unfounded':>18}{'num-grounded':>16}{'autonomous':>12}")
    for arm in sorted(pooled):
        t = pooled[arm]
        print(f"  {arm:<12}{_rate(t.evidence_unfounded, t.evidence_claims):>18}"
              f"{_rate(t.regime_unfounded, t.regime_claims):>18}"
              f"{_rate(t.nums_grounded, t.nums_total):>16}"
              f"{_rate(t.autonomous_sim_records, t.n_records):>12}")
        out["pooled"][arm] = {"evidence_unfounded": t.evidence_unfounded, "evidence_claims": t.evidence_claims,
                              "evid_unfounded_rate": (t.evidence_unfounded / t.evidence_claims) if t.evidence_claims else None,
                              "regime_unfounded": t.regime_unfounded, "regime_claims": t.regime_claims,
                              "nums_grounded": t.nums_grounded, "nums_total": t.nums_total,
                              "n_records": t.n_records}
    print("\nPER SEED (evidence-unfounded rate):")
    seeds = sorted({s for _, s in per_seed})
    print(f"  {'arm':<12}" + "".join(f"{('seed'+s):>14}" for s in seeds))
    for arm in sorted({a for a, _ in per_seed}):
        cells = []
        for s in seeds:
            t = per_seed.get((arm, s))
            cells.append(_rate(t.evidence_unfounded, t.evidence_claims) if t else "—")
            if t:
                out["per_seed"].setdefault(arm, {})[s] = {
                    "evid_unfounded_rate": (t.evidence_unfounded / t.evidence_claims) if t.evidence_claims else None,
                    "evidence_claims": t.evidence_claims}
        print(f"  {arm:<12}" + "".join(f"{c:>14}" for c in cells))

    if "grounded" in pooled and "ungrounded" in pooled:
        g, u = pooled["grounded"], pooled["ungrounded"]
        gr = g.evidence_unfounded / g.evidence_claims if g.evidence_claims else float("nan")
        ur = u.evidence_unfounded / u.evidence_claims if u.evidence_claims else float("nan")
        print(f"\nHEADLINE: evidence-confabulation rate  ungrounded {ur:.0%}  ->  grounded {gr:.0%}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
