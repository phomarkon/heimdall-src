"""A17 — Counterfactual feasibility distribution (Plan v2 Track E.3).

For every verifier-rejected val bid, compute the minimum-σ perturbation that
flips the verdict using ``counterfactual_bid``. Reports the distribution
(median, p25, p75, fraction infeasible within an L2 σ-ball of 6) plus
per-rejection rationale strings.

This is the auditability headline: "the median rejected bid would have been
accepted if forecast wind had been +X MW higher" — a sentence Danfoss can
ship to operators.

This script is data-driven: caller supplies a JSONL of rejected bids
(structured as in ``apps/verifier`` outputs) plus the trained forecaster
to invoke per perturbation.

Skeleton-only here — the heavy bid-trace generation depends on Tim's
agent-runner outputs. For now we emit a placeholder JSON so the proposal
table renders; replace with real bids when Tim's track lands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUTPUT_JSON = REPO / "experiments" / "outputs" / "a17.json"


def _summary(deltas: list[float]) -> dict:
    arr = np.array([d for d in deltas if np.isfinite(d)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "infeasible_pct": 100.0}
    feas = (arr <= 6.0).mean() * 100
    return {
        "n": int(arr.size),
        "feasible_within_6sigma_pct": float(feas),
        "delta_sigma_median": float(np.median(arr)),
        "delta_sigma_p25": float(np.percentile(arr, 25)),
        "delta_sigma_p75": float(np.percentile(arr, 75)),
        "delta_sigma_max": float(np.max(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rejected-bids", type=Path, default=None,
                   help="JSONL of rejected verifier verdicts. If absent, emits placeholder.")
    p.add_argument("--out", type=Path, default=OUTPUT_JSON)
    args = p.parse_args()

    if args.rejected_bids is None or not args.rejected_bids.exists():
        record = {
            "status": "placeholder",
            "note": (
                "Tim-blocked: requires verifier trace JSONL from apps/verifier outputs. "
                "Re-run with --rejected-bids <path> once Tim's society-run produces "
                "bid traces."
            ),
            "summary": _summary([]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        return 0

    # Real path: iterate rejected bids, invoke counterfactual_bid, persist deltas.
    # (Implementation deferred until trace schema is finalised with Tim.)
    print("real-path implementation pending Tim's trace schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
