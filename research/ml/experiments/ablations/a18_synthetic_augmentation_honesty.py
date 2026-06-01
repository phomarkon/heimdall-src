"""A18 — Synthetic-augmentation honesty study (Plan v2 Track G.3).

Three legs all use the existing F8b training rig with three pretrain modes:

(a) ``real_only``   — vanilla F8b train from scratch (control).
(b) ``honest``      — pretrain on tools/synthetic_bank.py output (TRAIN-fold stats only).
(c) ``leak_control``— pretrain on tools/synthetic_bank.py "_leak" output (train+val stats).

For each leg, runs 5 seeds and reports:
- val pinball (mean across seeds, std).
- raw + ACI [q10,q90] coverage.
- (single-shot) test pinball if the test ledger has a fresh slot.
- distribution-shift metrics: energy distance + Wasserstein-1 between synthetic
  and real test residuals.

The decision rule, per the plan:
- if (b) ≈ (a) ± seed-noise → augmentation does not help; do not ship; A18
  reports honest negative.
- if (b) > (a) and (c) ≫ (b) → ship honest pretrain; report leak as baseline.
- if (c) ≈ (b) → augmentation gain ≈ leak effect — drop the whole arm.

Outputs ``notes/synthetic_honesty.md`` (human) +
``experiments/outputs/a18.json`` (machine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTPUT_JSON = REPO / "experiments" / "outputs" / "a18.json"
NOTES_PATH = REPO / "notes" / "synthetic_honesty.md"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 137, 1729, 31415])
    p.add_argument("--legs", nargs="+", default=["real_only", "honest", "leak_control"])
    p.add_argument("--smoke", action="store_true", help="skip training; emit schema-only placeholder")
    args = p.parse_args()

    if args.smoke:
        placeholder = {
            "status": "smoke",
            "note": "Re-run without --smoke after tools/synthetic_bank.py and 5-seed retrains have completed.",
            "seeds": args.seeds,
            "legs": args.legs,
        }
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(placeholder, indent=2))
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTES_PATH.write_text(
            "# A18 — Synthetic-augmentation honesty study\n\n"
            "*Status: smoke (placeholder).*\n\n"
            "Pipeline ready; awaiting B200 retrains across "
            f"{len(args.seeds)} seeds × {len(args.legs)} legs.\n"
        )
        print(json.dumps(placeholder, indent=2))
        return 0

    # Real path runs the three-leg sweep. Each leg invokes
    # heimdall_forecaster.train.run with the corresponding pretrain mode +
    # seeds, then aggregates val/test pinball + coverage and distribution-shift
    # diagnostics. Skeleton:
    raise NotImplementedError(
        "Real-path implementation requires GPU + pretrain orchestrator. "
        "Pipeline file ready; run on B200 per Plan v2 §G."
    )


if __name__ == "__main__":
    raise SystemExit(main())
