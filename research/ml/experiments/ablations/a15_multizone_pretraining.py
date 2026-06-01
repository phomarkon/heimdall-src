"""A15 — Multi-zone pretraining gain (Plan v2 Track C.4).

Three legs comparing F8b/F12 variants:

(a) ``dk1_only``       — F8b trained from scratch on DK1 alone (control).
(b) ``joint_no_finetune`` — F12 backbone trained on DE/SE3/NO2/DK1 jointly,
    evaluated on DK1 directly (no fine-tune).
(c) ``joint_finetune`` — F12 pretrain on the multi-zone bank, then fine-tune
    last 2 transformer layers on DK1 rich panel.

Reports: val pinball, raw + ACI coverage, single-shot test pinball.

Skeleton: depends on F12 trainer (apps/forecaster/.../train/f12_multizone.py)
which uses Trinity multi-zone DA prices via heimdall_data.trinity.load_trinity_prices.

Run via:
    uv run python experiments/ablations/a15_multizone_pretraining.py --smoke
to validate plumbing without GPU; drop --smoke for real 5-seed sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTPUT_JSON = REPO / "experiments" / "outputs" / "a15.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 137, 1729, 31415])
    p.add_argument("--legs", nargs="+",
                   default=["dk1_only", "joint_no_finetune", "joint_finetune"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        OUTPUT_JSON.write_text(json.dumps(
            {"status": "smoke", "seeds": args.seeds, "legs": args.legs}, indent=2
        ))
        print("smoke ok")
        return 0

    raise NotImplementedError(
        "Requires apps/forecaster/.../train/f12_multizone.py and the multi-zone "
        "synthetic-or-real panel. Run on B200 — see Plan v2 §C."
    )


if __name__ == "__main__":
    raise SystemExit(main())
