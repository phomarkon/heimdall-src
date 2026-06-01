"""Post-hoc labeller for freeform-matrix traces.

Walks traces.jsonl in one or more run directories and emits a JSONL report:
for every `propose_action` decision, classify whether the (side, qty, price)
tuple matches an accepted simulate_* call earlier in the same agent-tick or
falls off-grid, and whether the shadow_required_simulation (post-hoc) deemed
it feasible.

This is the analysis side of the selector-trap → proposal-feasibility study:
- selector cell (preprobe=full, exact-match guard): expected near-100% seeded_match.
- hybrid cell (preprobe=full, shadow guard): mixture; the off_grid share is the
  "did the LLM venture off the seeded set?" signal.
- freeform cell (preprobe=none, shadow guard): all proposals come from the
  LLM's own simulate_* probes (or from no probe at all = pure dead-reckoning).

Usage:
    python tools/analysis/freeform_proposal_source.py \
        --matrix-dir ai-society/runs/freeform-matrix-20260526 \
        --out tools/analysis/freeform_proposal_source.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRICE_TOL = 0.5      # EUR/MWh
QTY_TOL = 0.05       # MWh — wider than min-size 0.25 / 5 so rounding doesn't false-negative
SIM_TOOLS = {
    "simulate_bid", "simulate_ev_bid", "simulate_wind_bid",
    "simulate_generator_bid", "simulate_retailer_bid", "simulate_renewables_bid",
}


def _match(a: dict, qty: float, price: float, side: str) -> bool:
    args = a.get("arguments") or {}
    if str(args.get("side", "")).lower() != side:
        return False
    try:
        aq = float(args.get("quantity_mwh"))
        ap = float(args.get("limit_price_eur_mwh"))
    except (TypeError, ValueError):
        return False
    return abs(aq - qty) <= QTY_TOL and abs(ap - price) <= PRICE_TOL


def _is_accepted(tc: dict) -> bool:
    result = tc.get("result") or {}
    if not isinstance(result, dict):
        return False
    if result.get("accepted") is True:
        return True
    verdict = result.get("verdict")
    if isinstance(verdict, dict) and verdict.get("accepted") is True:
        return True
    return False


def classify_row(row: dict) -> dict | None:
    decision = row.get("decision") or {}
    if (decision.get("action") or "").lower() != "bid":
        return None
    try:
        qty = float(decision.get("quantity_mwh"))
        price = float(decision.get("limit_price_eur_mwh"))
    except (TypeError, ValueError):
        return None
    side = str(decision.get("side") or "").lower()

    sims_total = 0
    sims_accepted = 0
    seeded_match = False
    seeded_match_accepted = False
    llm_probe_match = False
    llm_probe_match_accepted = False
    shadow_accepted: bool | None = None
    shadow_reasons: list[str] = []

    for tc in (row.get("tool_calls") or []):
        name = tc.get("name", "")
        if name in SIM_TOOLS:
            sims_total += 1
            acc = _is_accepted(tc)
            if acc:
                sims_accepted += 1
            if _match(tc, qty, price, side):
                provenance = (tc.get("provenance") or tc.get("source") or "").lower()
                if provenance.startswith("runner") or provenance == "seeded":
                    seeded_match = True
                    seeded_match_accepted = seeded_match_accepted or acc
                else:
                    llm_probe_match = True
                    llm_probe_match_accepted = llm_probe_match_accepted or acc
        elif name == "shadow_required_simulation":
            result = tc.get("result") or {}
            shadow_accepted = bool(result.get("accepted")) if isinstance(result, dict) else None
            reasons = result.get("reasons") if isinstance(result, dict) else None
            if isinstance(reasons, list):
                shadow_reasons = [str(r) for r in reasons]

    if seeded_match:
        proposal_source = "seeded_match"
    elif llm_probe_match:
        proposal_source = "llm_probe_match"
    else:
        proposal_source = "llm_off_grid"

    return {
        "run_id": row.get("run_id"),
        "agent_id": row.get("agent_id"),
        "archetype": row.get("archetype"),
        "step": row.get("step"),
        "side": side,
        "quantity_mwh": qty,
        "limit_price_eur_mwh": price,
        "proposal_source": proposal_source,
        "verifier_accepted": row.get("verifier_accepted"),
        "verifier_reason_codes": row.get("verifier_reason_codes"),
        "shadow_accepted": shadow_accepted,
        "shadow_reasons": shadow_reasons,
        "simulate_calls_total": sims_total,
        "simulate_calls_accepted": sims_accepted,
    }


def process_run(run_dir: Path) -> list[dict]:
    trace = run_dir / "traces.jsonl"
    if not trace.exists():
        return []
    out: list[dict] = []
    with trace.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = classify_row(row)
            if label is not None:
                out.append(label)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True,
                        help="Directory containing per-run subdirectories with traces.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for run_dir in sorted(p for p in args.matrix_dir.iterdir() if p.is_dir()):
        if not (run_dir / "traces.jsonl").exists():
            continue
        rows = process_run(run_dir)
        all_rows.extend(rows)
        print(f"{run_dir.name}: {len(rows)} bid proposals labelled", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"wrote {len(all_rows)} labelled proposals to {args.out}", file=sys.stderr)

    # quick summary by run + proposal_source
    from collections import Counter
    bycell: dict[str, Counter] = {}
    for r in all_rows:
        run_id = r["run_id"] or ""
        # cell name = second token in ff-matrix-{cell}-...
        parts = run_id.split("-", 4)
        cell = parts[2] if len(parts) >= 3 else run_id
        bycell.setdefault(cell, Counter())[r["proposal_source"]] += 1
    print("\nproposal_source counts by cell:", file=sys.stderr)
    for cell, c in sorted(bycell.items()):
        total = sum(c.values()) or 1
        share = {k: f"{v/total:.1%}" for k, v in c.items()}
        print(f"  {cell:14s} n={total}  {share}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
