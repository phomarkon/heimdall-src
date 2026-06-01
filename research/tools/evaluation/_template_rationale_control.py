"""Stress test: can a DETERMINISTIC template match the LLM's explanation quality?

The completeness/selectivity headline compares the LLM against `det_rich`, whose rationale is a
deliberately thin constant string. The fair, adversarial control is a deterministic template that
string-formats the SAME tick's tool outputs into a driver-named rationale — no LLM. If that template
matches the LLM on coverage/selectivity/faithfulness, then the explanation *artifact* does not require
an LLM, and the LLM's value narrows to autonomy/open-endedness (which this data cannot show).

This script reads a det/LLM run's traces, rebuilds the ground-truth drivers from each bid's tool
results, synthesizes a template rationale, writes a shadow run dir, and prints the path so
evaluate_explanation_quality.py can score it next to the real arms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_explanation_quality import _trace_ground_truth  # noqa: E402


def template_rationale(decision: dict, g: dict) -> str:
    """Deterministic, faithful-by-construction rationale from founded trace drivers only."""
    parts: list[str] = []
    side = decision.get("side")
    # side justification (contrastive) — only state edges that are really present
    if g["edge_present"] and g["up_edge"] is not None:
        up, dn = g["up_edge"], g["down_edge"]
        if side == "up":
            parts.append(f"Up-edge {up:.1f} EUR/MWh exceeds down-edge {dn:.1f} EUR/MWh, so an up-bid is chosen.")
        elif side == "down":
            parts.append(f"Down-edge {dn:.1f} EUR/MWh dominates up-edge {up:.1f} EUR/MWh, so a down-bid is chosen.")
    # profit
    if g["sim_profit"]:
        parts.append(f"Simulator-accepted with worst-case/expected profit around {max(g['sim_profit']):.1f} EUR.")
    elif g["accepted"]:
        parts.append("Candidate accepted by the simulator.")
    # regime
    if g["regime_label"]:
        parts.append(f"Market regime is {g['regime_label'].replace('_', ' ')}.")
    # uncertainty (selectivity: flag it explicitly when high/ambiguous)
    if g["uncertainty_label"]:
        u = g["uncertainty_label"]
        if u == "high" or g["side_ambiguity"]:
            parts.append(f"Uncertainty is {u} with side ambiguity, so size is kept conservative.")
        else:
            parts.append(f"Uncertainty is {u}.")
    # constraint (only when really present)
    if g["pressure_label"] and g["pressure_label"] not in ("neutral", "balanced", None):
        parts.append(f"Border shows {g['pressure_label'].replace('_', ' ')}.")
    if g["outage_impact"]:
        parts.append(f"Outage impact: {g['outage_impact']}.")
    return " ".join(parts) or "Deterministic template: no founded drivers."


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for line in (src / "traces.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        decision = rec.get("decision") or {}
        if decision.get("action") == "bid":
            g = _trace_ground_truth(rec.get("tool_calls") or [])
            new = template_rationale(decision, g)
            decision["rationale"] = new
            rec["decision"] = decision
            rec["rationale"] = new
        out_lines.append(json.dumps(rec))
    (dst / "traces.jsonl").write_text("\n".join(out_lines) + "\n")
    print(f"wrote {dst}/traces.jsonl ({len(out_lines)} records)")


if __name__ == "__main__":
    main()
