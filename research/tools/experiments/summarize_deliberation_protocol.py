from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize bounded-deliberation protocol health before P&L.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories or glob patterns.")
    args = parser.parse_args()
    run_dirs = _expand(args.run_dirs)
    rows = [_summarize_run(path) for path in run_dirs]
    print(json.dumps({"run_count": len(rows), "runs": rows}, indent=2, sort_keys=True))
    return 0


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if (path / "traces.jsonl").exists()]


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    lines = [json.loads(line) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    totals = {
        "agent_ticks": 0,
        "inquiry_tool_ticks": 0,
        "note_ticks": 0,
        "peer_responses": 0,
        "peer_requests": 0,
        "fulfilled_peer_requests": 0,
        "retry_feedback": 0,
        "unsupported_bid_retries": 0,
        "final_bids": 0,
        "llm_self_sim_backed_bids": 0,
        "runner_seeded_calls": 0,
    }
    examples: list[dict[str, Any]] = []
    for row in lines:
        calls = row.get("tool_calls") or []
        has_delib = any(call.get("name") in {"propose_deliberation_note", "deliberation_board", "deliberation_phase_summary"} for call in calls)
        if not has_delib:
            continue
        totals["agent_ticks"] += 1
        totals["runner_seeded_calls"] += int(row.get("seeded_tool_call_count") or 0)
        if any(call.get("provenance") == "llm_requested" and _is_evidence_call(call) for call in calls):
            totals["inquiry_tool_ticks"] += 1
        if any(call.get("name") == "propose_deliberation_note" and call.get("ok") is True for call in calls):
            totals["note_ticks"] += 1
        totals["peer_responses"] += sum(1 for call in calls if call.get("name") == "propose_peer_response" and call.get("ok") is True)
        phase_summary = next((call for call in reversed(calls) if call.get("name") == "deliberation_phase_summary"), None)
        if phase_summary:
            result = phase_summary.get("result") or {}
            request_count = int(result.get("peer_request_count") or 0)
            action_probe_count = int(result.get("action_probe_count") or 0)
            totals["peer_requests"] += request_count
            if request_count and action_probe_count:
                totals["fulfilled_peer_requests"] += request_count
        totals["retry_feedback"] += sum(1 for call in calls if call.get("name") in {"deliberation_retry_feedback", "peer_request_unfulfilled"})
        totals["unsupported_bid_retries"] += sum(
            1
            for call in calls
            if call.get("name") == "deliberation_retry_feedback" and (call.get("arguments") or {}).get("reason") == "unsupported_bid"
        )
        decision = row.get("decision") or {}
        if decision.get("action") == "bid":
            totals["final_bids"] += 1
            if _has_llm_self_simulator(row, calls):
                totals["llm_self_sim_backed_bids"] += 1
        if len(examples) < 5:
            note = next((call for call in calls if call.get("name") == "propose_deliberation_note" and call.get("ok") is True), None)
            response = next((call for call in calls if call.get("name") == "propose_peer_response" and call.get("ok") is True), None)
            if note or response:
                examples.append(
                    {
                        "step": row.get("step"),
                        "agent_id": row.get("agent_id"),
                        "archetype": row.get("archetype"),
                        "note": ((note or {}).get("result") or {}).get("note"),
                        "peer_response": ((response or {}).get("result") or {}).get("response"),
                    }
                )
    agent_ticks = max(1, totals["agent_ticks"])
    peer_requests = max(1, totals["peer_requests"])
    final_bids = max(1, totals["final_bids"])
    return {
        "run_id": run_dir.name,
        "ablation_strategy": summary.get("ablation_strategy"),
        "agent_count": summary.get("agent_count"),
        "ticks": summary.get("ticks"),
        "rates": {
            "inquiry_tool_call_rate": round(totals["inquiry_tool_ticks"] / agent_ticks, 6),
            "deliberation_note_rate": round(totals["note_ticks"] / agent_ticks, 6),
            "peer_request_fulfillment_rate": round(totals["fulfilled_peer_requests"] / peer_requests, 6),
            "accepted_bid_backed_by_llm_requested_simulator_rate": round(totals["llm_self_sim_backed_bids"] / final_bids, 6),
        },
        "counts": totals,
        "examples": examples,
    }


def _is_evidence_call(call: dict[str, Any]) -> bool:
    name = str(call.get("name") or "")
    return not name.startswith("propose_") and name not in {
        "deliberation_board",
        "deliberation_phase_summary",
        "deliberation_retry_feedback",
        "deliberation_note_missing",
        "peer_request_unfulfilled",
        "selected_candidate_diagnostics",
        "phase_tool_blocked",
    }


def _has_llm_self_simulator(row: dict[str, Any], calls: list[dict[str, Any]]) -> bool:
    required_by_archetype = {
        "p2h": "simulate_bid",
        "ev": "simulate_ev_bid",
        "wind": "simulate_wind_bid",
        "generator": "simulate_generator_bid",
        "renewables": "simulate_renewables_bid",
        "retailer": "simulate_retailer_bid",
    }
    decision = row.get("decision") or {}
    required = required_by_archetype.get(str(row.get("archetype")))
    if required is None:
        return False
    for call in calls:
        args = call.get("arguments") or {}
        result = call.get("result") or {}
        if (
            call.get("name") == required
            and call.get("provenance") == "llm_requested"
            and result.get("accepted") is True
            and (result.get("authority") == "authoritative" or result.get("controls_acceptance") is True)
            and args.get("side") == decision.get("side")
            and _same_float(args.get("quantity_mwh"), decision.get("quantity_mwh"))
            and _same_float(args.get("limit_price_eur_mwh"), decision.get("limit_price_eur_mwh"))
        ):
            return True
    return False


def _same_float(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
