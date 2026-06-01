from __future__ import annotations

import urllib.request
import json

from heimdall_contracts import Persona
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord


def _with_provenance(record: ToolCallRecord, provenance: str) -> ToolCallRecord:
    return record.model_copy(update={"provenance": provenance})


def _with_provenance_all(records: list[ToolCallRecord], provenance: str) -> list[ToolCallRecord]:
    return [_with_provenance(record, provenance) for record in records]


def _tool_provenance_counts(records: list[ToolCallRecord]) -> dict[str, int]:
    counts = {
        "runner_seeded": 0,
        "llm_requested": 0,
        "forced_final": 0,
        "runner_diagnostic": 0,
        "retry": 0,
        "unknown": 0,
    }
    for record in records:
        provenance = record.provenance if record.provenance in counts else "unknown"
        counts[provenance] += 1
    return counts


def _trace_tool_counter_fields(records: list[ToolCallRecord]) -> dict[str, object]:
    counts = _tool_provenance_counts(records)
    return {
        "seeded_tool_call_count": counts["runner_seeded"],
        "llm_tool_call_count": counts["llm_requested"],
        "forced_tool_call_count": counts["forced_final"],
        "diagnostic_tool_call_count": counts["runner_diagnostic"],
        "retry_tool_call_count": counts["retry"],
        "unknown_tool_call_count": counts["unknown"],
        "tool_call_provenance_counts": counts,
    }


def _is_evidence_record(record: ToolCallRecord) -> bool:
    if record.name.startswith("propose_"):
        return False
    if record.name in {
        "deliberation_board",
        "deliberation_phase_summary",
        "deliberation_retry_feedback",
        "deliberation_note_missing",
        "peer_request_unfulfilled",
        "selected_candidate_diagnostics",
        "phase_tool_blocked",
    }:
        return False
    return True


def _evidence_tool_call_count(records: list[ToolCallRecord]) -> int:
    return sum(1 for record in records if record.provenance == "llm_requested" and _is_evidence_record(record))


def _action_relevant_probe_count(persona: Persona, records: list[ToolCallRecord]) -> int:
    from heimdall_ai_society._prompts import _required_simulation_tool, _required_feasibility_tool
    required = _required_simulation_tool(persona)
    feasibility = _required_feasibility_tool(persona)
    return sum(
        1
        for record in records
        if record.provenance == "llm_requested"
        and record.name in {required, feasibility, "get_limit_price_guidance", "get_candidate_sizing_guidance", "get_decision_trace_summary"}
    )


def _has_action_relevant_probe(persona: Persona, records: list[ToolCallRecord]) -> bool:
    return _action_relevant_probe_count(persona, records) > 0


def _has_self_simulator_probe(persona: Persona, records: list[ToolCallRecord]) -> bool:
    from heimdall_ai_society._prompts import _required_simulation_tool
    required = _required_simulation_tool(persona)
    return any(record.name == required and record.provenance == "llm_requested" for record in records)


def _record_controls_acceptance(record: ToolCallRecord) -> bool:
    return record.result.get("authority") == "authoritative" or record.result.get("controls_acceptance") is True


def _has_seeded_accepted_candidate(persona: Persona, records: list[ToolCallRecord]) -> bool:
    """True when an authoritative simulator already accepted a candidate before the LLM acts,
    i.e. the runner pre-seeded a selectable menu (``preprobe_mode='full'`` selector path).
    When False under a bid-seeking objective the LLM must gather its own evidence, so the first
    tool round is forced to call a tool rather than allowing an immediate no-tool abstain."""
    from heimdall_ai_society._prompts import _required_simulation_tool
    required = _required_simulation_tool(persona)
    return any(
        record.name == required
        and record.ok
        and _record_controls_acceptance(record)
        and record.result.get("accepted") is True
        for record in records
    )


def _latest_rejected_required(required: str, records: list[ToolCallRecord]) -> ToolCallRecord | None:
    for record in reversed(records):
        if record.name == required and record.ok and record.result.get("accepted") is not True:
            return record
    return None


def _matching_accepted_simulation(required: str, decision: LLMBidDecision, tool_calls: list[ToolCallRecord]) -> ToolCallRecord | None:
    for record in tool_calls:
        if record.name != required or not record.ok:
            continue
        if not _record_controls_acceptance(record) or record.result.get("accepted") is not True:
            continue
        if record.arguments.get("side") != decision.side:
            continue
        if abs(float(record.arguments.get("quantity_mwh", -1.0)) - float(decision.quantity_mwh or -2.0)) > 1e-9:
            continue
        if abs(float(record.arguments.get("limit_price_eur_mwh", -1e9)) - float(decision.limit_price_eur_mwh or -2e9)) > 1e-9:
            continue
        return record
    return None


def _accepted_candidate_count(persona: Persona, records: list[ToolCallRecord]) -> int:
    from heimdall_ai_society._prompts import _required_simulation_tool
    required = _required_simulation_tool(persona)
    return sum(1 for record in records if record.name == required and record.result.get("accepted") is True)


def _append_selected_candidate_diagnostic(records: list[ToolCallRecord], decision: LLMBidDecision) -> None:
    if decision.action != "bid":
        records.append(
            ToolCallRecord(
                name="selected_candidate_diagnostics",
                arguments={"action": decision.action},
                ok=True,
                result={"ok": True, "selected": False, "reason": "non_bid_action"},
                provenance="runner_diagnostic",
            )
        )
        return
    match = None
    for tool_name in ["simulate_bid", "simulate_ev_bid", "simulate_wind_bid", "simulate_generator_bid", "simulate_retailer_bid", "simulate_renewables_bid"]:
        match = _matching_accepted_simulation(tool_name, decision, records)
        if match is not None:
            break
    records.append(
        ToolCallRecord(
            name="selected_candidate_diagnostics",
            arguments=decision.model_dump(mode="json"),
            ok=True,
            result={
                "ok": True,
                "selected": match is not None,
                "matched_tool": match.name if match is not None else None,
                "candidate": match.arguments if match is not None else None,
                "simulator_result": match.result if match is not None else None,
            },
            provenance="runner_diagnostic",
        )
    )


def _same_bid(left: LLMBidDecision, right: LLMBidDecision) -> bool:
    if left.action != "bid" or right.action != "bid":
        return False
    if left.side != right.side:
        return False
    if abs(float(left.quantity_mwh or -1.0) - float(right.quantity_mwh or -2.0)) > 1e-9:
        return False
    return abs(float(left.limit_price_eur_mwh or -1e9) - float(right.limit_price_eur_mwh or -2e9)) <= 1e-9


def _float_result(payload: dict[str, object], key: str) -> float:
    try:
        value = payload.get(key)
        if value is None and key == "worst_case_profit_eur":
            value = payload.get("rough_worst_case_profit_eur")
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _served_model(base_url: str, api_key: str) -> str | None:
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") or []
        return str(data[0].get("id")) if data else None
    except Exception:
        return None


def _served_models(base_urls: list[str], api_key: str) -> dict[str, str | None]:
    return {base_url: _served_model(base_url, api_key) for base_url in base_urls}


def _llm_failure(exc: Exception) -> LLMBidDecision:
    return LLMBidDecision(
        action="abstain",
        rationale=f"LLM call failed; abstaining for safety: {exc}",
        confidence=0.0,
    )
