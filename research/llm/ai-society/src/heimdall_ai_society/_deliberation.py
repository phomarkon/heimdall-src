from __future__ import annotations

import asyncio
import json

from heimdall_contracts import Persona
from heimdall_ai_society.llm_client import OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.memory import MemoryItem, memory_prompt_context
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tool_policy import policy_for_persona
from heimdall_ai_society.tools import AgentToolExecutor, decision_from_tool_calls
from packages.simulator import ScenarioAssetStateStore

from heimdall_ai_society._trace_helpers import (
    _append_selected_candidate_diagnostic,
    _evidence_tool_call_count,
    _has_action_relevant_probe,
    _has_self_simulator_probe,
    _is_evidence_record,
    _matching_accepted_simulation,
    _action_relevant_probe_count,
)
from heimdall_ai_society._prompts import (
    _compact_tool_records_for_prompt,
    _prompt,
    _required_simulation_tool,
)
from heimdall_ai_society._decision import (
    _execute_agent_tool,
    _executor_for_persona,
    _run_tool_round,
    _phase_tool_specs,
    _needs_authoritative_simulation,
    _downgrade_unsupported_bid,
)
from heimdall_ai_society._deterministic import _deterministic_decision


def _deliberation_diagnostic(
    name: str,
    arguments: dict[str, object],
    result: dict[str, object],
    *,
    provenance: str = "runner_diagnostic",
) -> ToolCallRecord:
    return ToolCallRecord(name=name, arguments=arguments, ok=bool(result.get("ok", True)), result=result, provenance=provenance)


async def _run_deliberation_protocol_tick(
    *,
    personas: list[Persona],
    tick: TickContext,
    persona_ticks: list[TickContext],
    llm: OpenAICompatibleLLMClient | None,
    profile: str,
    objective: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    max_tool_rounds: int,
    final_bid_guard: str,
    safety_toolset: str,
    candidate_sizing_mode: str,
    candidate_sizing_cap_fraction: float,
    candidate_sizing_min_mwh: float,
    candidate_sizing_max_candidates: int,
    deliberation_inquiry_rounds: int,
    deliberation_action_rounds: int,
    deliberation_min_tool_calls: int,
    deliberation_require_action_probe: bool,
    deliberation_max_peer_notes: int,
    memory_by_agent: dict[str, list[MemoryItem]],
    tool_cache: dict[tuple[str, str], ToolCallRecord],
) -> list[tuple[LLMBidDecision, list[ToolCallRecord]]]:
    from heimdall_ai_society._strategies import _agent_role
    if llm is None:
        return [(_deterministic_decision(persona, persona_ticks[idx]), []) for idx, persona in enumerate(personas)]
    inquiry_tasks = [
        _deliberation_inquiry_for_persona(
            persona=persona,
            tick=persona_ticks[idx],
            llm=llm,
            agent_role=_agent_role(profile, idx),
            objective=objective,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            simulator_semaphore=simulator_semaphore,
            safety_toolset=safety_toolset,
            inquiry_rounds=deliberation_inquiry_rounds,
            min_tool_calls=deliberation_min_tool_calls,
            memory_context=memory_prompt_context(memory_by_agent.get(persona.agent_id, [])),
            tool_cache=tool_cache,
        )
        for idx, persona in enumerate(personas)
    ]
    inquiry_records = await asyncio.gather(*inquiry_tasks)
    board = _build_deliberation_board(
        tick=tick,
        personas=personas,
        records_by_agent=inquiry_records,
        max_peer_notes=deliberation_max_peer_notes,
    )
    action_tasks = [
        _deliberation_final_for_persona(
            persona=persona,
            tick=persona_ticks[idx],
            llm=llm,
            agent_role=_agent_role(profile, idx),
            objective=objective,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            simulator_semaphore=simulator_semaphore,
            max_tool_rounds=max_tool_rounds,
            final_bid_guard=final_bid_guard,
            safety_toolset=safety_toolset,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
            action_rounds=deliberation_action_rounds,
            require_action_probe=deliberation_require_action_probe,
            board=board,
            inquiry_records=inquiry_records[idx],
            memory_context=memory_prompt_context(memory_by_agent.get(persona.agent_id, [])),
            tool_cache=tool_cache,
        )
        for idx, persona in enumerate(personas)
    ]
    return list(await asyncio.gather(*action_tasks))


async def _deliberation_inquiry_for_persona(
    *,
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient,
    agent_role: str,
    objective: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    safety_toolset: str,
    inquiry_rounds: int,
    min_tool_calls: int,
    memory_context: dict[str, object] | None,
    tool_cache: dict[tuple[str, str], ToolCallRecord],
) -> list[ToolCallRecord]:
    executor = _executor_for_persona(
        persona=persona,
        tick=tick,
        data_tools=data_tools,
        simulator_tool=simulator_tool,
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache=tool_cache,
    )
    messages = _prompt(
        persona,
        tick,
        agent_role=agent_role,
        objective=objective,
        ablation_strategy="comm_deliberation_protocol",
        safety_toolset=safety_toolset,
        communication_context=_minimal_deliberation_context(tick=tick),
        memory_context=memory_context,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Deliberation phase 1: inquiry. Call at least one real evidence tool before any note. "
                "Do not call propose_action. Inspect context, uncertainty, candidate sizing, feasibility, or simulator evidence relevant to your role."
            ),
        }
    )
    records: list[ToolCallRecord] = []
    inquiry_tools = _phase_tool_specs(safety_toolset, phase="inquiry")
    rounds = max(1, inquiry_rounds)
    for _ in range(rounds):
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=inquiry_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
        )
    if _evidence_tool_call_count(records) < min_tool_calls:
        feedback = _deliberation_diagnostic(
            "deliberation_retry_feedback",
            {"phase": "inquiry", "reason": "missing_required_tool_call"},
            {"ok": True, "message": "call at least one evidence tool first"},
            provenance="retry",
        )
        records.append(feedback)
        messages.append({"role": "user", "content": "Protocol retry: call at least one evidence tool first, then stop."})
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=inquiry_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
        )
    if _evidence_tool_call_count(records) < min_tool_calls:
        records.append(
            _deliberation_diagnostic(
                "deliberation_retry_feedback",
                {"phase": "inquiry", "reason": "forced_minimum_evidence_tool"},
                {"ok": True, "message": "protocol forced run_forecaster after no autonomous evidence call"},
                provenance="retry",
            )
        )
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": "You still have not called an evidence tool. Call run_forecaster now so your deliberation note has real tool evidence.",
                },
            ],
            records=records,
            tools=inquiry_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
            tool_choice={"type": "function", "function": {"name": "run_forecaster"}},
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Now call propose_deliberation_note exactly once. Cite the evidence_refs you used and request a peer, "
                "archetype, tool, or candidate probe when another agent should check something."
            ),
        }
    )
    await _run_tool_round(
        llm=llm,
        executor=executor,
        messages=messages,
        records=records,
        tools=_phase_tool_specs(safety_toolset, phase="note"),
        simulator_semaphore=simulator_semaphore,
        provenance="llm_requested",
        tool_choice={"type": "function", "function": {"name": "propose_deliberation_note"}},
    )
    if not any(record.name == "propose_deliberation_note" and record.ok for record in records):
        records.append(
            _deliberation_diagnostic(
                "deliberation_note_missing",
                {"phase": "inquiry"},
                {"ok": True, "reason": "model did not provide a valid deliberation note"},
            )
        )
    return records


async def _deliberation_final_for_persona(
    *,
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient,
    agent_role: str,
    objective: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    max_tool_rounds: int,
    final_bid_guard: str,
    safety_toolset: str,
    candidate_sizing_mode: str,
    candidate_sizing_cap_fraction: float,
    candidate_sizing_min_mwh: float,
    candidate_sizing_max_candidates: int,
    action_rounds: int,
    require_action_probe: bool,
    board: dict[str, object],
    inquiry_records: list[ToolCallRecord],
    memory_context: dict[str, object] | None,
    tool_cache: dict[tuple[str, str], ToolCallRecord],
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    executor = _executor_for_persona(
        persona=persona,
        tick=tick,
        data_tools=data_tools,
        simulator_tool=simulator_tool,
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache=tool_cache,
    )
    records = list(inquiry_records)
    peer_requests = _peer_requests_for_persona(board, persona)
    records.append(
        _deliberation_diagnostic(
            "deliberation_board",
            {"strategy": "comm_deliberation_protocol"},
            {"ok": True, "authority": "advisory", "board": board, "peer_requests_for_agent": peer_requests},
        )
    )
    messages = _prompt(
        persona,
        tick,
        agent_role=agent_role,
        objective=objective,
        ablation_strategy="comm_deliberation_protocol",
        safety_toolset=safety_toolset,
        communication_context=board,
        memory_context=memory_context,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Deliberation phase 2: exchange and action probe. Read the board, respond to at least one useful peer note "
                "when you agree or object, and run your own action-relevant feasibility or simulator tool before bidding. "
                + json.dumps({"peer_requests_for_you": peer_requests}, sort_keys=True)
            ),
        }
    )
    action_tools = _phase_tool_specs(safety_toolset, phase="action_probe")
    if peer_requests:
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": "Respond to the deliberation board before probing. Call propose_peer_response exactly once.",
                },
            ],
            records=records,
            tools=action_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
            tool_choice={"type": "function", "function": {"name": "propose_peer_response"}},
        )
    for _ in range(max(1, action_rounds)):
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=action_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
        )
    if require_action_probe and policy_for_persona(persona).can_submit_bid and peer_requests and not _has_action_relevant_probe(persona, records):
        records.append(
            _deliberation_diagnostic(
                "peer_request_unfulfilled",
                {"agent_id": persona.agent_id, "requested_count": len(peer_requests)},
                {"ok": True, "reason": "action-capable agent did not run a requested action probe"},
                provenance="retry",
            )
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Protocol retry: peers requested an action probe. Run your archetype feasibility or simulator tool now, "
                    "or be explicit in final action that you are watching/abstaining because no safe probe is justified."
                ),
            }
        )
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=action_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
        )
    if require_action_probe and policy_for_persona(persona).can_submit_bid and peer_requests and not _has_action_relevant_probe(persona, records):
        records.append(
            _deliberation_diagnostic(
                "deliberation_retry_feedback",
                {"phase": "action_probe", "reason": "forced_candidate_sizing_probe"},
                {"ok": True, "message": "protocol forced candidate sizing guidance after ignored peer request"},
                provenance="retry",
            )
        )
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": "Call get_candidate_sizing_guidance now for your archetype so final action can cite a real action probe.",
                },
            ],
            records=records,
            tools=action_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
            tool_choice={"type": "function", "function": {"name": "get_candidate_sizing_guidance"}},
        )
    if require_action_probe and policy_for_persona(persona).can_submit_bid and peer_requests and not _has_self_simulator_probe(persona, records):
        peer_candidate = _first_peer_candidate(peer_requests)
        if peer_candidate is not None:
            records.append(
                _deliberation_diagnostic(
                    "deliberation_retry_feedback",
                    {"phase": "action_probe", "reason": "peer_requested_self_simulator_probe"},
                    {"ok": True, "message": "protocol ran own simulator for a peer-requested candidate"},
                    provenance="retry",
                )
            )
            sim_record = await _execute_agent_tool(
                executor,
                _required_simulation_tool(persona),
                peer_candidate,
                simulator_semaphore,
                provenance="llm_requested",
            )
            records.append(sim_record)
            clear_candidate = _clear_seeking_candidate(peer_candidate)
            if clear_candidate != peer_candidate:
                clear_record = await _execute_agent_tool(
                    executor,
                    _required_simulation_tool(persona),
                    clear_candidate,
                    simulator_semaphore,
                    provenance="llm_requested",
                )
                records.append(clear_record)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your own simulator results for peer-requested and clear-seeking candidates are now visible. "
                        "For up bids, lower limit prices clear more easily; for down bids, higher limit prices clear more easily. "
                        "Prefer an accepted clear-seeking self-simulator candidate when available: "
                    )
                    + json.dumps(_compact_tool_records_for_prompt(records[-2:]), sort_keys=True),
                }
            )
    records.append(
        _deliberation_diagnostic(
            "deliberation_phase_summary",
            {"agent_id": persona.agent_id},
            {
                "ok": True,
                "inquiry_tool_calls": _evidence_tool_call_count(inquiry_records),
                "note_count": sum(1 for record in records if record.name == "propose_deliberation_note" and record.ok),
                "peer_response_count": sum(1 for record in records if record.name == "propose_peer_response" and record.ok),
                "peer_request_count": len(peer_requests),
                "action_probe_count": _action_relevant_probe_count(persona, records),
            },
        )
    )
    decision_records_before_final = len(records)
    final_tools = _phase_tool_specs(safety_toolset, phase="final")
    messages.append(
        {
            "role": "user",
            "content": (
                "Deliberation phase 3: final action. Call propose_action exactly once. A bid must exactly match your own "
                f"accepted {_required_simulation_tool(persona)} result; peer evidence is advisory only. If no accepted self-simulator exists, watch or abstain."
                " When multiple accepted self-simulator candidates exist, prefer the more clear-seeking one: lower limit price for up, higher limit price for down."
            ),
        }
    )
    await _run_tool_round(
        llm=llm,
        executor=executor,
        messages=messages,
        records=records,
        tools=final_tools,
        simulator_semaphore=simulator_semaphore,
        provenance="forced_final",
        tool_choice={"type": "function", "function": {"name": "propose_action"}},
    )
    decision = decision_from_tool_calls(records[decision_records_before_final:]) or decision_from_tool_calls(records)
    if decision is None:
        _append_selected_candidate_diagnostic(records, LLMBidDecision(action="abstain", rationale="model did not call propose_action", confidence=0.0))
        return LLMBidDecision(action="abstain", rationale="model did not call propose_action", confidence=0.0), records
    if final_bid_guard == "simulator_exact_match" and _needs_authoritative_simulation(persona, decision, records):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Protocol retry: you proposed a bid without your own accepted authoritative simulator evidence. "
                    f"Call {_required_simulation_tool(persona)} for that candidate and then call propose_action again; otherwise watch or abstain."
                ),
            }
        )
        records.append(
            _deliberation_diagnostic(
                "deliberation_retry_feedback",
                {"phase": "final_action", "reason": "unsupported_bid"},
                {"ok": True, "message": "final bid requires own accepted simulator evidence"},
                provenance="retry",
            )
        )
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=action_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="llm_requested",
        )
        await _run_tool_round(
            llm=llm,
            executor=executor,
            messages=messages,
            records=records,
            tools=final_tools,
            simulator_semaphore=simulator_semaphore,
            provenance="forced_final",
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
        )
        decision = decision_from_tool_calls(records) or decision
        if _needs_authoritative_simulation(persona, decision, records):
            decision = _downgrade_unsupported_bid(persona, decision, records)
    _append_selected_candidate_diagnostic(records, decision)
    return decision, records


def _build_deliberation_board(
    *,
    tick: TickContext,
    personas: list[Persona],
    records_by_agent: list[list[ToolCallRecord]],
    max_peer_notes: int,
) -> dict[str, object]:
    notes = []
    peer_requests = []
    for persona, records in zip(personas, records_by_agent, strict=True):
        note_record = next((record for record in reversed(records) if record.name == "propose_deliberation_note" and record.ok), None)
        note = note_record.result.get("note", {}) if note_record is not None else {}
        compact_evidence = _compact_tool_records_for_prompt([
            record for record in records
            if record.provenance == "llm_requested" and _is_evidence_record(record)
        ])[:6]
        item = {
            "agent_id": persona.agent_id,
            "archetype": persona.archetype.value,
            "risk_attitude": persona.risk_attitude.value,
            "note": note,
            "evidence": compact_evidence,
        }
        notes.append(item)
        if isinstance(note, dict) and any(note.get(key) for key in ["requested_peer_id", "requested_archetype", "requested_tool", "requested_candidate"]):
            peer_requests.append(
                {
                    "from_agent_id": persona.agent_id,
                    "requested_peer_id": note.get("requested_peer_id"),
                    "requested_archetype": note.get("requested_archetype"),
                    "requested_tool": note.get("requested_tool"),
                    "requested_candidate": note.get("requested_candidate"),
                    "side_belief": note.get("side_belief"),
                }
            )
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    return {
        "strategy": "comm_deliberation_protocol",
        "phase": "exchange",
        "market_digest": {
            "timestamp": tick.timestamp.isoformat(),
            "zone": tick.forecast.zone,
            "last_price_eur_mwh": tick.market_price_eur_mwh,
            "mfrr_up_interval_eur_mwh": [up_lower, up_upper],
            "mfrr_down_interval_eur_mwh": [down_lower, down_upper],
        },
        "notes": notes[:max_peer_notes],
        "peer_requests": peer_requests[:max_peer_notes],
        "note_count": len(notes),
        "peer_request_count": len(peer_requests),
    }


def _peer_requests_for_persona(board: dict[str, object], persona: Persona) -> list[dict[str, object]]:
    requests = board.get("peer_requests", [])
    if not isinstance(requests, list):
        return []
    matched = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        requested_peer = request.get("requested_peer_id")
        requested_archetype = request.get("requested_archetype")
        if requested_peer and requested_peer == persona.agent_id:
            matched.append(request)
        elif requested_archetype and str(requested_archetype).lower() == persona.archetype.value:
            matched.append(request)
        elif not requested_peer and not requested_archetype and policy_for_persona(persona).can_submit_bid:
            matched.append(request)
    return matched[:4]


def _minimal_deliberation_context(*, tick: TickContext) -> dict[str, object]:
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    return {
        "strategy": "comm_deliberation_protocol",
        "phase": "inquiry",
        "market_digest": {
            "timestamp": tick.timestamp.isoformat(),
            "zone": tick.forecast.zone,
            "last_price_eur_mwh": tick.market_price_eur_mwh,
            "mfrr_up_interval_eur_mwh": [up_lower, up_upper],
            "mfrr_down_interval_eur_mwh": [down_lower, down_upper],
        },
    }


def _first_peer_candidate(peer_requests: list[dict[str, object]]) -> dict[str, object] | None:
    for request in peer_requests:
        candidate = request.get("requested_candidate")
        if not isinstance(candidate, dict):
            continue
        if {"side", "quantity_mwh", "limit_price_eur_mwh"}.issubset(candidate):
            side = candidate.get("side")
            if side not in {"up", "down"}:
                continue
            try:
                return {
                    "side": side,
                    "quantity_mwh": float(candidate["quantity_mwh"]),
                    "limit_price_eur_mwh": float(candidate["limit_price_eur_mwh"]),
                }
            except (TypeError, ValueError):
                continue
    return None


def _clear_seeking_candidate(candidate: dict[str, object]) -> dict[str, object]:
    side = candidate.get("side")
    try:
        price = float(candidate["limit_price_eur_mwh"])
    except (KeyError, TypeError, ValueError):
        return candidate
    adjusted = dict(candidate)
    if side == "up":
        adjusted["limit_price_eur_mwh"] = round(price - 10.0, 2)
    elif side == "down":
        adjusted["limit_price_eur_mwh"] = round(price + 10.0, 2)
    return adjusted


def _empty_deliberation_totals() -> dict[str, int]:
    return {
        "agent_ticks": 0,
        "inquiry_tool_agent_ticks": 0,
        "deliberation_note_agent_ticks": 0,
        "peer_response_count": 0,
        "peer_request_count": 0,
        "peer_request_fulfilled_count": 0,
        "retry_feedback_count": 0,
        "action_capable_agent_ticks": 0,
        "action_probe_compliant_ticks": 0,
        "unsupported_bid_proposal_count": 0,
        "final_bid_count": 0,
        "accepted_bid_llm_self_simulator_count": 0,
    }


def _accumulate_deliberation_totals(
    totals: dict[str, int],
    *,
    persona: Persona,
    decision: LLMBidDecision,
    records: list[ToolCallRecord],
) -> None:
    if not any(record.name in {"propose_deliberation_note", "deliberation_board", "deliberation_phase_summary"} for record in records):
        return
    totals["agent_ticks"] += 1
    if _evidence_tool_call_count(records) > 0:
        totals["inquiry_tool_agent_ticks"] += 1
    if any(record.name == "propose_deliberation_note" and record.ok for record in records):
        totals["deliberation_note_agent_ticks"] += 1
    totals["peer_response_count"] += sum(1 for record in records if record.name == "propose_peer_response" and record.ok)
    phase_summary = next((record for record in reversed(records) if record.name == "deliberation_phase_summary"), None)
    if phase_summary is not None:
        request_count = int(phase_summary.result.get("peer_request_count", 0) or 0)
        action_probe_count = int(phase_summary.result.get("action_probe_count", 0) or 0)
        totals["peer_request_count"] += request_count
        if request_count and action_probe_count:
            totals["peer_request_fulfilled_count"] += request_count
    totals["retry_feedback_count"] += sum(1 for record in records if record.name in {"deliberation_retry_feedback", "peer_request_unfulfilled"})
    if policy_for_persona(persona).can_submit_bid:
        totals["action_capable_agent_ticks"] += 1
        if _has_action_relevant_probe(persona, records) or decision.action != "bid":
            totals["action_probe_compliant_ticks"] += 1
    if any(record.name == "deliberation_retry_feedback" and record.arguments.get("reason") == "unsupported_bid" for record in records):
        totals["unsupported_bid_proposal_count"] += 1
    if decision.action == "bid":
        totals["final_bid_count"] += 1
        match = _matching_accepted_simulation(_required_simulation_tool(persona), decision, records)
        if match is not None and match.provenance == "llm_requested":
            totals["accepted_bid_llm_self_simulator_count"] += 1


def _finalize_deliberation_totals(totals: dict[str, int]) -> dict[str, object]:
    agent_ticks = max(1, totals["agent_ticks"])
    action_ticks = max(1, totals["action_capable_agent_ticks"])
    peer_requests = max(1, totals["peer_request_count"])
    final_bids = max(1, totals["final_bid_count"])
    return {
        **totals,
        "inquiry_tool_call_rate": round(totals["inquiry_tool_agent_ticks"] / agent_ticks, 6),
        "deliberation_note_rate": round(totals["deliberation_note_agent_ticks"] / agent_ticks, 6),
        "action_probe_compliance_rate": round(totals["action_probe_compliant_ticks"] / action_ticks, 6),
        "peer_request_fulfillment_rate": round(totals["peer_request_fulfilled_count"] / peer_requests, 6),
        "accepted_bid_backed_by_llm_requested_simulator_rate": round(totals["accepted_bid_llm_self_simulator_count"] / final_bids, 6),
    }
