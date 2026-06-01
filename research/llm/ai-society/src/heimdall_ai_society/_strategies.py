from __future__ import annotations

import asyncio
import json

from heimdall_contracts import Persona
from heimdall_ai_society.llm_client import LLMClientError, OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.memory import MemoryItem, memory_prompt_context
from heimdall_ai_society.schemas import LLMBidDecision, SocietyTraceRecord, ToolCallRecord
from heimdall_ai_society.tool_policy import policy_for_persona
from heimdall_ai_society.tools import AgentToolExecutor, commit_asset_state_from_record, decision_from_tool_calls
from packages.simulator import ScenarioAssetStateStore

from heimdall_ai_society._trace_helpers import (
    _append_selected_candidate_diagnostic,
    _float_result,
    _matching_accepted_simulation,
    _trace_tool_counter_fields,
)
from heimdall_ai_society._prompts import _required_simulation_tool
from heimdall_ai_society._candidates import _matching_candidate_row, _rank_seeded_candidates
from heimdall_ai_society._context import (
    _communication_context,
    _merge_watch_reasons,
    _peer_summary,
)
from heimdall_ai_society._decision import (
    _decide_for_persona,
    _is_info_archetype,
)
from heimdall_ai_society._deterministic import (
    _critic_action_record,
    _critic_tool_specs,
    _high_fill_candidate_sort_key,
)


def _uses_society_communication(strategy: str) -> bool:
    return strategy.startswith("comm_")


def _uses_chair(strategy: str) -> bool:
    return strategy in {
        "comm_society_chair",
        "comm_society_chair_2agree",
        "comm_society_chair_riskveto",
        "comm_society_chair_intel",
    }


async def _run_info_then_action_tick(
    *,
    personas: list[Persona],
    tick: TickContext,
    persona_ticks: list[TickContext],
    llm: OpenAICompatibleLLMClient | None,
    profile: str,
    tool_mode: str,
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    max_tool_rounds: int,
    final_bid_guard: str,
    safety_toolset: str,
    preprobe_mode: str,
    candidate_sizing_mode: str,
    candidate_sizing_cap_fraction: float,
    candidate_sizing_min_mwh: float,
    candidate_sizing_max_candidates: int,
    chooser_mode: str,
    shared_digest: dict[str, object],
    memory_by_agent: dict[str, list[MemoryItem]],
    tool_cache: dict[tuple[str, str], ToolCallRecord],
    forecast_diversity_context: dict[str, object] | None = None,
) -> list[tuple[LLMBidDecision, list[ToolCallRecord]]]:
    info_indices = [
        idx for idx, persona in enumerate(personas)
        if _is_info_archetype(persona.archetype.value)
    ]
    action_indices = [idx for idx in range(len(personas)) if idx not in info_indices]
    outcomes: list[tuple[LLMBidDecision, list[ToolCallRecord]] | None] = [None] * len(personas)
    base_context = _communication_context(
        strategy=ablation_strategy,
        tick=tick,
        personas=personas,
        peer_summaries=[],
        shared_digest=shared_digest,
    )
    info_tasks = [
        _decide_for_persona(
            personas[idx],
            persona_ticks[idx],
            llm,
            agent_role=_agent_role(profile, idx),
            tool_mode=tool_mode,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            simulator_semaphore=simulator_semaphore,
            max_tool_rounds=max_tool_rounds,
            final_bid_guard=final_bid_guard,
            safety_toolset=safety_toolset,
            preprobe_mode=preprobe_mode,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
            chooser_mode=chooser_mode,
            communication_context=base_context,
            memory_context=memory_prompt_context(memory_by_agent.get(personas[idx].agent_id, [])),
            tool_cache=tool_cache,
            forecast_diversity_context=forecast_diversity_context,
        )
        for idx in info_indices
    ]
    info_results = await asyncio.gather(*info_tasks) if info_tasks else []
    info_summaries = []
    for idx, result in zip(info_indices, info_results, strict=True):
        outcomes[idx] = result
        info_summaries.append(_peer_summary(personas[idx], result[0], result[1]))
    action_context = _communication_context(
        strategy=ablation_strategy,
        tick=tick,
        personas=personas,
        peer_summaries=info_summaries,
        shared_digest=shared_digest,
    )
    action_context["expert_summaries"] = info_summaries
    action_tasks = [
        _decide_for_persona(
            personas[idx],
            persona_ticks[idx],
            llm,
            agent_role=_agent_role(profile, idx),
            tool_mode=tool_mode,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            simulator_semaphore=simulator_semaphore,
            max_tool_rounds=max_tool_rounds,
            final_bid_guard=final_bid_guard,
            safety_toolset=safety_toolset,
            preprobe_mode=preprobe_mode,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
            chooser_mode=chooser_mode,
            communication_context=action_context,
            memory_context=memory_prompt_context(memory_by_agent.get(personas[idx].agent_id, [])),
            tool_cache=tool_cache,
            forecast_diversity_context=forecast_diversity_context,
        )
        for idx in action_indices
    ]
    action_results = await asyncio.gather(*action_tasks) if action_tasks else []
    for idx, result in zip(action_indices, action_results, strict=True):
        outcomes[idx] = result
    return [item for item in outcomes if item is not None]


async def _central_supervisor_decision(
    *,
    llm: OpenAICompatibleLLMClient | None,
    context: dict[str, object],
    candidates: list[dict[str, object]],
) -> LLMBidDecision:
    if not candidates or llm is None:
        return LLMBidDecision(
            action="watch" if candidates else "abstain",
            rationale="central supervisor found no executable accepted candidate" if not candidates else "central supervisor LLM unavailable; holding accepted candidates for audit",
            confidence=0.3,
            watch_label="watch" if candidates else "ignore",
            opportunity_label="weak" if candidates else "none",
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are Heimdall's central market supervisor. Specialists report isolated evidence, but only you may argue for the final action. "
                "You do not invent, resize, reprice, amend, or cancel orders. Select at most one exact accepted candidate from accepted_candidates, or choose watch/abstain. "
                "A deterministic execution gateway will reject any mutation. The soft quota is pressure, not permission to bypass simulator evidence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instructions": [
                        "When quota_status is behind and accepted_candidates is non-empty, strongly prefer selecting the best justified exact candidate.",
                        "Do not accept all or reject all by habit; explain why this tick should consume or preserve capital.",
                        "Use watch when accepted candidates exist but conflict, risk, or weak economics justify preserving quota.",
                        "Use abstain only when no accepted candidate and no useful watch signal exists.",
                    ],
                    **context,
                },
                sort_keys=True,
            ),
        },
    ]
    try:
        message = await llm.tool_round(
            messages,
            _critic_tool_specs(),
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
        )
    except LLMClientError as exc:
        return LLMBidDecision(action="watch", rationale=f"central supervisor LLM failed; preserving candidates for audit: {exc}", confidence=0.0)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        if function.get("name") != "propose_action":
            continue
        try:
            return LLMBidDecision.model_validate(json.loads(function.get("arguments") or "{}"))
        except Exception:
            continue
    return LLMBidDecision(action="watch", rationale="central supervisor did not return a valid action; preserving candidates for audit", confidence=0.0)


def _execution_gateway_validate(
    proposed: LLMBidDecision,
    candidates: list[dict[str, object]],
    *,
    max_orders_per_tick: int,
) -> tuple[LLMBidDecision, ToolCallRecord]:
    if proposed.action != "bid":
        return proposed, ToolCallRecord(
            name="execution_gateway_validation",
            arguments=proposed.model_dump(mode="json"),
            ok=True,
            result={"ok": True, "gateway_outcome": "non_bid", "executed": False},
            provenance="runner_diagnostic",
        )
    if max_orders_per_tick < 1:
        final = _gateway_downgrade(proposed, "too_many_orders")
        return final, _gateway_record(proposed, final, "too_many_orders", None)
    match = _matching_candidate_row(proposed, candidates)
    if match is None:
        final = _gateway_downgrade(proposed, "mutation_or_unbacked")
        return final, _gateway_record(proposed, final, "mutation_or_unbacked", None)
    final = proposed.model_copy(update={"rationale": f"central supervisor selected gateway-validated specialist candidate: {proposed.rationale}"})
    return final, _gateway_record(proposed, final, "executed", match)


def _gateway_downgrade(proposed: LLMBidDecision, reason: str) -> LLMBidDecision:
    return LLMBidDecision(
        action="watch",
        rationale=f"execution gateway downgraded supervisor bid to watch: {reason}",
        confidence=min(proposed.confidence, 0.4),
        watch_label="must_watch",
        risk_label="high",
        uncertainty_label=proposed.uncertainty_label,
        opportunity_label=proposed.opportunity_label,
        watch_reasons=_merge_watch_reasons(proposed.watch_reasons, ["accepted_bid_available"]),
        priority_label="high",
        priority_score=max(proposed.priority_score, 0.7),
        operator_action="inspect",
        priority_reason="accepted_candidate",
    )


def _gateway_record(
    proposed: LLMBidDecision,
    final: LLMBidDecision,
    outcome: str,
    selected_candidate: dict[str, object] | None,
) -> ToolCallRecord:
    return ToolCallRecord(
        name="execution_gateway_validation",
        arguments=proposed.model_dump(mode="json"),
        ok=True,
        result={
            "ok": True,
            "gateway_outcome": outcome,
            "executed": outcome == "executed",
            "selected_candidate": selected_candidate,
            "final_decision": final.model_dump(mode="json"),
        },
        provenance="runner_diagnostic",
    )


def _supervisor_quota_status(quota_state: dict[str, object]) -> str:
    target = int(quota_state.get("run_target_bids") or 0)
    seen = int(quota_state.get("seen_ticks") or 0)
    executed = int(quota_state.get("executed_bids") or 0)
    soft_per_tick = target / max(1, int(quota_state.get("total_ticks") or 24))
    expected_so_far = soft_per_tick * seen
    if executed + 0.75 < expected_so_far:
        return "behind"
    if executed > expected_so_far + 1.25:
        return "ahead"
    return "on_track"


def _supervisor_candidate_menu(
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
    tick: TickContext,
) -> list[dict[str, object]]:
    rows = []
    for persona, role, _decision, tool_calls, _verdict, _reasons in inputs:
        source_records = [
            record
            for record in tool_calls
            if record.name in {"simulate_bid", "simulate_ev_bid", "simulate_wind_bid", "simulate_generator_bid", "simulate_retailer_bid", "simulate_renewables_bid"}
        ]
        for row in _rank_seeded_candidates(tool_calls, tick):
            if row.get("accepted") is not True:
                continue
            source = next(
                (
                    record
                    for record in source_records
                    if record.arguments == row.get("arguments") and record.result.get("accepted") is True
                ),
                None,
            )
            rows.append(
                {
                    **row,
                    "agent_id": persona.agent_id,
                    "archetype": persona.archetype.value,
                    "agent_role": role,
                    "risk_attitude": persona.risk_attitude.value,
                    "forecaster_id": persona.forecaster_id,
                    "tool": source.name if source is not None else None,
                    "simulator_result": source.result if source is not None else {},
                }
            )
    return sorted(rows, key=_high_fill_candidate_sort_key, reverse=True)


def _commit_supervisor_selected_asset_state(
    *,
    supervisor_record: SocietyTraceRecord,
    personas: list[Persona],
    simulator_tool: object | None,
    asset_state_store: ScenarioAssetStateStore,
    tick: TickContext,
) -> None:
    gateway = next((record for record in supervisor_record.tool_calls if record.name == "execution_gateway_validation"), None)
    if gateway is None:
        return
    selected = gateway.result.get("selected_candidate")
    if not isinstance(selected, dict):
        return
    agent_id = selected.get("agent_id")
    persona = next((item for item in personas if item.agent_id == agent_id), None)
    if persona is None or persona.archetype.value == "p2h":
        return
    simulator_result = selected.get("simulator_result")
    arguments = selected.get("arguments")
    tool_name = selected.get("tool")
    if not isinstance(simulator_result, dict) or not isinstance(arguments, dict) or not isinstance(tool_name, str):
        return
    commit_asset_state_from_record(
        state_store=asset_state_store,
        simulator_tool=simulator_tool,  # type: ignore[arg-type]
        persona=persona,
        forecast=tick.forecast,
        record=ToolCallRecord(name=tool_name, arguments=arguments, ok=True, result=simulator_result),
    )


def _supervisor_specialist_reports(
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
) -> list[dict[str, object]]:
    reports = []
    for persona, role, decision, tool_calls, verdict, reasons in inputs:
        accepted = [
            record.arguments
            for record in tool_calls
            if record.name.startswith("simulate") and record.result.get("accepted") is True
        ][:3]
        reports.append(
            {
                "agent_id": persona.agent_id,
                "archetype": persona.archetype.value,
                "agent_role": role,
                "risk_attitude": persona.risk_attitude.value,
                "forecaster_id": persona.forecaster_id,
                "reported_action": decision.action,
                "watch_label": decision.watch_label,
                "risk_label": decision.risk_label,
                "confidence": decision.confidence,
                "verifier_accepted": verdict,
                "reason_codes": reasons,
                "accepted_candidate_count": len(accepted),
                "accepted_candidates": accepted,
                "rationale": decision.rationale[:360],
            }
        )
    return reports


def _chair_decision(
    *,
    strategy: str,
    tick: TickContext,
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
) -> tuple[LLMBidDecision, ToolCallRecord, bool | None, list[str]]:
    accepted = []
    watch_votes = 0
    must_watch_votes = 0
    high_risk_roles = []
    for persona, role, decision, tool_calls, verdict, _reasons in inputs:
        if decision.action in {"watch", "bid"} or decision.watch_label in {"watch", "must_watch"}:
            watch_votes += 1
        if decision.watch_label == "must_watch" or decision.action == "bid":
            must_watch_votes += 1
        if role in {"risk_officer", "society_chair", "physical_feasibility_scout"} and decision.risk_label == "high":
            high_risk_roles.append({"agent_id": persona.agent_id, "role": role, "side": decision.side})
        if decision.action != "bid" or verdict is not True:
            continue
        match = _matching_accepted_simulation(_required_simulation_tool(persona), decision, tool_calls)
        if match is None:
            continue
        accepted.append(
            {
                "persona": persona,
                "role": role,
                "decision": decision,
                "simulation": match,
                "archetype": persona.archetype.value,
                "side": decision.side,
            }
        )

    side_archetypes: dict[str, set[str]] = {"up": set(), "down": set()}
    for item in accepted:
        side = str(item["side"])
        if side in side_archetypes:
            side_archetypes[side].add(str(item["archetype"]))
    side_counts = {side: len(archetypes) for side, archetypes in side_archetypes.items()}
    supported_sides = [side for side, count in side_counts.items() if count > 0]
    contested = len(supported_sides) > 1
    threshold = 2 if strategy in {"comm_society_chair", "comm_society_chair_2agree", "comm_society_chair_riskveto", "comm_society_chair_intel"} else 1
    candidate_side = max(side_counts, key=lambda side: side_counts[side])
    risk_veto = strategy in {"comm_society_chair_riskveto", "comm_society_chair_intel"} and any(
        entry["side"] not in {None, candidate_side} for entry in high_risk_roles
    )
    intel_conservative = strategy == "comm_society_chair_intel" and (contested or side_counts[candidate_side] < threshold)
    selected = None
    if accepted and side_counts[candidate_side] >= threshold and not risk_veto and not intel_conservative:
        selected = _rank_chair_candidates([item for item in accepted if item["side"] == candidate_side])[0]

    final_action = "abstain"
    reasons: list[str] = []
    if selected is not None:
        selected_decision: LLMBidDecision = selected["decision"]  # type: ignore[assignment]
        sim: ToolCallRecord = selected["simulation"]  # type: ignore[assignment]
        decision = LLMBidDecision(
            action="bid",
            side=selected_decision.side,
            quantity_mwh=selected_decision.quantity_mwh,
            limit_price_eur_mwh=selected_decision.limit_price_eur_mwh,
            rationale=_chair_rationale(
                final_action="bid",
                side=selected_decision.side,
                side_counts=side_counts,
                contested=contested,
                risk_veto=risk_veto,
                selected=selected,
            ),
            confidence=min(0.85, max(0.55, selected_decision.confidence)),
            watch_label="must_watch",
            risk_label="medium" if contested else selected_decision.risk_label,
            uncertainty_label=selected_decision.uncertainty_label,
            opportunity_label="actionable",
            watch_reasons=_merge_watch_reasons(selected_decision.watch_reasons, ["accepted_bid_available"] + (["cross_agent_disagreement"] if contested else [])),
        )
        final_action = "bid"
        verifier_accepted: bool | None = True
        reasons = []
        selected_payload = {"agent_id": selected["persona"].agent_id, "archetype": selected["archetype"], "candidate": sim.arguments, "simulator_result": sim.result}
    elif accepted or watch_votes > 0 or must_watch_votes > 0:
        watch_label = "must_watch" if accepted or must_watch_votes > 0 else "watch"
        final_action = watch_label
        downgrade_reasons = []
        if accepted and side_counts[candidate_side] < threshold:
            downgrade_reasons.append("insufficient_independent_side_consensus")
        if contested:
            downgrade_reasons.append("accepted_candidate_side_conflict")
        if risk_veto:
            downgrade_reasons.append("risk_role_veto")
        if not downgrade_reasons and not accepted:
            downgrade_reasons.append("watch_evidence_without_accepted_candidate")
        decision = LLMBidDecision(
            action="watch",
            rationale=_chair_rationale(
                final_action=watch_label,
                side=candidate_side if accepted else None,
                side_counts=side_counts,
                contested=contested,
                risk_veto=risk_veto,
                selected=None,
                downgrade_reasons=downgrade_reasons,
            ),
            confidence=0.65 if watch_label == "must_watch" else 0.45,
            watch_label=watch_label,
            risk_label="high" if risk_veto or contested else "medium",
            uncertainty_label="high" if contested else "medium",
            opportunity_label="actionable" if accepted else "weak",
            watch_reasons=_merge_watch_reasons(
                [],
                ["accepted_bid_available" if accepted else "", "cross_agent_disagreement" if contested else "", "activation_risk"],
            ),
        )
        verifier_accepted = None
        reasons = downgrade_reasons
        selected_payload = None
    else:
        decision = LLMBidDecision(
            action="abstain",
            rationale="chair found no accepted candidates and no watch evidence from the society",
            confidence=0.3,
        )
        verifier_accepted = None
        selected_payload = None

    record = ToolCallRecord(
        name="society_chair_consensus",
        arguments={"strategy": strategy},
        ok=True,
        result={
            "ok": True,
            "authority": "deterministic_consensus",
            "final_action": final_action,
            "side_counts": side_counts,
            "accepted_candidate_count": len(accepted),
            "watch_vote_count": watch_votes,
            "must_watch_vote_count": must_watch_votes,
            "contested": contested,
            "risk_veto": risk_veto,
            "high_risk_roles": high_risk_roles,
            "selected": selected_payload,
            "policy": "chair can only select exact accepted simulator/proxy candidates or downgrade to watch",
        },
        provenance="runner_diagnostic",
    )
    return decision, record, verifier_accepted, reasons


def _rank_chair_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    def score(item: dict[str, object]) -> tuple[float, float, float]:
        decision: LLMBidDecision = item["decision"]  # type: ignore[assignment]
        sim: ToolCallRecord = item["simulation"]  # type: ignore[assignment]
        expected = _float_result(sim.result, "rough_expected_profit_eur")
        worst = _float_result(sim.result, "worst_case_profit_eur")
        return (expected, worst, float(decision.confidence))

    return sorted(candidates, key=score, reverse=True)


def _chair_rationale(
    *,
    final_action: str,
    side: str | None,
    side_counts: dict[str, int],
    contested: bool,
    risk_veto: bool,
    selected: dict[str, object] | None,
    downgrade_reasons: list[str] | None = None,
) -> str:
    evidence = f"side consensus up={side_counts.get('up', 0)}, down={side_counts.get('down', 0)}"
    flags = []
    if contested:
        flags.append("side disagreement")
    if risk_veto:
        flags.append("risk veto")
    if selected is not None:
        persona: Persona = selected["persona"]  # type: ignore[assignment]
        return (
            f"chair final {final_action}: selected exact accepted {side} candidate from {persona.agent_id} "
            f"({persona.archetype.value}); {evidence}; "
            "forecast/context/simulator evidence supported bidding"
            + (f"; flags: {', '.join(flags)}" if flags else "")
        )[:1000]
    reasons = ", ".join(downgrade_reasons or ["no accepted candidate"])
    return (
        f"chair final {final_action}: downgraded to watch because {reasons}; {evidence}; "
        "preserving important-hour explanation and uncertainty signal"
        + (f"; flags: {', '.join(flags)}" if flags else "")
    )[:1000]


def _initial_bid_budget_state(budget: int) -> dict[str, object]:
    return {
        "budget_per_agent": int(budget),
        "bids_submitted": 0,
        "bids_remaining": int(budget),
        "last_action": None,
        "last_verifier_accepted": None,
        "last_accepted_bid": None,
        "recent_actions": [],
    }


def _bid_budget_prompt_context(
    state: dict[str, object],
    *,
    enabled: bool,
    history_ticks: int,
) -> dict[str, object] | None:
    if not enabled:
        return None
    recent = state.get("recent_actions") if isinstance(state.get("recent_actions"), list) else []
    return {
        "authority": "live_non_leaking_bid_budget",
        "policy": (
            "Bids are scarce. Spend them only on high-confidence, simulator-accepted candidates. "
            "This context contains only prior in-run actions and verifier/simulator status, not evaluator truth."
        ),
        "budget_per_agent": state.get("budget_per_agent", 0),
        "bids_submitted": state.get("bids_submitted", 0),
        "bids_remaining": state.get("bids_remaining", 0),
        "last_action": state.get("last_action"),
        "last_verifier_accepted": state.get("last_verifier_accepted"),
        "last_accepted_bid": state.get("last_accepted_bid"),
        "recent_actions": recent[-history_ticks:] if history_ticks > 0 else [],
    }


def _with_bid_budget_context(
    context: dict[str, object] | None,
    budget_context: dict[str, object] | None,
) -> dict[str, object] | None:
    if budget_context is None:
        return context
    merged = dict(context or {})
    merged["bid_budget_context"] = budget_context
    return merged


def _bid_budget_context_record(context: dict[str, object]) -> ToolCallRecord:
    return ToolCallRecord(
        name="bid_budget_context",
        arguments={},
        ok=True,
        result={"ok": True, "authority": "live_non_leaking", "context": context},
        provenance="runner_diagnostic",
    )


def _bid_budget_exhausted_record(context: dict[str, object]) -> ToolCallRecord:
    return ToolCallRecord(
        name="bid_budget_exhausted",
        arguments={},
        ok=True,
        result={"ok": True, "authority": "policy", "context": context},
        provenance="runner_diagnostic",
    )


def _enforce_bid_budget(decision: LLMBidDecision, context: dict[str, object]) -> tuple[LLMBidDecision, bool]:
    try:
        remaining = int(context.get("bids_remaining", 0) or 0)
    except (TypeError, ValueError):
        remaining = 0
    if decision.action != "bid" or remaining > 0:
        return decision, False
    return (
        LLMBidDecision(
            action="watch",
            rationale="bid budget exhausted; policy downgraded proposed bid to watch",
            confidence=min(decision.confidence, 0.45),
            watch_label="watch",
            risk_label=decision.risk_label,
            uncertainty_label=decision.uncertainty_label,
            opportunity_label=decision.opportunity_label,
            watch_reasons=_merge_watch_reasons(decision.watch_reasons, ["accepted_bid_available"]),
            priority_label=decision.priority_label,
            priority_score=decision.priority_score,
            operator_action=decision.operator_action,
            priority_reason=decision.priority_reason,
        ),
        True,
    )


def _update_bid_budget_state(
    state: dict[str, object],
    *,
    step: int,
    decision: LLMBidDecision,
    verifier_accepted: bool | None,
    verifier_reason_codes: list[str],
    history_ticks: int,
) -> None:
    accepted_bid = decision.action == "bid" and verifier_accepted is True
    if accepted_bid:
        state["bids_submitted"] = int(state.get("bids_submitted", 0) or 0) + 1
        state["bids_remaining"] = max(0, int(state.get("budget_per_agent", 0) or 0) - int(state["bids_submitted"]))
        state["last_accepted_bid"] = {
            "step": step,
            "side": decision.side,
            "quantity_mwh": decision.quantity_mwh,
            "limit_price_eur_mwh": decision.limit_price_eur_mwh,
        }
    state["last_action"] = decision.action
    state["last_verifier_accepted"] = verifier_accepted
    recent = state.get("recent_actions")
    if not isinstance(recent, list):
        recent = []
        state["recent_actions"] = recent
    recent.append(
        {
            "step": step,
            "action": decision.action,
            "side": decision.side,
            "quantity_mwh": decision.quantity_mwh,
            "limit_price_eur_mwh": decision.limit_price_eur_mwh,
            "verifier_accepted": verifier_accepted,
            "verifier_reason_codes": verifier_reason_codes[:4],
        }
    )
    if history_ticks >= 0:
        del recent[: max(0, len(recent) - history_ticks)]


# --- Remaining functions that need to live here ---

def _initial_supervisor_quota_state(soft_quota_per_24_ticks: int, ticks: int) -> dict[str, object]:
    target = int(round((soft_quota_per_24_ticks / 24.0) * max(ticks, 1)))
    if soft_quota_per_24_ticks > 0:
        target = max(1, target)
    return {
        "soft_quota_per_24_ticks": int(soft_quota_per_24_ticks),
        "run_target_bids": max(0, target),
        "total_ticks": int(ticks),
        "executed_bids": 0,
        "seen_ticks": 0,
        "last_action": None,
    }


def _empty_supervisor_totals() -> dict[str, object]:
    return {
        "trace_count": 0,
        "available_candidate_count": 0,
        "selected_bid_count": 0,
        "watch_count": 0,
        "abstain_count": 0,
        "mutation_or_unbacked_count": 0,
        "quota_status_counts": {},
    }


def _accumulate_supervisor_totals(totals: dict[str, object], record: SocietyTraceRecord) -> None:
    totals["trace_count"] = int(totals["trace_count"]) + 1
    decision = record.decision
    if decision.action == "bid":
        totals["selected_bid_count"] = int(totals["selected_bid_count"]) + 1
    elif decision.action == "watch":
        totals["watch_count"] = int(totals["watch_count"]) + 1
    elif decision.action == "abstain":
        totals["abstain_count"] = int(totals["abstain_count"]) + 1
    for call in record.tool_calls:
        if call.name == "central_supervisor_context":
            totals["available_candidate_count"] = int(totals["available_candidate_count"]) + int(call.result.get("accepted_candidate_count") or 0)
            status = str(call.result.get("quota_status") or "unknown")
            counts = totals["quota_status_counts"]  # type: ignore[assignment]
            counts[status] = int(counts.get(status, 0)) + 1
        if call.name == "execution_gateway_validation" and call.result.get("gateway_outcome") in {"mutation_or_unbacked", "too_many_orders"}:
            totals["mutation_or_unbacked_count"] = int(totals["mutation_or_unbacked_count"]) + 1


def _finalize_supervisor_totals(totals: dict[str, object], quota_state: dict[str, object]) -> dict[str, object]:
    return {
        **totals,
        "quota_state": quota_state,
    }


async def _central_supervisor_trace_record(
    *,
    run_id: str,
    step: int,
    tick: TickContext,
    zone: str,
    llm: OpenAICompatibleLLMClient | None,
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
    quota_state: dict[str, object],
    soft_quota_per_24_ticks: int,
    max_orders_per_tick: int,
    memory_context: dict[str, object] | None,
) -> SocietyTraceRecord:
    candidates = _supervisor_candidate_menu(inputs, tick)
    quota_state["seen_ticks"] = int(quota_state.get("seen_ticks") or 0) + 1
    quota_status = _supervisor_quota_status(quota_state)
    context = {
        "authority": "derived_from_specialist_reports",
        "policy": "Specialists recommend; only the supervisor plus execution gateway can create the market action.",
        "soft_quota_per_24_ticks": soft_quota_per_24_ticks,
        "max_orders_per_tick": max_orders_per_tick,
        "quota_state": dict(quota_state),
        "quota_status": quota_status,
        "accepted_candidates": candidates[:12],
        "specialist_reports": _supervisor_specialist_reports(inputs),
        "memory_context": memory_context or {},
    }
    records = [
        ToolCallRecord(
            name="central_supervisor_context",
            arguments={"step": step},
            ok=True,
            result={
                "ok": True,
                "quota_status": quota_status,
                "accepted_candidate_count": len(candidates),
                "context": context,
            },
            provenance="runner_diagnostic",
        )
    ]
    proposed = await _central_supervisor_decision(llm=llm, context=context, candidates=candidates)
    records.append(_critic_action_record(proposed.model_dump(mode="json")))
    final, gateway = _execution_gateway_validate(proposed, candidates, max_orders_per_tick=max_orders_per_tick)
    records.append(gateway)
    if final.action == "bid":
        quota_state["executed_bids"] = int(quota_state.get("executed_bids") or 0) + 1
    quota_state["last_action"] = final.action
    _append_selected_candidate_diagnostic(records, final)
    lower, upper = tick.forecast.interval_for_side(final.side or "up")
    return SocietyTraceRecord(
        run_id=run_id,
        step=step,
        timestamp=tick.timestamp,
        observed_at=tick.timestamp,
        agent_id="supervisor-000",
        zone=zone,
        archetype="central_supervisor",
        agent_role="central_supervisor",
        llm_id="L5",
        forecaster_id="supervisor",
        forecast_backend="supervisor_synthesis",
        decision=final,
        verifier_mode="execution_gateway",
        verifier_accepted=True if final.action == "bid" else None,
        verifier_reason_codes=[] if final.action == "bid" else [str(gateway.result.get("gateway_outcome") or "non_bid")],
        market_price_eur_mwh=tick.market_price_eur_mwh,
        forecast_interval_eur_mwh=(lower, upper),
        rationale=final.rationale,
        unavailable_reason=tick.unavailable_reason,
        tool_calls=records,
        **_trace_tool_counter_fields(records),
    )


def _chair_trace_record(
    *,
    run_id: str,
    step: int,
    tick: TickContext,
    zone: str,
    strategy: str,
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
) -> SocietyTraceRecord:
    decision, consensus_record, verifier_accepted, reasons = _chair_decision(strategy=strategy, tick=tick, inputs=inputs)
    interval = tick.forecast.interval_for_side(decision.side or "up")
    tool_calls = [consensus_record]
    if any(persona.archetype.value == "decision_auditor" for persona, *_ in inputs):
        tool_calls.append(_decision_audit_record(strategy=strategy, inputs=inputs, consensus=consensus_record))
    return SocietyTraceRecord(
        run_id=run_id,
        step=step,
        timestamp=tick.timestamp,
        observed_at=tick.timestamp,
        agent_id="society-chair",
        zone=zone,
        archetype="society_chair",
        agent_role="society_chair",
        llm_id="deterministic_consensus",
        forecaster_id="society_digest",
        forecast_backend="society_digest",
        decision=decision,
        verifier_mode="consensus",
        verifier_accepted=verifier_accepted,
        verifier_reason_codes=reasons,
        market_price_eur_mwh=tick.market_price_eur_mwh,
        forecast_interval_eur_mwh=interval,
        rationale=decision.rationale,
        unavailable_reason=tick.unavailable_reason,
        tool_calls=tool_calls,
        **_trace_tool_counter_fields(tool_calls),
    )


def _decision_audit_record(
    *,
    strategy: str,
    inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]],
    consensus: ToolCallRecord,
) -> ToolCallRecord:
    action_counts: dict[str, int] = {}
    info_summaries = []
    action_summaries = []
    for persona, role, decision, tool_calls, verdict, reasons in inputs:
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        summary = {
            "agent_id": persona.agent_id,
            "archetype": persona.archetype.value,
            "role": role,
            "action": decision.action,
            "verifier_accepted": verdict,
            "reason_codes": reasons,
            "tool_count": len(tool_calls),
        }
        if _is_info_archetype(persona.archetype.value):
            info_summaries.append(summary)
        else:
            action_summaries.append(summary)
    return ToolCallRecord(
        name="decision_audit",
        arguments={"strategy": strategy},
        ok=True,
        result={
            "ok": True,
            "authority": "derived_from_society_trace",
            "action_counts": action_counts,
            "action_agents": action_summaries,
            "info_agents": info_summaries,
            "chair_final_action": consensus.result.get("final_action"),
            "chair_selected": consensus.result.get("selected"),
        },
        provenance="runner_diagnostic",
    )


def _specialist_recommendation_record(decision: LLMBidDecision) -> ToolCallRecord:
    return ToolCallRecord(
        name="specialist_recommendation",
        arguments=decision.model_dump(mode="json"),
        ok=True,
        result={
            "ok": True,
            "authority": "recommendation_only",
            "execution_policy": "central_supervisor_gateway_only",
            "recommended_action": decision.model_dump(mode="json"),
        },
        provenance="runner_diagnostic",
    )


def _agent_role(profile: str, idx: int) -> str:
    if profile == "info_specialists_v1":
        roles = [
            "grid_constraint_analyst",
            "outage_impact_scorer",
            "limit_price_specialist",
            "candidate_sizing_specialist",
            "uncertainty_auditor",
            "decision_auditor",
        ]
        return roles[idx % len(roles)]
    if profile == "action_core_8_plus_info_specialists":
        roles = [
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
            "grid_constraint_analyst",
            "outage_impact_scorer",
            "limit_price_specialist",
            "candidate_sizing_specialist",
            "uncertainty_auditor",
            "decision_auditor",
        ]
        return roles[idx % len(roles)]
    if profile == "jao_grid_v1":
        roles = [
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
            "grid_constraint_analyst",
            "outage_impact_scorer",
            "uncertainty_auditor",
            "decision_auditor",
        ]
        return roles[idx % len(roles)]
    if profile in {"mixed_expert_18_sideaware", "mixed_expert_20_sideaware"}:
        roles = [
            "risk_officer",
            "activation_scout",
            "down_opportunity_scout",
            "physical_feasibility_scout",
            "opportunity_scout",
            "physical_feasibility_scout",
            "activation_scout",
            "volatility_scout",
            "risk_officer",
            "historian",
            "grid_constraint_analyst",
            "outage_impact_scorer",
            "limit_price_specialist",
            "candidate_sizing_specialist",
            "uncertainty_auditor",
            "decision_auditor",
            "risk_officer",
            "society_chair",
            "counterfactual_opportunity_scout",
            "counterfactual_risk_monitor",
        ]
        return roles[idx % len(roles)]
    if profile == "diverse_expert_action":
        roles = [
            "activation_scout",
            "volatility_scout",
            "physical_feasibility_scout",
            "opportunity_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "historian",
            "opportunity_scout",
        ]
        return roles[idx % len(roles)]
    if profile == "action_core_8":
        roles = [
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
        ]
        return roles[idx % len(roles)]
    if profile in {"action_core_9_chair", "action_core_10_safety"}:
        roles = [
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
            "society_chair",
            "risk_officer",
        ]
        return roles[idx % len(roles)]
    if profile == "action_core_8_aggressive":
        roles = [
            "opportunity_scout",
            "activation_scout",
            "opportunity_scout",
            "activation_scout",
            "volatility_scout",
            "opportunity_scout",
            "physical_feasibility_scout",
            "physical_feasibility_scout",
        ]
        return roles[idx % len(roles)]
    if profile in {"action_core_8_safety", "action_core_8_toolsplit"}:
        roles = [
            "risk_officer",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
        ]
        return roles[idx % len(roles)]
    if profile in {"balanced_intelligence", "crowd_intelligence"}:
        roles = [
            "physical_feasibility_scout",
            "physical_feasibility_scout",
            "activation_scout",
            "physical_feasibility_scout",
            "volatility_scout",
            "risk_officer",
            "opportunity_scout",
            "historian",
            "explanation_editor",
            "volatility_scout",
            "activation_scout",
            "society_chair",
        ]
        return roles[idx % len(roles)]
    if profile == "market_expert_panel":
        roles = ["market_mechanics_expert", "imbalance_analytics_expert", "trading_risk_monitor"]
        return roles[idx % len(roles)]
    if profile == "p2h_specialist_v2":
        roles = ["p2h_simulator_specialist", "imbalance_analytics_expert", "trading_risk_monitor"]
        return roles[idx % len(roles)]
    if profile == "ev_specialist_v2":
        roles = ["ev_virtual_battery_specialist", "imbalance_analytics_expert", "trading_risk_monitor"]
        return roles[idx % len(roles)]
    if profile == "p2h_info_then_action_v2":
        roles = ["market_mechanics_expert", "imbalance_analytics_expert", "trading_risk_monitor", "p2h_simulator_specialist"]
        return roles[idx % len(roles)]
    if profile == "ev_info_then_action_v2":
        roles = ["market_mechanics_expert", "imbalance_analytics_expert", "trading_risk_monitor", "ev_virtual_battery_specialist"]
        return roles[idx % len(roles)]
    if profile == "market_experts_plus_action_core_6":
        roles = [
            "market_mechanics_expert",
            "imbalance_analytics_expert",
            "trading_risk_monitor",
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
        ]
        return roles[idx % len(roles)]
    if profile == "action_core_8_plus_market_expert":
        roles = [
            "opportunity_scout",
            "activation_scout",
            "risk_officer",
            "physical_feasibility_scout",
            "volatility_scout",
            "historian",
            "risk_officer",
            "physical_feasibility_scout",
            "market_mechanics_expert",
            "imbalance_analytics_expert",
            "trading_risk_monitor",
        ]
        return roles[idx % len(roles)]
    return "action_agent"
