from __future__ import annotations

import asyncio
import json
from datetime import timedelta

from heimdall_contracts import Persona
from heimdall_ai_society.llm_client import LLMClientError, OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tool_policy import policy_for_persona
from heimdall_ai_society.tools import (
    AgentToolExecutor,
    decision_from_tool_calls,
    openai_tool_specs,
    retrieve_knowledge_tool_spec,
)
from packages.simulator import ScenarioAssetStateStore

from heimdall_ai_society._trace_helpers import (
    _append_selected_candidate_diagnostic,
    _float_result,
    _has_seeded_accepted_candidate,
    _llm_failure,
    _matching_accepted_simulation,
    _record_controls_acceptance,
    _with_provenance,
    _accepted_candidate_count,
)
from heimdall_ai_society._prompts import (
    _compact_tool_records_for_prompt,
    _final_action_instruction,
    _frontier_feedback_prompt,
    _prompt,
    _required_simulation_tool,
)
from heimdall_ai_society._candidates import (
    _rank_seeded_candidates,
    _seed_candidate_tools,
    _seed_context_tools,
    _seed_specialist_tools,
)
from heimdall_ai_society._context import (
    _merge_watch_reasons,
    _seed_retrieval,
)


def _executor_for_persona(
    *,
    persona: Persona,
    tick: TickContext,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    tool_cache: dict[tuple[str, str], ToolCallRecord],
) -> AgentToolExecutor:
    observed_at = tick.timestamp - timedelta(minutes=persona.info_latency_min)
    scoped_data_tools = data_tools
    if scoped_data_tools is not None and hasattr(scoped_data_tools, "with_observed_at"):
        scoped_data_tools = scoped_data_tools.with_observed_at(observed_at)  # type: ignore[assignment,union-attr]
    return AgentToolExecutor(
        persona=persona,
        forecast=tick.forecast,
        data_tools=scoped_data_tools,
        simulator_tool=simulator_tool,  # type: ignore[arg-type]
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache=tool_cache,
    )


async def _execute_agent_tool(
    executor: AgentToolExecutor,
    name: str,
    arguments: dict[str, object],
    simulator_semaphore: asyncio.Semaphore,
    *,
    provenance: str = "unknown",
) -> ToolCallRecord:
    if name in {
        "simulate_bid",
        "simulate_ev_bid",
        "simulate_wind_bid",
        "simulate_generator_bid",
        "simulate_retailer_bid",
        "simulate_renewables_bid",
    }:
        async with simulator_semaphore:
            return _with_provenance(executor.execute(name, arguments), provenance)
    return _with_provenance(executor.execute(name, arguments), provenance)


async def _run_tool_round(
    *,
    llm: OpenAICompatibleLLMClient,
    executor: AgentToolExecutor,
    messages: list[dict],
    records: list[ToolCallRecord],
    tools: list[dict[str, object]],
    simulator_semaphore: asyncio.Semaphore,
    provenance: str,
    tool_choice: dict[str, object] | None = None,
) -> bool:
    message = await llm.tool_round(messages, tools, tool_choice=tool_choice)
    messages.append(message)
    calls = message.get("tool_calls") or []
    if not calls:
        return False
    allowed = _tool_names(tools)
    for call in calls:
        function = call.get("function", {})
        name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if name not in allowed:
            from heimdall_ai_society._deliberation import _deliberation_diagnostic
            record = _deliberation_diagnostic(
                "phase_tool_blocked",
                {"tool": name},
                {"ok": False, "error_code": "tool_hidden_in_phase"},
                provenance="runner_diagnostic",
            )
        else:
            record = await _execute_agent_tool(executor, name, arguments, simulator_semaphore, provenance=provenance)
        records.append(record)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "name": name,
                "content": json.dumps(record.result if record.ok else {"ok": False, "error": record.error}),
            }
        )
    return True


def _phase_tool_specs(safety_toolset: str, *, phase: str) -> list[dict[str, object]]:
    tools = _openai_tool_specs_for_safety_toolset(safety_toolset)
    if phase == "final":
        keep = {"propose_action"}
    elif phase == "note":
        keep = {"propose_deliberation_note"}
    elif phase == "action_probe":
        hidden = {"propose_action", "propose_bid", "propose_deliberation_note"}
        keep = _tool_names(tools) - hidden
    elif phase == "inquiry":
        hidden = {"propose_action", "propose_bid", "propose_deliberation_note", "propose_peer_response"}
        keep = _tool_names(tools) - hidden
    else:
        keep = _tool_names(tools)
    return [tool for tool in tools if _tool_name(tool) in keep]


def _tool_name(tool: dict[str, object]) -> str:
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    return str(function.get("name", "")) if isinstance(function, dict) else ""


def _tool_names(tools: list[dict[str, object]]) -> set[str]:
    return {_tool_name(tool) for tool in tools}


def _openai_tool_specs_for_safety_toolset(safety_toolset: str) -> list[dict[str, object]]:
    tools = openai_tool_specs()
    if safety_toolset != "context_only":
        return tools
    hidden = {
        "get_bid_feasibility",
        "get_ev_bid_feasibility",
        "get_wind_bid_feasibility",
        "get_generator_bid_feasibility",
        "get_retailer_bid_feasibility",
        "get_renewables_bid_feasibility",
        "simulate_bid",
        "simulate_ev_bid",
        "simulate_wind_bid",
        "simulate_generator_bid",
        "simulate_retailer_bid",
        "simulate_renewables_bid",
        "get_limit_price_guidance",
        "get_candidate_rejection_summary",
        "get_candidate_sizing_guidance",
        "get_decision_trace_summary",
    }
    filtered = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if name not in hidden:
            filtered.append(tool)
    return filtered


def _shadow_required_simulation(
    *,
    persona: Persona,
    decision: LLMBidDecision,
    tick: TickContext,
    data_tools: object | None,
    simulator_tool: object,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
) -> ToolCallRecord | None:
    if decision.action != "bid":
        return None
    if decision.side is None or decision.quantity_mwh is None or decision.limit_price_eur_mwh is None:
        return None
    observed_at = tick.timestamp - timedelta(minutes=persona.info_latency_min)
    if data_tools is not None and hasattr(data_tools, "with_observed_at"):
        data_tools = data_tools.with_observed_at(observed_at)  # type: ignore[assignment,union-attr]
    executor = AgentToolExecutor(
        persona=persona,
        forecast=tick.forecast,
        data_tools=data_tools,
        simulator_tool=simulator_tool,  # type: ignore[arg-type]
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache={},
    )
    required = _required_simulation_tool(persona)
    arguments = {
        "side": decision.side,
        "quantity_mwh": decision.quantity_mwh,
        "limit_price_eur_mwh": decision.limit_price_eur_mwh,
    }
    shadow = executor.execute(required, arguments)
    result = shadow.result if isinstance(shadow.result, dict) else {}
    return ToolCallRecord(
        name="shadow_required_simulation",
        arguments={"required_tool": required, **arguments},
        ok=shadow.ok,
        result={
            "ok": shadow.ok,
            "authority": "shadow_post_decision",
            "required_tool": required,
            "shadow_accepted": result.get("accepted"),
            "shadow_reason_codes": result.get("reason_codes", []),
            "shadow_worst_case_profit_eur": result.get("worst_case_profit_eur") or result.get("rough_worst_case_profit_eur"),
            "shadow_expected_profit_eur": result.get("expected_profit_eur") or result.get("rough_expected_profit_eur"),
            "shadow_controls_acceptance": _record_controls_acceptance(shadow),
            "shadow_result": result,
        },
        error=shadow.error,
        provenance="runner_diagnostic",
    )


def _enforce_action_policy(
    persona: Persona,
    decision: LLMBidDecision,
    tool_calls: list[ToolCallRecord],
    *,
    agent_role: str = "action_agent",
    ablation_strategy: str = "baseline",
    tick: TickContext | None = None,
    final_bid_guard: str = "simulator_exact_match",
) -> LLMBidDecision:
    if decision.action == "abstain" and any(
        record.name.startswith("simulate") and record.result.get("reason_codes")
        for record in tool_calls
    ):
        return decision.model_copy(
            update={
                "action": "watch",
                "rationale": "policy converted abstain to watch because simulator rejection clusters are useful market intelligence",
                "confidence": max(decision.confidence, 0.4),
                "watch_label": "watch",
                "risk_label": decision.risk_label if decision.risk_label != "low" else "medium",
                "opportunity_label": decision.opportunity_label if decision.opportunity_label != "none" else "weak",
                "watch_reasons": _merge_watch_reasons(decision.watch_reasons, ["verifier_rejection_cluster"]),
            }
        )
    if decision.action != "bid":
        return decision
    if agent_role in {"society_chair", "explanation_editor"}:
        return decision.model_copy(
            update={
                "action": "watch",
                "side": None,
                "quantity_mwh": None,
                "limit_price_eur_mwh": None,
                "rationale": f"policy converted {agent_role} bid to watch; synthesis roles do not bid",
                "confidence": min(decision.confidence, 0.5),
                "watch_label": "must_watch" if decision.watch_label == "must_watch" else "watch",
                "watch_reasons": _merge_watch_reasons(decision.watch_reasons, ["accepted_bid_available"]),
            }
        )
    policy = policy_for_persona(persona)
    if not policy.can_submit_bid:
        return LLMBidDecision(
            action="watch",
            rationale=f"policy converted unsupported {persona.archetype.value} bid to watch",
            confidence=min(decision.confidence, 0.4),
            watch_label="watch",
            risk_label=decision.risk_label,
            uncertainty_label=decision.uncertainty_label,
            opportunity_label=decision.opportunity_label,
            watch_reasons=_merge_watch_reasons(decision.watch_reasons, ["accepted_bid_available"]),
        )
    if final_bid_guard == "schema_only_shadow":
        return decision
    if not policy.bid_requires_authoritative_simulation:
        return decision
    required = _required_simulation_tool(persona)
    accepted_simulation = _matching_accepted_simulation(required, decision, tool_calls)
    if accepted_simulation is not None:
        p2h_filtered = _p2h_v2_price_regime_filter_bid_decision(persona, decision, tick=tick)
        if p2h_filtered is not None:
            return p2h_filtered
        ev_filtered = _ev_v2_caution_filter_bid_decision(persona, decision, accepted_simulation, tick=tick)
        if ev_filtered is not None:
            return ev_filtered
        filtered = _risk_filter_bid_decision(
            decision,
            accepted_simulation,
            tick=tick,
            ablation_strategy=ablation_strategy,
        )
        if filtered is not None:
            return filtered
        return decision
    if tool_calls:
        return LLMBidDecision(
            action="watch",
            rationale=f"policy converted bid to watch because {required} did not accept the candidate",
            confidence=min(decision.confidence, 0.45),
            watch_label="watch",
            risk_label=decision.risk_label,
            uncertainty_label=decision.uncertainty_label,
            opportunity_label=decision.opportunity_label,
            watch_reasons=_merge_watch_reasons(decision.watch_reasons, ["verifier_rejection_cluster"]),
        )
    return decision


def _needs_authoritative_simulation(
    persona: Persona,
    decision: LLMBidDecision,
    tool_calls: list[ToolCallRecord],
) -> bool:
    if decision.action != "bid":
        return False
    policy = policy_for_persona(persona)
    if not policy.bid_requires_authoritative_simulation:
        return False
    required = _required_simulation_tool(persona)
    return _matching_accepted_simulation(required, decision, tool_calls) is None


def _downgrade_unsupported_bid(
    persona: Persona,
    decision: LLMBidDecision,
    tool_calls: list[ToolCallRecord],
) -> LLMBidDecision:
    from heimdall_ai_society._deterministic import _watch_score_from_records
    required = _required_simulation_tool(persona)
    attempted = any(record.name == required for record in tool_calls)
    watch_score = _watch_score_from_records(tool_calls)
    simulate_rejections = [
        reason
        for record in tool_calls
        if record.name == required
        for reason in record.result.get("reason_codes", [])
    ]
    if attempted or watch_score >= 0.35:
        return LLMBidDecision(
            action="watch",
            rationale=(
                "downgraded unsupported autonomous bid: no exact accepted controlling "
                f"{required} call matched the proposed bid"
            ),
            confidence=min(max(decision.confidence, 0.35), 0.65),
            watch_label="watch",
            risk_label="medium",
            uncertainty_label=decision.uncertainty_label,
            opportunity_label="weak",
            watch_reasons=_merge_watch_reasons(
                decision.watch_reasons,
                ["activation_risk" if watch_score >= 0.35 else "", "verifier_rejection_cluster" if simulate_rejections else ""],
            ),
        )
    return LLMBidDecision(
        action="abstain",
        rationale=(
            "downgraded unsupported autonomous bid: the agent did not obtain an accepted "
            f"{required} simulator result"
        ),
        confidence=0.2,
    )


def _repair_placeholder_retry_bid(persona: Persona, records: list[ToolCallRecord], *, tick: TickContext) -> LLMBidDecision | None:
    placeholder_bid = any(
        record.name in {"propose_action", "propose_bid"}
        and str(record.arguments.get("action")) == "bid"
        and (
            float(record.arguments.get("quantity_mwh") or 0.0) <= 0.0
            or float(record.arguments.get("limit_price_eur_mwh") or 0.0) == 0.0
        )
        for record in records
    )
    if not placeholder_bid:
        return None
    accepted = [
        record
        for record in records
        if record.name == _required_simulation_tool(persona)
        and record.ok
        and _record_controls_acceptance(record)
        and record.result.get("accepted") is True
    ]
    if not accepted:
        return None
    ranked = _rank_seeded_candidates(accepted, tick)
    best_args = ranked[0]["arguments"] if ranked else accepted[0].arguments
    return LLMBidDecision(
        action="bid",
        side=str(best_args["side"]),
        quantity_mwh=float(best_args["quantity_mwh"]),
        limit_price_eur_mwh=float(best_args["limit_price_eur_mwh"]),
        rationale="repaired retry placeholder bid to the best exact simulator-accepted candidate",
        confidence=0.72,
        watch_label="must_watch",
        risk_label="medium",
        uncertainty_label="medium",
        opportunity_label="actionable",
        watch_reasons=["accepted_bid_available"],
    )


async def _retry_final_action(
    *,
    llm: OpenAICompatibleLLMClient,
    executor: AgentToolExecutor,
    messages: list[dict],
    records: list[ToolCallRecord],
    persona: Persona,
    objective: str,
    tools: list[dict],
    tick: TickContext,
    simulator_semaphore: asyncio.Semaphore,
) -> tuple[LLMBidDecision, list[ToolCallRecord]] | None:
    accepted_candidates = [
        {"tool": record.name, "arguments": record.arguments, "result": record.result}
        for record in records
        if record.name == _required_simulation_tool(persona) and record.result.get("accepted") is True
    ]
    rejected_reasons: dict[str, int] = {}
    for record in records:
        if record.name != _required_simulation_tool(persona):
            continue
        for reason in record.result.get("reason_codes", []):
            rejected_reasons[str(reason)] = rejected_reasons.get(str(reason), 0) + 1
    retry_prompt = {
        "role": "user",
        "content": (
            "Retry once as a council reconsideration. Revise watch/risk/uncertainty labels freely. The final bid, if any, must exactly match one accepted "
            f"{_required_simulation_tool(persona)} candidate. If no accepted candidate exists, choose watch when this hour is important. "
            + json.dumps(
                {
                    "objective": objective,
                    "accepted_candidates": accepted_candidates,
                    "rejected_reason_counts": rejected_reasons,
                },
                sort_keys=True,
            )
        ),
    }
    retry_message = await llm.tool_round(
        [*messages, retry_prompt],
        tools,
        tool_choice={"type": "function", "function": {"name": "propose_action"}},
    )
    records.append(
        ToolCallRecord(
            name="retry_council_diagnostic",
            arguments={"accepted_candidate_count": len(accepted_candidates)},
            ok=True,
            result={"ok": True, "authority": "advisory", "rejected_reason_counts": rejected_reasons},
            provenance="retry",
        )
    )
    for call in retry_message.get("tool_calls") or []:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        records.append(
            await _execute_agent_tool(
                executor,
                function.get("name", ""),
                arguments,
                simulator_semaphore,
                provenance="retry",
            )
        )
    decision = decision_from_tool_calls(records)
    if decision is not None and not _needs_authoritative_simulation(persona, decision, records):
        _append_selected_candidate_diagnostic(records, decision)
        return decision, records
    repaired = _repair_placeholder_retry_bid(persona, records, tick=tick)
    if repaired is not None:
        _append_selected_candidate_diagnostic(records, repaired)
        return repaired, records
    return None


def _p2h_v2_price_regime_filter_bid_decision(
    persona: Persona,
    decision: LLMBidDecision,
    *,
    tick: TickContext | None,
) -> LLMBidDecision | None:
    if persona.archetype.value != "p2h" or tick is None:
        return None
    if decision.side == "up" and tick.market_price_eur_mwh < 70.0:
        return decision.model_copy(
            update={
                "action": "watch",
                "side": None,
                "quantity_mwh": None,
                "limit_price_eur_mwh": None,
                "rationale": (
                    "P2H V2 converted accepted up candidate to watch: current price regime is below the "
                    "up-activation promotion gate; simulator candidate remains useful watch evidence"
                ),
                "confidence": min(decision.confidence, 0.62),
                "watch_label": "must_watch" if decision.watch_label == "must_watch" else "watch",
                "risk_label": decision.risk_label if decision.risk_label != "low" else "medium",
                "opportunity_label": decision.opportunity_label,
                "watch_reasons": _merge_watch_reasons(
                    decision.watch_reasons,
                    ["accepted_bid_available", "activation_risk"],
                ),
            }
        )
    return None


def _ev_v2_caution_filter_bid_decision(
    persona: Persona,
    decision: LLMBidDecision,
    accepted_simulation: ToolCallRecord,
    *,
    tick: TickContext | None,
) -> LLMBidDecision | None:
    if persona.archetype.value != "ev":
        return None
    if _ev_low_price_down_probe_allowed(decision, accepted_simulation, tick):
        return None
    reasons: list[str] = []
    if decision.risk_label in {"medium", "high"}:
        reasons.append(f"{decision.risk_label}_risk")
    if decision.uncertainty_label in {"medium", "high"}:
        reasons.append(f"{decision.uncertainty_label}_uncertainty")
    if decision.confidence < 0.9:
        reasons.append("ev_confidence_below_promotion_gate")
    worst_case_profit = _float_result(accepted_simulation.result, "worst_case_profit_eur")
    if worst_case_profit < 50.0:
        reasons.append("ev_worst_case_edge_below_promotion_gate")
    if not reasons:
        return None
    return decision.model_copy(
        update={
            "action": "watch",
            "side": None,
            "quantity_mwh": None,
            "limit_price_eur_mwh": None,
            "rationale": (
                "EV V2 converted accepted simulator candidate to watch: "
                f"{', '.join(reasons)}; virtual-battery accepted candidates remain advisory until "
                "side evidence is stronger"
            ),
            "confidence": min(decision.confidence, 0.65),
            "watch_label": "must_watch" if decision.watch_label == "must_watch" else "watch",
            "risk_label": decision.risk_label if decision.risk_label != "low" else "medium",
            "opportunity_label": decision.opportunity_label,
            "watch_reasons": _merge_watch_reasons(
                decision.watch_reasons,
                ["accepted_bid_available", "forecast_uncertainty"],
            ),
        }
    )


def _ev_low_price_down_probe_allowed(
    decision: LLMBidDecision,
    accepted_simulation: ToolCallRecord,
    tick: TickContext | None,
) -> bool:
    if tick is None or decision.side != "down":
        return False
    if decision.quantity_mwh is None or float(decision.quantity_mwh) > 0.25 + 1e-9:
        return False
    if tick.market_price_eur_mwh > 55.0:
        return False
    worst_case_profit = _float_result(accepted_simulation.result, "worst_case_profit_eur")
    expected_profit = _float_result(accepted_simulation.result, "expected_profit_eur")
    if worst_case_profit < 2.0 or expected_profit < 10.0:
        return False
    return decision.confidence >= 0.7 and decision.opportunity_label == "actionable"


def _risk_filter_bid_decision(
    decision: LLMBidDecision,
    accepted_simulation: ToolCallRecord,
    *,
    tick: TickContext | None,
    ablation_strategy: str,
) -> LLMBidDecision | None:
    from heimdall_ai_society._prompts import _opportunity_hint
    if ablation_strategy != "comm_broadcast_digest_risk_filter":
        return None
    reasons: list[str] = []
    if decision.confidence < 0.62:
        reasons.append("low_confidence")
    if decision.risk_label == "high" and decision.opportunity_label != "actionable":
        reasons.append("high_risk_without_actionable_opportunity")
    if tick is not None:
        up_lower, up_upper = tick.forecast.interval_for_side("up")
        down_lower, down_upper = tick.forecast.interval_for_side("down")
        hint = _opportunity_hint(
            price=tick.market_price_eur_mwh,
            up_lower=up_lower,
            up_upper=up_upper,
            down_lower=down_lower,
            down_upper=down_upper,
        )
        hinted_side = hint.get("candidate_bid_side")
        if hinted_side in {"up", "down"} and decision.side != hinted_side:
            expected_profit = _float_result(accepted_simulation.result, "rough_expected_profit_eur")
            worst_case_profit = _float_result(accepted_simulation.result, "worst_case_profit_eur")
            if expected_profit < 250.0 or worst_case_profit < 0.0:
                reasons.append("side_disagrees_with_market_digest")
    if not reasons:
        return None
    return decision.model_copy(
        update={
            "action": "watch",
            "side": None,
            "quantity_mwh": None,
            "limit_price_eur_mwh": None,
            "rationale": f"risk filter converted accepted bid to watch: {', '.join(reasons)}; simulator result remains evidence only",
            "confidence": min(decision.confidence, 0.58),
            "watch_label": "must_watch" if decision.watch_label == "must_watch" else "watch",
            "risk_label": decision.risk_label if decision.risk_label != "low" else "medium",
            "opportunity_label": decision.opportunity_label,
            "watch_reasons": _merge_watch_reasons(decision.watch_reasons, ["accepted_bid_available", "cross_agent_disagreement"]),
        }
    )


def _is_info_archetype(archetype: str) -> bool:
    return archetype in {
        "market_mechanics_expert",
        "imbalance_analytics_expert",
        "trading_risk_monitor",
        "grid_constraint_analyst",
        "outage_impact_scorer",
        "limit_price_specialist",
        "candidate_sizing_specialist",
        "uncertainty_auditor",
        "decision_auditor",
    }


async def _decide_for_persona(
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient | None,
    *,
    agent_role: str = "action_agent",
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
    final_bid_guard: str = "simulator_exact_match",
    safety_toolset: str = "full",
    preprobe_mode: str = "full",
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
    chooser_mode: str = "llm",
    communication_context: dict[str, object] | None = None,
    memory_context: dict[str, object] | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    forecast_diversity_context: dict[str, object] | None = None,
    seed_outage_context: bool = False,
    rationale_directive: str = "",
    retriever: object | None = None,
    rag_top_k: int = 4,
    rag_max_chars: int = 700,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    from heimdall_ai_society._deterministic import (
        _deterministic_decision,
        _deterministic_best_accepted_decision,
        _deterministic_high_fill_accepted_decision,
        _deterministic_llm_critic_decision,
        _deterministic_watch_threshold_decision,
        _llm_fill_selector_decision,
    )
    if tick.unavailable_reason is not None:
        return (
            LLMBidDecision(
                action="abstain",
                rationale=f"market context unavailable: {tick.unavailable_reason}",
                confidence=0.0,
            ),
            [],
        )
    if chooser_mode == "deterministic_best_accepted":
        return _deterministic_best_accepted_decision(
            persona=persona,
            tick=tick,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            tool_cache=tool_cache,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
        )
    if chooser_mode == "deterministic_high_fill_accepted":
        return _deterministic_high_fill_accepted_decision(
            persona=persona,
            tick=tick,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            tool_cache=tool_cache,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
        )
    if chooser_mode == "deterministic_watch_threshold":
        return _deterministic_watch_threshold_decision(
            persona=persona,
            tick=tick,
            data_tools=data_tools,
            tool_cache=tool_cache,
        )
    if chooser_mode == "deterministic_llm_critic":
        return await _deterministic_llm_critic_decision(
            persona=persona,
            tick=tick,
            llm=llm,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            simulator_semaphore=simulator_semaphore,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
            communication_context=communication_context,
            memory_context=memory_context,
            tool_cache=tool_cache,
            forecast_diversity_context=forecast_diversity_context,
        )
    if chooser_mode == "llm_fill_selector":
        return await _llm_fill_selector_decision(
            persona=persona,
            tick=tick,
            llm=llm,
            objective=objective,
            ablation_strategy=ablation_strategy,
            data_tools=data_tools,
            simulator_tool=simulator_tool,
            asset_simulator_mode=asset_simulator_mode,
            asset_proxy_style=asset_proxy_style,
            asset_state_store=asset_state_store,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
            communication_context=communication_context,
            memory_context=memory_context,
            tool_cache=tool_cache,
            forecast_diversity_context=forecast_diversity_context,
        )
    if llm is None:
        return _deterministic_decision(persona, tick), []
    if tool_mode != "openai_tools":
        try:
            return await llm.decide(
                _prompt(
                    persona,
                    tick,
                    agent_role=agent_role,
                    objective=objective,
                    ablation_strategy=ablation_strategy,
                    safety_toolset=safety_toolset,
                    communication_context=communication_context,
                    memory_context=memory_context,
                )
            ), []
        except LLMClientError as exc:
            return _llm_failure(exc), []
    return await _decide_with_tools(
        persona,
        tick,
        llm,
        agent_role=agent_role,
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
        communication_context=communication_context,
        memory_context=memory_context,
        tool_cache=tool_cache,
        seed_outage_context=seed_outage_context,
        rationale_directive=rationale_directive,
        retriever=retriever,
        rag_top_k=rag_top_k,
        rag_max_chars=rag_max_chars,
    )


async def _decide_with_tools(
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient,
    *,
    agent_role: str = "action_agent",
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    max_tool_rounds: int,
    final_bid_guard: str = "simulator_exact_match",
    safety_toolset: str = "full",
    preprobe_mode: str = "full",
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
    communication_context: dict[str, object] | None = None,
    memory_context: dict[str, object] | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    seed_outage_context: bool = False,
    rationale_directive: str = "",
    retriever: object | None = None,
    rag_top_k: int = 4,
    rag_max_chars: int = 700,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    observed_at = tick.timestamp - timedelta(minutes=persona.info_latency_min)
    if data_tools is not None and hasattr(data_tools, "with_observed_at"):
        data_tools = data_tools.with_observed_at(observed_at)  # type: ignore[assignment,union-attr]
    executor = AgentToolExecutor(
        persona=persona,
        forecast=tick.forecast,
        data_tools=data_tools,
        simulator_tool=simulator_tool,  # type: ignore[arg-type]
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache=tool_cache,
        retriever=retriever,
        rag_top_k=rag_top_k,
        rag_max_chars=rag_max_chars,
    )
    messages: list[dict] = _prompt(
        persona,
        tick,
        agent_role=agent_role,
        objective=objective,
        ablation_strategy=ablation_strategy,
        safety_toolset=safety_toolset,
        communication_context=communication_context,
        memory_context=memory_context,
    )
    tools = _openai_tool_specs_for_safety_toolset(safety_toolset)
    if retriever is not None:
        tools = [*tools, retrieve_knowledge_tool_spec()]
    records: list[ToolCallRecord] = []
    if preprobe_mode in {"full", "context_only", "specialist_context"}:
        records.extend(_seed_context_tools(persona, executor, include_outages=seed_outage_context))
    if preprobe_mode == "full":
        records.extend(_seed_specialist_tools(persona, executor, safety_toolset=safety_toolset))
    elif preprobe_mode == "specialist_context" and _is_info_archetype(persona.archetype.value):
        records.extend(_seed_specialist_tools(persona, executor, safety_toolset=safety_toolset))
    if preprobe_mode == "full" and safety_toolset != "context_only":
        records.extend(_seed_candidate_tools(
            objective=objective,
            ablation_strategy=ablation_strategy,
            persona=persona,
            tick=tick,
            executor=executor,
            candidate_sizing_mode=candidate_sizing_mode,
            candidate_sizing_cap_fraction=candidate_sizing_cap_fraction,
            candidate_sizing_min_mwh=candidate_sizing_min_mwh,
            candidate_sizing_max_candidates=candidate_sizing_max_candidates,
        ))
    if retriever is not None:
        records.extend(_seed_retrieval(persona, tick, executor, top_k=rag_top_k))
    if records:
        executor._candidate_diagnostics = records
        if preprobe_mode == "context_only":
            tool_prompt = (
                "The experiment runner preloaded the current non-leaking context tools. "
                "No simulator, verifier, feasibility, candidate-menu, or ranker results are available before final action. "
                "Choose and call candidate, feasibility, and simulator tools yourself if you want to justify a bid. "
            )
        elif preprobe_mode == "specialist_context":
            tool_prompt = (
                "The experiment runner preloaded only context and specialist diagnostic tools. "
                "No action-agent candidate, feasibility, simulator, or ranker results are available before final action. "
                "Choose and call the relevant action tools yourself if you want to justify a bid. "
            )
        else:
            tool_prompt = (
                "The experiment runner pre-probed the current opportunity. "
                "Use these real tool results; do not claim simulator acceptance unless your archetype's simulator accepted=true. "
            )
        if retriever is not None:
            tool_prompt += (
                "A retrieve_knowledge result (leak-safe historical regime stats, prior-run lessons, and "
                "methodology — only data available at/before this tick) is included. Use it to choose the "
                "activation side, bid size, and whether to participate. Call retrieve_knowledge again with a "
                "more specific query if you need more evidence before proposing. "
            )
        messages.append(
            {
                "role": "user",
                "content": tool_prompt + json.dumps(_compact_tool_records_for_prompt(records), sort_keys=True),
            }
        )
    if communication_context is not None:
        records.append(
            ToolCallRecord(
                name="society_communication_context",
                arguments={"strategy": ablation_strategy},
                ok=True,
                result={"ok": True, "authority": "advisory", "context": communication_context},
                provenance="runner_diagnostic",
            )
        )
    try:
        unsupported_bid_reprompted = False
        force_first_tool_call = objective in {"bid_seeking", "stress_test"} and not _has_seeded_accepted_candidate(persona, records)
        for round_index in range(max_tool_rounds):
            if force_first_tool_call and round_index == 0:
                message = await llm.tool_round(messages, tools, tool_choice="required")
            else:
                message = await llm.tool_round(messages, tools)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                break
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                record = await _execute_agent_tool(executor, name, arguments, simulator_semaphore, provenance="llm_requested")
                records.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "name": name,
                        "content": json.dumps(record.result if record.ok else {"ok": False, "error": record.error}),
                    }
                )
                decision = decision_from_tool_calls(records)
                if decision is not None:
                    if final_bid_guard == "simulator_exact_match" and _needs_authoritative_simulation(persona, decision, records):
                        if ablation_strategy == "cp13_llm_probe_refine_frontier":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": _frontier_feedback_prompt(persona, decision, records, tick),
                                }
                            )
                            continue
                        if preprobe_mode != "full" and unsupported_bid_reprompted:
                            downgraded = _downgrade_unsupported_bid(persona, decision, records)
                            _append_selected_candidate_diagnostic(records, downgraded)
                            return downgraded, records
                        unsupported_bid_reprompted = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You proposed a bid, but no authoritative simulator tool accepted it. "
                                    "Call your archetype's simulator tool with the candidate bid, "
                                    "then call propose_action again. If simulation rejects or is unavailable, watch or abstain."
                                ),
                            }
                        )
                        continue
                    _append_selected_candidate_diagnostic(records, decision)
                    return decision, records
        final_message = await llm.tool_round(
            [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Now call propose_action exactly once. Include action, rationale, and confidence. "
                        "Also include watch_label, risk_label, uncertainty_label, opportunity_label, watch_reasons, "
                        "priority_label, priority_score, operator_action, and priority_reason. "
                        f"{_final_action_instruction(objective)} "
                        "For action=bid include side, quantity_mwh, and limit_price_eur_mwh. "
                        f"Current experiment objective: {objective}."
                        + (f" {rationale_directive}" if rationale_directive else "")
                    ),
                },
            ],
            tools,
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
        )
        for call in final_message.get("tool_calls") or []:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            records.append(
                await _execute_agent_tool(
                    executor,
                    function.get("name", ""),
                    arguments,
                    simulator_semaphore,
                    provenance="forced_final",
                )
            )
        decision = decision_from_tool_calls(records)
        if (
            decision is not None
            and ablation_strategy in {"comm_retry_council", "comm_info_then_action"}
            and decision.action != "bid"
            and final_bid_guard == "simulator_exact_match"
            and _accepted_candidate_count(persona, records) > 0
        ):
            retry = await _retry_final_action(
                llm=llm,
                executor=executor,
                messages=messages,
                records=records,
                persona=persona,
                objective=objective,
                tools=tools,
                tick=tick,
                simulator_semaphore=simulator_semaphore,
            )
            if retry is not None:
                return retry
        if decision is not None and (
            final_bid_guard == "schema_only_shadow"
            or not _needs_authoritative_simulation(persona, decision, records)
        ):
            _append_selected_candidate_diagnostic(records, decision)
            return decision, records
        if final_bid_guard == "simulator_exact_match":
            repaired = _repair_placeholder_retry_bid(persona, records, tick=tick)
            if repaired is not None:
                _append_selected_candidate_diagnostic(records, repaired)
                return repaired, records
            if decision is not None and preprobe_mode != "full" and _needs_authoritative_simulation(persona, decision, records):
                downgraded = _downgrade_unsupported_bid(persona, decision, records)
                _append_selected_candidate_diagnostic(records, downgraded)
                return downgraded, records
        if (
            decision is None
            and ablation_strategy == "comm_info_then_action"
            and final_bid_guard == "simulator_exact_match"
            and _accepted_candidate_count(persona, records) > 0
        ):
            retry = await _retry_final_action(
                llm=llm,
                executor=executor,
                messages=messages,
                records=records,
                persona=persona,
                objective=objective,
                tools=tools,
                tick=tick,
                simulator_semaphore=simulator_semaphore,
            )
            if retry is not None:
                return retry
        if final_bid_guard == "simulator_exact_match" and ablation_strategy == "comm_retry_council":
            retry = await _retry_final_action(
                llm=llm,
                executor=executor,
                messages=messages,
                records=records,
                persona=persona,
                objective=objective,
                tools=tools,
                tick=tick,
                simulator_semaphore=simulator_semaphore,
            )
            if retry is not None:
                return retry
        return LLMBidDecision(action="abstain", rationale="model did not call propose_action", confidence=0.0), records
    except LLMClientError as exc:
        return _llm_failure(exc), records
