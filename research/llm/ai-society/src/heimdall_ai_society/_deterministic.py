from __future__ import annotations

import asyncio
import json
from datetime import timedelta

from heimdall_contracts import Persona
from heimdall_ai_society.llm_client import LLMClientError, OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tools import AgentToolExecutor, decision_from_tool_calls, openai_tool_specs
from packages.simulator import ScenarioAssetStateStore

from heimdall_ai_society._trace_helpers import (
    _append_selected_candidate_diagnostic,
    _float_result,
    _matching_accepted_simulation,
    _record_controls_acceptance,
    _same_bid,
)
from heimdall_ai_society._candidates import (
    _rank_seeded_candidates,
    _seed_candidate_tools,
    _seed_context_tools,
    _matching_candidate_row,
)
from heimdall_ai_society._context import (
    _forecast_diversity_for_candidate,
    _merge_watch_reasons,
)
from heimdall_ai_society._prompts import _required_simulation_tool, _opportunity_hint


def _deterministic_decision(persona: Persona, tick: TickContext) -> LLMBidDecision:
    lower, upper = tick.forecast.interval_for_side("up")
    if persona.archetype.value not in {"p2h", "ev"}:
        if persona.agent_id.endswith(("001", "004", "007")):
            return LLMBidDecision(
                action="abstain",
                rationale="dry-run policy: context/advisory-only agent has no simulator-backed bid adapter",
                confidence=0.25,
            )
        return LLMBidDecision(
            action="watch",
            rationale="dry-run policy: context/advisory-only agent watches this MTU but does not submit a bid",
            confidence=0.4,
            watch_label="watch",
            risk_label="medium",
            uncertainty_label="medium",
            opportunity_label="weak",
            watch_reasons=["forecast_uncertainty"],
        )
    if persona.agent_id.endswith(("001", "004", "007")):
        return LLMBidDecision(
            action="abstain",
            rationale="dry-run policy: abstain for diversity and trace coverage",
            confidence=0.25,
        )
    return LLMBidDecision(
        action="bid",
        side="up",
        quantity_mwh=max(0.25, min(2.0, persona.capacity_mw * 0.025)),
        limit_price_eur_mwh=round((lower + upper) / 2.0, 2),
        rationale=f"dry-run policy: {persona.archetype.value} bid near forecast interval midpoint",
        confidence=0.5,
        watch_label="watch",
        opportunity_label="actionable",
        watch_reasons=["accepted_bid_available"],
    )


def _deterministic_best_accepted_decision(
    *,
    persona: Persona,
    tick: TickContext,
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str = "proxy",
    asset_proxy_style: str = "market",
    asset_state_store: ScenarioAssetStateStore | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
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
    )
    records = _seed_context_tools(persona, executor)
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
    required = _required_simulation_tool(persona)
    accepted = [
        record
        for record in records
        if record.name == required
        and record.ok
        and _record_controls_acceptance(record)
        and record.result.get("accepted") is True
    ]
    if accepted:
        ranked = _rank_seeded_candidates(accepted, tick)
        best_args = ranked[0]["arguments"] if ranked else accepted[0].arguments
        decision = LLMBidDecision(
            action="bid",
            side=str(best_args["side"]),
            quantity_mwh=float(best_args["quantity_mwh"]),
            limit_price_eur_mwh=float(best_args["limit_price_eur_mwh"]),
            rationale="deterministic_best_accepted selected the highest-ranked exact simulator-accepted candidate",
            confidence=0.75,
        )
        _append_selected_candidate_diagnostic(records, decision)
        return decision, records

    watch_score = _watch_score_from_records(records)
    if watch_score >= 0.35 or any(
        record.name.startswith("simulate") and record.result.get("reason_codes")
        for record in records
    ):
        decision = LLMBidDecision(
            action="watch",
            rationale=f"deterministic_best_accepted found no accepted candidate; watch_score={watch_score:.3f}",
            confidence=max(0.35, min(0.7, watch_score)),
        )
    else:
        decision = LLMBidDecision(
            action="abstain",
            rationale=f"deterministic_best_accepted found no accepted candidate and weak watch signal; watch_score={watch_score:.3f}",
            confidence=0.3,
        )
    _append_selected_candidate_diagnostic(records, decision)
    return decision, records


def _deterministic_high_fill_accepted_decision(
    *,
    persona: Persona,
    tick: TickContext,
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str = "proxy",
    asset_proxy_style: str = "market",
    asset_state_store: ScenarioAssetStateStore | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    proposed, records = _deterministic_best_accepted_decision(
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
    accepted = [row for row in _rank_seeded_candidates(records, tick) if row.get("accepted") is True]
    accepted = sorted(accepted, key=_high_fill_candidate_sort_key, reverse=True)
    if not accepted:
        records.append(_fill_selector_review_record("no_accepted_candidate", proposed, None, False, candidate_count=0))
        return proposed, records
    best_args = accepted[0]["arguments"]
    final = LLMBidDecision(
        action="bid",
        side=str(best_args["side"]),
        quantity_mwh=float(best_args["quantity_mwh"]),
        limit_price_eur_mwh=float(best_args["limit_price_eur_mwh"]),
        rationale="deterministic_high_fill_accepted selected the exact accepted candidate with the highest clearability proxy",
        confidence=0.75,
    )
    records.append(_fill_selector_review_record("select_candidate", final, accepted[0], False, candidate_count=len(accepted)))
    _append_selected_candidate_diagnostic(records, final)
    return final, records


async def _deterministic_llm_critic_decision(
    *,
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient | None,
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    simulator_semaphore: asyncio.Semaphore,
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
    communication_context: dict[str, object] | None = None,
    memory_context: dict[str, object] | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    forecast_diversity_context: dict[str, object] | None = None,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    proposed, records = _deterministic_best_accepted_decision(
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
    forecast_summary = _forecast_diversity_for_candidate(forecast_diversity_context, proposed.side)
    records.append(
        ToolCallRecord(
            name="forecast_diversity_summary",
            arguments={"agent_id": persona.agent_id, "candidate_side": proposed.side},
            ok=True,
            result={"ok": True, "summary": forecast_summary},
            provenance="runner_diagnostic",
        )
    )
    if proposed.action != "bid" or llm is None:
        return proposed, records

    prompt_payload = {
        "critic_role": "You may only keep the deterministic simulator-backed bid or veto it to watch/abstain.",
        "hard_constraints": [
            "Do not change side, quantity_mwh, or limit_price_eur_mwh.",
            "A changed bid will be ignored and logged as a mutation attempt.",
            "Veto only for quality/risk: wrong-side risk, low clearability, high disagreement, or weak simulator economics.",
        ],
        "deterministic_bid": proposed.model_dump(mode="json"),
        "candidate_ranking": _rank_seeded_candidates(records, tick)[:8],
        "forecast_diversity_summary": forecast_summary,
        "communication_context": communication_context or {},
        "memory_context": memory_context or {},
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Heimdall bid critic, not a bidder. The deterministic proposer already selected the best exact "
                "simulator-accepted candidate. Your job is to reduce wrong-side and low-clearability bids using the visible "
                "forecast diversity summary. Return one propose_action call: keep_bid by repeating the same bid fields, "
                "veto_to_watch with action=watch, or veto_to_abstain with action=abstain."
            ),
        },
        {"role": "user", "content": json.dumps(prompt_payload, sort_keys=True)},
    ]
    tools = _critic_tool_specs()
    try:
        message = await llm.tool_round(
            messages,
            tools,
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
        )
    except LLMClientError as exc:
        records.append(
            ToolCallRecord(
                name="llm_critic_review",
                arguments={"agent_id": persona.agent_id},
                ok=True,
                result={
                    "ok": True,
                    "critic_outcome": "no_valid_critic_decision",
                    "final_action": proposed.action,
                    "llm_error": str(exc),
                    "mutation_attempt": False,
                    "veto_reasons": [],
                    "forecast_disagreement_veto": False,
                },
                provenance="runner_diagnostic",
            )
        )
        return proposed, records

    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if name != "propose_action":
            records.append(
                ToolCallRecord(
                    name="llm_critic_blocked_tool",
                    arguments={"tool": name},
                    ok=False,
                    result={"ok": False, "error_code": "critic_can_only_propose_action"},
                    provenance="runner_diagnostic",
                )
            )
            continue
        records.append(_critic_action_record(arguments))
    critic_decision = decision_from_tool_calls(records)
    return _apply_llm_critic_review(proposed, critic_decision, records, forecast_summary), records


async def _llm_fill_selector_decision(
    *,
    persona: Persona,
    tick: TickContext,
    llm: OpenAICompatibleLLMClient | None,
    objective: str,
    ablation_strategy: str,
    data_tools: object | None,
    simulator_tool: object | None,
    asset_simulator_mode: str,
    asset_proxy_style: str,
    asset_state_store: ScenarioAssetStateStore,
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
    communication_context: dict[str, object] | None = None,
    memory_context: dict[str, object] | None = None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
    forecast_diversity_context: dict[str, object] | None = None,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    fallback, records = _deterministic_high_fill_accepted_decision(
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
    accepted = [row for row in _rank_seeded_candidates(records, tick) if row.get("accepted") is True]
    accepted = sorted(accepted, key=_high_fill_candidate_sort_key, reverse=True)
    rejected = [row for row in _rank_seeded_candidates(records, tick) if row.get("accepted") is not True][:8]
    forecast_summary = _forecast_diversity_for_candidate(forecast_diversity_context, fallback.side)
    records.append(
        ToolCallRecord(
            name="forecast_diversity_summary",
            arguments={"agent_id": persona.agent_id, "candidate_side": fallback.side},
            ok=True,
            result={"ok": True, "summary": forecast_summary},
            provenance="runner_diagnostic",
        )
    )
    if not accepted or llm is None:
        records.append(_fill_selector_review_record("no_accepted_candidate" if not accepted else "no_llm", fallback, accepted[0] if accepted else None, False, candidate_count=len(accepted)))
        return fallback, records

    prompt_payload = {
        "selector_role": "Maximize bid fill reliability while keeping unsupported bids at zero.",
        "hard_constraints": [
            "Select only one exact accepted candidate by repeating side, quantity_mwh, and limit_price_eur_mwh.",
            "Do not invent, reprice, resize, or switch side. Mutations are ignored and logged.",
            "Prefer high clear_probability_proxy, then non-negative worst_case_profit_eur, then lower quantity risk.",
            "Use watch when the accepted menu is likely to produce wrong-side, no-activation, or low-clearability bids.",
        ],
        "accepted_candidate_menu": accepted[:8],
        "rejected_candidate_diagnostics": rejected,
        "forecast_diversity_summary": forecast_summary,
        "communication_context": communication_context or {},
        "memory_context": memory_context or {},
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Heimdall fill-rate selector. You are not a price setter. "
                "Your only allowed bid is an exact candidate from accepted_candidate_menu. "
                "Return one propose_action call: bid with exact candidate fields, watch, or abstain."
            ),
        },
        {"role": "user", "content": json.dumps(prompt_payload, sort_keys=True)},
    ]
    try:
        message = await llm.tool_round(
            messages,
            _critic_tool_specs(),
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
        )
    except LLMClientError as exc:
        records.append(_fill_selector_error_record(str(exc), fallback, accepted[0], len(accepted)))
        return fallback, records

    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if name != "propose_action":
            records.append(
                ToolCallRecord(
                    name="fill_selector_blocked_tool",
                    arguments={"tool": name},
                    ok=False,
                    result={"ok": False, "error_code": "fill_selector_can_only_propose_action"},
                    provenance="runner_diagnostic",
                )
            )
            continue
        records.append(_critic_action_record(arguments))
    selected = decision_from_tool_calls(records)
    return _apply_fill_selector_review(fallback, selected, accepted, records), records


def _deterministic_watch_threshold_decision(
    *,
    persona: Persona,
    tick: TickContext,
    data_tools: object | None,
    tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
) -> tuple[LLMBidDecision, list[ToolCallRecord]]:
    observed_at = tick.timestamp - timedelta(minutes=persona.info_latency_min)
    if data_tools is not None and hasattr(data_tools, "with_observed_at"):
        data_tools = data_tools.with_observed_at(observed_at)  # type: ignore[assignment,union-attr]
    executor = AgentToolExecutor(
        persona=persona,
        forecast=tick.forecast,
        data_tools=data_tools,
        simulator_tool=None,
        tool_cache=tool_cache,
    )
    records = _seed_context_tools(persona, executor)
    watch_score = _watch_score_from_records(records)
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    width = max(up_upper - up_lower, down_upper - down_lower)
    edge = max(abs(up_lower - tick.market_price_eur_mwh), abs(tick.market_price_eur_mwh - down_upper))
    if watch_score >= 0.5 or edge >= 25.0 or width >= 50.0:
        decision = LLMBidDecision(
            action="watch",
            rationale=f"deterministic_watch_threshold marked important hour: watch_score={watch_score:.3f}, edge={edge:.2f}, interval_width={width:.2f}",
            confidence=max(0.45, min(0.8, max(watch_score, edge / 100.0, width / 160.0))),
            watch_label="must_watch",
            risk_label="high" if width >= 80.0 else "medium",
            uncertainty_label="high" if width >= 80.0 else "medium",
            opportunity_label="actionable" if edge >= 25.0 else "weak",
            watch_reasons=_merge_watch_reasons(
                [],
                [
                    "activation_risk" if watch_score >= 0.5 else "",
                    "price_volatility" if edge >= 25.0 else "",
                    "forecast_uncertainty" if width >= 50.0 else "",
                ],
            ),
        )
    elif watch_score >= 0.35 or edge >= 15.0 or width >= 35.0:
        decision = LLMBidDecision(
            action="watch",
            rationale=f"deterministic_watch_threshold marked watch hour: watch_score={watch_score:.3f}, edge={edge:.2f}, interval_width={width:.2f}",
            confidence=max(0.35, min(0.65, max(watch_score, edge / 100.0, width / 160.0))),
            watch_label="watch",
            risk_label="medium",
            uncertainty_label="medium" if width >= 35.0 else "low",
            opportunity_label="weak",
            watch_reasons=_merge_watch_reasons(
                [],
                [
                    "activation_risk" if watch_score >= 0.35 else "",
                    "price_volatility" if edge >= 15.0 else "",
                    "forecast_uncertainty" if width >= 35.0 else "",
                ],
            ),
        )
    else:
        decision = LLMBidDecision(
            action="abstain",
            rationale=f"deterministic_watch_threshold found quiet hour: watch_score={watch_score:.3f}, edge={edge:.2f}, interval_width={width:.2f}",
            confidence=0.3,
        )
    return decision, records


def _watch_score_from_records(records: list[ToolCallRecord]) -> float:
    for record in records:
        if record.name == "get_activation_context":
            try:
                return float(record.result.get("watch_score", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _high_fill_candidate_sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
    try:
        quantity = float(args.get("quantity_mwh", 999.0))  # type: ignore[union-attr]
    except (TypeError, ValueError):
        quantity = 999.0
    try:
        clear_probability = float(row.get("clear_probability_proxy") or 0.0)
    except (TypeError, ValueError):
        clear_probability = 0.0
    try:
        worst_case_profit = float(row.get("worst_case_profit_eur") or 0.0)
    except (TypeError, ValueError):
        worst_case_profit = 0.0
    return clear_probability, worst_case_profit, -quantity


def _critic_tool_specs() -> list[dict[str, object]]:
    return [tool for tool in openai_tool_specs() if tool.get("function", {}).get("name") == "propose_action"]


def _critic_action_record(arguments: dict[str, object]) -> ToolCallRecord:
    try:
        decision = LLMBidDecision.model_validate(arguments)
        return ToolCallRecord(
            name="propose_action",
            arguments=arguments,
            ok=True,
            result={"ok": True, "decision": decision.model_dump(mode="json")},
            provenance="llm_requested",
        )
    except Exception as exc:
        return ToolCallRecord(
            name="propose_action",
            arguments=arguments,
            ok=False,
            error=str(exc),
            result={"ok": False, "error_code": "invalid_critic_action"},
            provenance="llm_requested",
        )


def _apply_llm_critic_review(
    proposed: LLMBidDecision,
    critic_decision: LLMBidDecision | None,
    records: list[ToolCallRecord],
    forecast_summary: dict[str, object],
) -> LLMBidDecision:
    if proposed.action != "bid":
        records.append(_critic_review_record("skipped_non_bid", proposed, critic_decision, forecast_summary))
        return proposed
    if critic_decision is None:
        records.append(_critic_review_record("no_valid_critic_decision", proposed, critic_decision, forecast_summary))
        return proposed
    if critic_decision.action in {"watch", "abstain"}:
        outcome = "veto_to_watch" if critic_decision.action == "watch" else "veto_to_abstain"
        final = LLMBidDecision(
            action=critic_decision.action,
            rationale=f"LLM critic vetoed deterministic bid: {critic_decision.rationale}",
            confidence=critic_decision.confidence,
            watch_label="must_watch" if critic_decision.action == "watch" else critic_decision.watch_label,
            risk_label=critic_decision.risk_label,
            uncertainty_label=critic_decision.uncertainty_label,
            opportunity_label=critic_decision.opportunity_label,
            watch_reasons=_merge_watch_reasons(critic_decision.watch_reasons, ["cross_agent_disagreement"] if _forecast_disagreement_veto(forecast_summary) else []),
            priority_label=critic_decision.priority_label,
            priority_score=critic_decision.priority_score,
            operator_action=critic_decision.operator_action,
            priority_reason=critic_decision.priority_reason,
        )
        records.append(_critic_review_record(outcome, final, critic_decision, forecast_summary))
        _append_selected_candidate_diagnostic(records, final)
        return final
    if _same_bid(proposed, critic_decision):
        final = proposed.model_copy(
            update={
                "rationale": f"LLM critic kept deterministic bid: {critic_decision.rationale}",
                "confidence": critic_decision.confidence,
                "watch_label": critic_decision.watch_label,
                "risk_label": critic_decision.risk_label,
                "uncertainty_label": critic_decision.uncertainty_label,
                "opportunity_label": critic_decision.opportunity_label,
                "watch_reasons": critic_decision.watch_reasons,
                "priority_label": critic_decision.priority_label,
                "priority_score": critic_decision.priority_score,
                "operator_action": critic_decision.operator_action,
                "priority_reason": critic_decision.priority_reason,
            }
        )
        records.append(_critic_review_record("keep_bid", final, critic_decision, forecast_summary))
        return final
    records.append(_critic_review_record("ignored_mutation", proposed, critic_decision, forecast_summary, mutation_attempt=True))
    return proposed


def _critic_review_record(
    outcome: str,
    final_decision: LLMBidDecision,
    critic_decision: LLMBidDecision | None,
    forecast_summary: dict[str, object],
    *,
    mutation_attempt: bool = False,
) -> ToolCallRecord:
    veto_reasons = _critic_veto_reasons(critic_decision, forecast_summary) if outcome in {"veto_to_watch", "veto_to_abstain"} else []
    return ToolCallRecord(
        name="llm_critic_review",
        arguments={
            "critic_outcome": outcome,
            "critic_decision": critic_decision.model_dump(mode="json") if critic_decision is not None else None,
        },
        ok=True,
        result={
            "ok": True,
            "critic_outcome": outcome,
            "final_action": final_decision.action,
            "mutation_attempt": mutation_attempt,
            "veto_reasons": veto_reasons,
            "forecast_disagreement_veto": outcome in {"veto_to_watch", "veto_to_abstain"} and _forecast_disagreement_veto(forecast_summary),
            "candidate_side_support": forecast_summary.get("candidate_side_support"),
            "final_decision": final_decision.model_dump(mode="json"),
        },
        provenance="runner_diagnostic",
    )


def _apply_fill_selector_review(
    fallback: LLMBidDecision,
    selected: LLMBidDecision | None,
    accepted: list[dict[str, object]],
    records: list[ToolCallRecord],
) -> LLMBidDecision:
    if selected is None:
        records.append(_fill_selector_review_record("no_valid_selector_decision", fallback, accepted[0] if accepted else None, False, candidate_count=len(accepted)))
        return fallback
    if selected.action in {"watch", "abstain"}:
        final = LLMBidDecision(
            action=selected.action,
            rationale=f"LLM fill selector chose {selected.action}: {selected.rationale}",
            confidence=selected.confidence,
            watch_label="must_watch" if selected.action == "watch" else selected.watch_label,
            risk_label=selected.risk_label,
            uncertainty_label=selected.uncertainty_label,
            opportunity_label=selected.opportunity_label,
            watch_reasons=selected.watch_reasons,
            priority_label=selected.priority_label,
            priority_score=selected.priority_score,
            operator_action=selected.operator_action,
            priority_reason=selected.priority_reason,
        )
        records.append(_fill_selector_review_record(selected.action, final, None, False, candidate_count=len(accepted)))
        _append_selected_candidate_diagnostic(records, final)
        return final
    match = _matching_candidate_row(selected, accepted)
    if match is None:
        records.append(_fill_selector_review_record("ignored_mutation", fallback, accepted[0] if accepted else None, True, candidate_count=len(accepted)))
        return fallback
    final = selected.model_copy(update={"rationale": f"LLM fill selector selected exact accepted candidate: {selected.rationale}"})
    records.append(_fill_selector_review_record("select_candidate", final, match, False, candidate_count=len(accepted)))
    _append_selected_candidate_diagnostic(records, final)
    return final


def _fill_selector_review_record(
    outcome: str,
    final_decision: LLMBidDecision,
    selected_candidate: dict[str, object] | None,
    mutation_attempt: bool,
    *,
    candidate_count: int,
) -> ToolCallRecord:
    return ToolCallRecord(
        name="fill_selector_review",
        arguments={"selector_outcome": outcome},
        ok=True,
        result={
            "ok": True,
            "selector_outcome": outcome,
            "final_action": final_decision.action,
            "mutation_attempt": mutation_attempt,
            "selected_clear_probability_proxy": selected_candidate.get("clear_probability_proxy") if selected_candidate else None,
            "selected_worst_case_profit_eur": selected_candidate.get("worst_case_profit_eur") if selected_candidate else None,
            "candidate_count": candidate_count,
            "final_decision": final_decision.model_dump(mode="json"),
        },
        provenance="runner_diagnostic",
    )


def _fill_selector_error_record(error: str, fallback: LLMBidDecision, selected_candidate: dict[str, object] | None, candidate_count: int) -> ToolCallRecord:
    record = _fill_selector_review_record("llm_error_fallback", fallback, selected_candidate, False, candidate_count=candidate_count)
    record.result["llm_error"] = error
    return record


def _forecast_disagreement_veto(forecast_summary: dict[str, object]) -> bool:
    support = str(forecast_summary.get("candidate_side_support", ""))
    disagreement = forecast_summary.get("interval_disagreement", {})
    spread = 0.0
    if isinstance(disagreement, dict):
        try:
            spread = float(disagreement.get("max_mid_spread_eur_mwh", 0.0) or 0.0)
        except (TypeError, ValueError):
            spread = 0.0
    return support in {"minority_supported", "mixed"} or spread >= 35.0


def _critic_veto_reasons(critic_decision: LLMBidDecision | None, forecast_summary: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if _forecast_disagreement_veto(forecast_summary):
        reasons.append("forecast_disagreement")
    if critic_decision is not None:
        if critic_decision.risk_label == "high":
            reasons.append("high_risk")
        if critic_decision.uncertainty_label == "high":
            reasons.append("high_uncertainty")
        if "price_volatility" in critic_decision.watch_reasons:
            reasons.append("price_volatility")
        if "activation_risk" in critic_decision.watch_reasons:
            reasons.append("activation_risk")
    return list(dict.fromkeys(reasons or ["critic_veto"]))
