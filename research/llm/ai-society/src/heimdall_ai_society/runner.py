from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from heimdall_contracts import Persona

from heimdall_ai_society.config import SocietyRunConfig
from heimdall_ai_society.llm_client import LLMClientError, OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import (
    RealMarketContext,
    SyntheticMarketContext,
    TickContext,
)
from heimdall_ai_society.memory import (
    MemoryItem,
    load_memory_bank,
    memory_audit_summary,
    memory_fingerprint,
    memory_prompt_context,
    select_memory_items,
)
from heimdall_ai_society.personas import build_personas
from heimdall_ai_society.schemas import LLMBidDecision, SocietyTraceRecord, ToolCallRecord
from heimdall_ai_society.tool_policy import policy_for_persona
from heimdall_ai_society.rag import RAGRetriever
from heimdall_ai_society.tools import (
    AgentToolExecutor,
    build_simulator_tool,
    commit_asset_state_from_record,
    decision_from_tool_calls,
    decision_to_bid,
    mock_verify,
    openai_tool_specs,
    retrieve_knowledge_tool_spec,
)
from packages.simulator import ScenarioAssetStateStore

# ---------------------------------------------------------------------------
# Re-exports from submodules — preserve backward compatibility for tests and
# other importers that reference ``heimdall_ai_society.runner.<symbol>``.
# ---------------------------------------------------------------------------

from heimdall_ai_society._constants import *  # noqa: F401,F403  — constants

from heimdall_ai_society._trace_helpers import (  # noqa: F401
    _with_provenance,
    _with_provenance_all,
    _tool_provenance_counts,
    _trace_tool_counter_fields,
    _is_evidence_record,
    _evidence_tool_call_count,
    _action_relevant_probe_count,
    _has_action_relevant_probe,
    _has_self_simulator_probe,
    _record_controls_acceptance,
    _has_seeded_accepted_candidate,
    _latest_rejected_required,
    _matching_accepted_simulation,
    _accepted_candidate_count,
    _append_selected_candidate_diagnostic,
    _same_bid,
    _float_result,
    _served_model,
    _served_models,
    _llm_failure,
)

from heimdall_ai_society._prompts import (  # noqa: F401
    _prompt,
    _decision_instruction,
    _candidate_probe_instruction,
    _tool_policy_guidance,
    _opportunity_hint,
    _ablation_instruction,
    _objective_instruction,
    _final_action_instruction,
    _compact_tool_records_for_prompt,
    _compact_candidate,
    _frontier_feedback_prompt,
    _required_simulation_tool,
    _feasibility_tool,
    _required_feasibility_tool,
)

from heimdall_ai_society._candidates import (  # noqa: F401
    _seed_candidate_tools,
    _seed_context_tools,
    _seed_specialist_tools,
    _candidate_arguments,
    _candidate_price_ablation_arguments,
    _specialist_candidate_arguments,
    _candidate_quantity_cap,
    _sizing_quantity_ladder,
    _dedupe_candidates,
    _candidate_menu,
    _clear_probability_proxy,
    _rank_seeded_candidates,
    _matching_candidate_row,
    _resize_to_physical_limit,
)

from heimdall_ai_society._context import (  # noqa: F401
    _filter_memory_bank,
    _forecast_backend_by_agent,
    _forecast_diversity_context,
    _forecast_diversity_for_candidate,
    _forecast_side_signal,
    _majority_side,
    _candidate_side_support,
    _update_side_diagnostics,
    _auto_rag_query,
    _seed_retrieval,
    _market_intelligence_digest,
    _enrich_intelligence_decision,
    _merge_watch_reasons,
    _communication_context,
    _peer_summary,
)

from heimdall_ai_society._deterministic import (  # noqa: F401
    _deterministic_decision,
    _deterministic_best_accepted_decision,
    _deterministic_high_fill_accepted_decision,
    _deterministic_llm_critic_decision,
    _llm_fill_selector_decision,
    _deterministic_watch_threshold_decision,
    _watch_score_from_records,
    _high_fill_candidate_sort_key,
    _critic_tool_specs,
    _critic_action_record,
    _apply_llm_critic_review,
    _critic_review_record,
    _apply_fill_selector_review,
    _fill_selector_review_record,
    _fill_selector_error_record,
    _forecast_disagreement_veto,
    _critic_veto_reasons,
)

from heimdall_ai_society._deliberation import (  # noqa: F401
    _run_deliberation_protocol_tick,
    _deliberation_inquiry_for_persona,
    _deliberation_final_for_persona,
    _build_deliberation_board,
    _peer_requests_for_persona,
    _minimal_deliberation_context,
    _first_peer_candidate,
    _clear_seeking_candidate,
    _deliberation_diagnostic,
    _empty_deliberation_totals,
    _accumulate_deliberation_totals,
    _finalize_deliberation_totals,
)

from heimdall_ai_society._decision import (  # noqa: F401
    _decide_for_persona,
    _decide_with_tools,
    _execute_agent_tool,
    _executor_for_persona,
    _run_tool_round,
    _phase_tool_specs,
    _tool_name,
    _tool_names,
    _openai_tool_specs_for_safety_toolset,
    _shadow_required_simulation,
    _enforce_action_policy,
    _needs_authoritative_simulation,
    _downgrade_unsupported_bid,
    _repair_placeholder_retry_bid,
    _retry_final_action,
    _p2h_v2_price_regime_filter_bid_decision,
    _ev_v2_caution_filter_bid_decision,
    _ev_low_price_down_probe_allowed,
    _risk_filter_bid_decision,
    _is_info_archetype,
)

from heimdall_ai_society._strategies import (  # noqa: F401
    _run_info_then_action_tick,
    _central_supervisor_decision,
    _execution_gateway_validate,
    _gateway_downgrade,
    _gateway_record,
    _supervisor_quota_status,
    _supervisor_candidate_menu,
    _commit_supervisor_selected_asset_state,
    _supervisor_specialist_reports,
    _chair_decision,
    _rank_chair_candidates,
    _chair_rationale,
    _initial_bid_budget_state,
    _bid_budget_prompt_context,
    _with_bid_budget_context,
    _bid_budget_context_record,
    _bid_budget_exhausted_record,
    _enforce_bid_budget,
    _update_bid_budget_state,
    _uses_society_communication,
    _uses_chair,
    _initial_supervisor_quota_state,
    _empty_supervisor_totals,
    _accumulate_supervisor_totals,
    _finalize_supervisor_totals,
    _central_supervisor_trace_record,
    _chair_trace_record,
    _decision_audit_record,
    _specialist_recommendation_record,
    _agent_role,
)

from heimdall_ai_society._totals import (  # noqa: F401
    _empty_critic_totals,
    _accumulate_critic_totals,
    _finalize_critic_totals,
    _empty_fill_selector_totals,
    _accumulate_fill_selector_totals,
    _finalize_fill_selector_totals,
    _mean_or_none,
)


async def run_society(config: SocietyRunConfig) -> Path:
    run_id = config.run_id or f"society-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"
    started = time.perf_counter()

    personas = build_personas(config.agent_count, config.archetype_cycle, profile=config.persona_profile, seed=config.seed)
    forecast_backend_by_agent, forecast_routing_warnings = _forecast_backend_by_agent(
        personas,
        fallback_backend=config.forecaster_backend,
        routing_mode=config.forecaster_routing_mode,
    )
    forecaster_seed = config.forecaster_seed if config.forecaster_seed is not None else config.seed
    if config.market_context == "real":
        market = RealMarketContext(
            zone=config.zone,
            start=config.start_timestamp,
            data_start=config.data_start,
            data_end=config.data_end,
            default_lookback_hours=config.default_lookback_hours,
            cache_refresh=config.cache_refresh,
            weather_locations=config.weather_locations,
            forecaster_backend=config.forecaster_backend,
            seed=forecaster_seed,
            context_dataset_dir=config.context_dataset_dir,
            cache_dir=config.data_cache_dir,
        )
    else:
        market = SyntheticMarketContext(
            seed=forecaster_seed,
            zone=config.zone,
            start=config.start_timestamp,
            forecaster_backend=config.forecaster_backend,
        )
    llm = None
    if config.llm.enabled and config.chooser_mode in {"llm", "deterministic_llm_critic", "llm_fill_selector"}:
        llm = OpenAICompatibleLLMClient(
            base_url=config.llm.base_url,
            base_urls=config.llm.base_urls,
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout_seconds=config.llm.timeout_seconds,
            provider=config.llm.provider,
            http_referer=config.llm.http_referer,
            app_title=config.llm.app_title,
            supports_response_format=config.llm.supports_response_format,
            max_concurrency=config.llm.max_concurrency,
            per_endpoint_max_concurrency=config.llm.per_endpoint_max_concurrency,
        )
        served_models = (
            _served_models(config.llm.endpoint_urls, config.llm.api_key)
            if config.llm.require_served_model_match
            else {}
        )
        mismatched = {endpoint: model for endpoint, model in served_models.items() if model != config.llm.model}
        if config.llm.require_served_model_match and mismatched:
            raise RuntimeError(f"configured LLM endpoints do not all serve {config.llm.model}: {mismatched}")
    else:
        served_models = {}

    simulator_tool = build_simulator_tool(tau_eur=config.verifier_tau_eur) if config.verifier_mode == "simulator" else None
    asset_state_store = ScenarioAssetStateStore.empty()
    simulator_semaphore = asyncio.Semaphore(config.simulator_max_concurrency)
    memory_bank = load_memory_bank(config.memory_bank_path) if config.memory_enabled else []
    memory_by_agent = {}
    if config.memory_enabled:
        scoped_bank = _filter_memory_bank(memory_bank, config.memory_scope_filter)
        for idx, persona in enumerate(personas):
            agent_role = _agent_role(config.persona_profile, idx)
            if config.memory_scope_filter == "synthesis" and agent_role not in {
                "society_chair",
                "risk_officer",
                "explanation_editor",
            }:
                memory_by_agent[persona.agent_id] = []
                continue
            items = select_memory_items(
                scoped_bank,
                agent_id=persona.agent_id,
                archetype=persona.archetype.value,
                agent_role=agent_role,
                run_start=config.start_timestamp,
                max_items_per_agent=config.memory_max_items_per_agent,
                max_prompt_chars=config.memory_max_prompt_chars,
            )
            memory_by_agent[persona.agent_id] = items
    retriever = None
    if config.rag.enabled:
        if config.rag.corpus_path is None:
            raise ValueError("rag.enabled requires rag.corpus_path")
        retriever = RAGRetriever.build(
            config.rag.corpus_path,
            backend=config.rag.backend,
            model_name=config.rag.embedding_model,
            device=config.rag.device,
            cache_dir=config.rag.cache_dir,
        )
    accepted = rejected = abstained = watched = invalid = 0
    tool_provenance_totals = {
        "runner_seeded": 0,
        "llm_requested": 0,
        "forced_final": 0,
        "runner_diagnostic": 0,
        "retry": 0,
        "unknown": 0,
    }
    side_diagnostics: dict[str, object] = {
        "final_bid_side_counts": {"up": 0, "down": 0},
        "accepted_simulator_candidate_counts": {},
        "rejected_simulator_candidate_counts": {},
        "agent_forecast_backend_counts": {},
    }
    deliberation_totals = _empty_deliberation_totals()
    critic_totals = _empty_critic_totals()
    fill_selector_totals = _empty_fill_selector_totals()
    supervisor_totals = _empty_supervisor_totals()
    supervisor_quota_state = _initial_supervisor_quota_state(config.supervisor_soft_quota_per_24_ticks, config.ticks)
    bid_budget_states = {
        persona.agent_id: _initial_bid_budget_state(config.bid_budget_per_agent)
        for persona in personas
    }
    bid_budget_exhausted_count = 0
    with trace_path.open("w", encoding="utf-8") as handle:
        for step in range(config.ticks):
            tick = market.next_tick()
            persona_ticks = [
                market.tick_for_forecaster(tick, forecast_backend_by_agent[persona.agent_id])
                if hasattr(market, "tick_for_forecaster")
                else tick
                for persona in personas
            ]
            forecast_diversity_context = _forecast_diversity_context(
                personas=personas,
                persona_ticks=persona_ticks,
                forecast_backend_by_agent=forecast_backend_by_agent,
            )
            data_tools = market.tools_for_tick(tick)
            tool_cache: dict[tuple[str, str], ToolCallRecord] = {}
            shared_digest = {}
            if config.ablation_strategy != "comm_deliberation_protocol":
                shared_digest = _market_intelligence_digest(
                    tick=tick,
                    personas=personas,
                    data_tools=data_tools,
                    simulator_tool=simulator_tool,
                    asset_simulator_mode=config.asset_simulator_mode,
                    asset_proxy_style=config.asset_proxy_style,
                    asset_state_store=asset_state_store,
                    objective=config.objective,
                    ablation_strategy=config.ablation_strategy,
                    safety_toolset=config.safety_toolset,
                    preprobe_mode=config.preprobe_mode,
                    tool_cache=tool_cache,
                )
            if config.ablation_strategy == "comm_deliberation_protocol":
                outcomes = await _run_deliberation_protocol_tick(
                    personas=personas,
                    tick=tick,
                    persona_ticks=persona_ticks,
                    llm=llm,
                    profile=config.persona_profile,
                    objective=config.objective,
                    data_tools=data_tools,
                    simulator_tool=simulator_tool,
                    asset_simulator_mode=config.asset_simulator_mode,
                    asset_proxy_style=config.asset_proxy_style,
                    asset_state_store=asset_state_store,
                    simulator_semaphore=simulator_semaphore,
                    max_tool_rounds=config.max_tool_rounds,
                    final_bid_guard=config.final_bid_guard,
                    safety_toolset=config.safety_toolset,
                    candidate_sizing_mode=config.candidate_sizing_mode,
                    candidate_sizing_cap_fraction=config.candidate_sizing_cap_fraction,
                    candidate_sizing_min_mwh=config.candidate_sizing_min_mwh,
                    candidate_sizing_max_candidates=config.candidate_sizing_max_candidates,
                    deliberation_inquiry_rounds=config.deliberation_inquiry_rounds,
                    deliberation_action_rounds=config.deliberation_action_rounds,
                    deliberation_min_tool_calls=config.deliberation_min_tool_calls,
                    deliberation_require_action_probe=config.deliberation_require_action_probe,
                    deliberation_max_peer_notes=config.deliberation_max_peer_notes,
                    memory_by_agent=memory_by_agent,
                    tool_cache=tool_cache,
                )
            elif config.ablation_strategy in {
                "comm_broadcast_digest",
                "comm_broadcast_digest_risk_filter",
                "comm_broadcast_digest_priority_calibration",
            }:
                communication_context = _communication_context(
                    strategy=config.ablation_strategy,
                    tick=tick,
                    personas=personas,
                    peer_summaries=[],
                    shared_digest=shared_digest,
                )
                tasks = [
                    _decide_for_persona(
                        persona,
                        persona_ticks[idx],
                        llm,
                        agent_role=_agent_role(config.persona_profile, idx),
                        tool_mode=config.tool_mode,
                        objective=config.objective,
                        ablation_strategy=config.ablation_strategy,
                        data_tools=data_tools,
                        simulator_tool=simulator_tool,
                        asset_simulator_mode=config.asset_simulator_mode,
                        asset_proxy_style=config.asset_proxy_style,
                        asset_state_store=asset_state_store,
                        simulator_semaphore=simulator_semaphore,
                        max_tool_rounds=config.max_tool_rounds,
                        final_bid_guard=config.final_bid_guard,
                        safety_toolset=config.safety_toolset,
                        preprobe_mode=config.preprobe_mode,
                        candidate_sizing_mode=config.candidate_sizing_mode,
                        candidate_sizing_cap_fraction=config.candidate_sizing_cap_fraction,
                        candidate_sizing_min_mwh=config.candidate_sizing_min_mwh,
                        candidate_sizing_max_candidates=config.candidate_sizing_max_candidates,
                        chooser_mode=config.chooser_mode,
                        communication_context=_with_bid_budget_context(
                            communication_context,
                            _bid_budget_prompt_context(
                                bid_budget_states[persona.agent_id],
                                enabled=config.bid_budget_enabled,
                                history_ticks=config.bid_budget_history_ticks,
                            ),
                        ),
                        memory_context=memory_prompt_context(memory_by_agent.get(persona.agent_id, [])),
                        tool_cache=tool_cache,
                        forecast_diversity_context=forecast_diversity_context,
                        retriever=retriever,
                        rag_top_k=config.rag.top_k,
                        rag_max_chars=config.rag.max_doc_chars,
                    )
                    for idx, persona in enumerate(personas)
                ]
                outcomes = await asyncio.gather(*tasks)
            elif config.ablation_strategy == "comm_info_then_action":
                outcomes = await _run_info_then_action_tick(
                    personas=personas,
                    tick=tick,
                    persona_ticks=persona_ticks,
                    llm=llm,
                    profile=config.persona_profile,
                    tool_mode=config.tool_mode,
                    objective=config.objective,
                    ablation_strategy=config.ablation_strategy,
                    data_tools=data_tools,
                    simulator_tool=simulator_tool,
                    asset_simulator_mode=config.asset_simulator_mode,
                    asset_proxy_style=config.asset_proxy_style,
                    asset_state_store=asset_state_store,
                    simulator_semaphore=simulator_semaphore,
                    max_tool_rounds=config.max_tool_rounds,
                    final_bid_guard=config.final_bid_guard,
                    safety_toolset=config.safety_toolset,
                    preprobe_mode=config.preprobe_mode,
                    candidate_sizing_mode=config.candidate_sizing_mode,
                    candidate_sizing_cap_fraction=config.candidate_sizing_cap_fraction,
                    candidate_sizing_min_mwh=config.candidate_sizing_min_mwh,
                    candidate_sizing_max_candidates=config.candidate_sizing_max_candidates,
                    chooser_mode=config.chooser_mode,
                    shared_digest=shared_digest,
                    memory_by_agent=memory_by_agent,
                    tool_cache=tool_cache,
                    forecast_diversity_context=forecast_diversity_context,
                )
            elif _uses_society_communication(config.ablation_strategy):
                peer_summaries: list[dict[str, object]] = []
                outcomes = []
                for idx, persona in enumerate(personas):
                    communication_context = _communication_context(
                        strategy=config.ablation_strategy,
                        tick=tick,
                        personas=personas,
                        peer_summaries=peer_summaries,
                        shared_digest=shared_digest,
                    )
                    decision, tool_calls = await _decide_for_persona(
                        persona,
                        persona_ticks[idx],
                        llm,
                        agent_role=_agent_role(config.persona_profile, idx),
                        tool_mode=config.tool_mode,
                        objective=config.objective,
                        ablation_strategy=config.ablation_strategy,
                        data_tools=data_tools,
                        simulator_tool=simulator_tool,
                        asset_simulator_mode=config.asset_simulator_mode,
                        asset_proxy_style=config.asset_proxy_style,
                        asset_state_store=asset_state_store,
                        simulator_semaphore=simulator_semaphore,
                        max_tool_rounds=config.max_tool_rounds,
                        final_bid_guard=config.final_bid_guard,
                        safety_toolset=config.safety_toolset,
                        preprobe_mode=config.preprobe_mode,
                        candidate_sizing_mode=config.candidate_sizing_mode,
                        candidate_sizing_cap_fraction=config.candidate_sizing_cap_fraction,
                        candidate_sizing_min_mwh=config.candidate_sizing_min_mwh,
                        candidate_sizing_max_candidates=config.candidate_sizing_max_candidates,
                        chooser_mode=config.chooser_mode,
                        communication_context=_with_bid_budget_context(
                            communication_context,
                            _bid_budget_prompt_context(
                                bid_budget_states[persona.agent_id],
                                enabled=config.bid_budget_enabled,
                                history_ticks=config.bid_budget_history_ticks,
                            ),
                        ),
                        memory_context=memory_prompt_context(memory_by_agent.get(persona.agent_id, [])),
                        tool_cache=tool_cache,
                        forecast_diversity_context=forecast_diversity_context,
                    )
                    outcomes.append((decision, tool_calls))
                    peer_summaries.append(_peer_summary(persona, decision, tool_calls))
            else:
                tasks = [
                    _decide_for_persona(
                        persona,
                        persona_ticks[idx],
                        llm,
                        agent_role=_agent_role(config.persona_profile, idx),
                        tool_mode=config.tool_mode,
                        objective=config.objective,
                        ablation_strategy=config.ablation_strategy,
                        data_tools=data_tools,
                        simulator_tool=simulator_tool,
                        asset_simulator_mode=config.asset_simulator_mode,
                        asset_proxy_style=config.asset_proxy_style,
                        asset_state_store=asset_state_store,
                        simulator_semaphore=simulator_semaphore,
                        max_tool_rounds=config.max_tool_rounds,
                        final_bid_guard=config.final_bid_guard,
                        safety_toolset=config.safety_toolset,
                        preprobe_mode=config.preprobe_mode,
                        candidate_sizing_mode=config.candidate_sizing_mode,
                        candidate_sizing_cap_fraction=config.candidate_sizing_cap_fraction,
                        candidate_sizing_min_mwh=config.candidate_sizing_min_mwh,
                        candidate_sizing_max_candidates=config.candidate_sizing_max_candidates,
                        chooser_mode=config.chooser_mode,
                        communication_context=_with_bid_budget_context(
                            shared_digest,
                            _bid_budget_prompt_context(
                                bid_budget_states[persona.agent_id],
                                enabled=config.bid_budget_enabled,
                                history_ticks=config.bid_budget_history_ticks,
                            ),
                        ),
                        memory_context=memory_prompt_context(memory_by_agent.get(persona.agent_id, [])),
                        tool_cache=tool_cache,
                        forecast_diversity_context=forecast_diversity_context,
                        seed_outage_context=config.seed_outage_context,
                        rationale_directive=config.rationale_directive,
                        retriever=retriever,
                        rag_top_k=config.rag.top_k,
                        rag_max_chars=config.rag.max_doc_chars,
                    )
                    for idx, persona in enumerate(personas)
                ]
                outcomes = await asyncio.gather(*tasks)
            tick_records: list[SocietyTraceRecord] = []
            chair_inputs: list[tuple[Persona, str, LLMBidDecision, list[ToolCallRecord], bool | None, list[str]]] = []
            for idx, (persona, (raw_decision, tool_calls)) in enumerate(zip(personas, outcomes, strict=True)):
                agent_role = _agent_role(config.persona_profile, idx)
                agent_tick = persona_ticks[idx]
                decision = _enrich_intelligence_decision(raw_decision, agent_tick, tool_calls)
                effective_decision = _enforce_action_policy(
                    persona,
                    decision,
                    tool_calls,
                    agent_role=agent_role,
                    ablation_strategy=config.ablation_strategy,
                    tick=agent_tick,
                    final_bid_guard=config.final_bid_guard,
                )
                if config.ablation_strategy == "comm_central_supervisor" and effective_decision.action == "bid":
                    tool_calls.append(_specialist_recommendation_record(effective_decision))
                    effective_decision = effective_decision.model_copy(
                        update={
                            "action": "watch",
                            "side": None,
                            "quantity_mwh": None,
                            "limit_price_eur_mwh": None,
                            "rationale": "central supervisor mode converted specialist bid to a non-executing recommendation",
                            "watch_label": "must_watch",
                            "watch_reasons": _merge_watch_reasons(effective_decision.watch_reasons, ["accepted_bid_available"]),
                        }
                    )
                if tick.unavailable_reason is not None:
                    effective_decision = LLMBidDecision(
                        action="abstain",
                        rationale=f"real market data unavailable: {tick.unavailable_reason}",
                        confidence=0.0,
                    )
                if config.bid_budget_enabled:
                    budget_context = _bid_budget_prompt_context(
                        bid_budget_states[persona.agent_id],
                        enabled=True,
                        history_ticks=config.bid_budget_history_ticks,
                    )
                    tool_calls.append(_bid_budget_context_record(budget_context))
                    effective_decision, exhausted = _enforce_bid_budget(effective_decision, budget_context)
                    if exhausted:
                        bid_budget_exhausted_count += 1
                        tool_calls.append(_bid_budget_exhausted_record(budget_context))
                verdict, reasons = mock_verify(effective_decision)
                bid = decision_to_bid(persona, effective_decision, agent_tick.forecast)
                if config.final_bid_guard == "schema_only_shadow" and bid is not None and simulator_tool is not None:
                    shadow = _shadow_required_simulation(
                        persona=persona,
                        decision=effective_decision,
                        tick=agent_tick,
                        data_tools=data_tools,
                        simulator_tool=simulator_tool,
                        asset_simulator_mode=config.asset_simulator_mode,
                        asset_proxy_style=config.asset_proxy_style,
                        asset_state_store=asset_state_store,
                    )
                    if shadow is not None:
                        tool_calls.append(shadow)
                    verdict = None
                    reasons = ["final_guard_disabled_shadow_scored"]
                elif config.verifier_mode == "simulator" and bid is not None and simulator_tool is not None:
                    sim_record = _matching_accepted_simulation(_required_simulation_tool(persona), effective_decision, tool_calls)
                    if sim_record is not None:
                        verdict = True
                        reasons = []
                        if persona.archetype.value != "p2h":
                            commit_asset_state_from_record(
                                state_store=asset_state_store,
                                simulator_tool=simulator_tool,
                                persona=persona,
                                forecast=agent_tick.forecast,
                                record=sim_record,
                            )
                    else:
                        verdict = False
                        reasons = ["missing_controlling_simulation"]

                if effective_decision.action == "abstain":
                    abstained += 1
                elif effective_decision.action == "watch":
                    watched += 1
                elif verdict:
                    accepted += 1
                    if effective_decision.side in {"up", "down"}:
                        side_counts = side_diagnostics["final_bid_side_counts"]  # type: ignore[assignment]
                        side_counts[effective_decision.side] = int(side_counts.get(effective_decision.side, 0)) + 1
                elif verdict is False:
                    rejected += 1
                else:
                    invalid += 1
                if config.bid_budget_enabled:
                    _update_bid_budget_state(
                        bid_budget_states[persona.agent_id],
                        step=step,
                        decision=effective_decision,
                        verifier_accepted=verdict,
                        verifier_reason_codes=reasons,
                        history_ticks=config.bid_budget_history_ticks,
                    )
                _update_side_diagnostics(
                    side_diagnostics,
                    persona=persona,
                    forecast_backend=forecast_backend_by_agent[persona.agent_id],
                    tool_calls=tool_calls,
                )
                _accumulate_deliberation_totals(deliberation_totals, persona=persona, decision=effective_decision, records=tool_calls)
                _accumulate_critic_totals(critic_totals, records=tool_calls)
                _accumulate_fill_selector_totals(fill_selector_totals, records=tool_calls)

                interval = agent_tick.forecast.interval_for_side(effective_decision.side or "up")
                record = SocietyTraceRecord(
                    run_id=run_id,
                    step=step,
                    timestamp=tick.timestamp,
                    observed_at=tick.timestamp - timedelta(minutes=persona.info_latency_min),
                    agent_id=persona.agent_id,
                    zone=config.zone,
                    archetype=persona.archetype.value,
                    agent_role=agent_role,
                    llm_id=persona.llm_id,
                    forecaster_id=persona.forecaster_id,
                    forecast_backend=forecast_backend_by_agent[persona.agent_id],
                    decision=effective_decision,
                    verifier_mode=config.verifier_mode,
                    verifier_accepted=verdict,
                    verifier_reason_codes=reasons,
                    market_price_eur_mwh=agent_tick.market_price_eur_mwh,
                    forecast_interval_eur_mwh=interval,
                    rationale=effective_decision.rationale,
                    unavailable_reason=tick.unavailable_reason,
                    tool_calls=tool_calls,
                    **_trace_tool_counter_fields(tool_calls),
                    memory_item_count=len(memory_by_agent.get(persona.agent_id, [])),
                    memory_fingerprint=memory_fingerprint(memory_by_agent.get(persona.agent_id, [])) if memory_by_agent.get(persona.agent_id) else None,
                    memory_lessons=memory_audit_summary(memory_by_agent.get(persona.agent_id, [])),
                )
                tick_records.append(record)
                chair_inputs.append((persona, agent_role, effective_decision, tool_calls, verdict, reasons))
            if _uses_chair(config.ablation_strategy):
                chair_record = _chair_trace_record(
                    run_id=run_id,
                    step=step,
                    tick=tick,
                    zone=config.zone,
                    strategy=config.ablation_strategy,
                    inputs=chair_inputs,
                )
                tick_records.append(chair_record)
                decision = chair_record.decision
                if decision.action == "abstain":
                    abstained += 1
                elif decision.action == "watch":
                    watched += 1
                elif chair_record.verifier_accepted:
                    accepted += 1
                elif chair_record.verifier_accepted is False:
                    rejected += 1
                else:
                    invalid += 1
            if config.ablation_strategy == "comm_central_supervisor":
                supervisor_record = await _central_supervisor_trace_record(
                    run_id=run_id,
                    step=step,
                    tick=tick,
                    zone=config.zone,
                    llm=llm,
                    inputs=chair_inputs,
                    quota_state=supervisor_quota_state,
                    soft_quota_per_24_ticks=config.supervisor_soft_quota_per_24_ticks,
                    max_orders_per_tick=config.supervisor_max_orders_per_tick,
                    memory_context=memory_prompt_context(memory_by_agent.get("supervisor-000", [])),
                )
                tick_records.append(supervisor_record)
                _accumulate_supervisor_totals(supervisor_totals, supervisor_record)
                decision = supervisor_record.decision
                if decision.action == "bid":
                    _commit_supervisor_selected_asset_state(
                        supervisor_record=supervisor_record,
                        personas=personas,
                        simulator_tool=simulator_tool,
                        asset_state_store=asset_state_store,
                        tick=tick,
                    )
                if decision.action == "abstain":
                    abstained += 1
                elif decision.action == "watch":
                    watched += 1
                elif supervisor_record.verifier_accepted:
                    accepted += 1
                    if decision.side in {"up", "down"}:
                        side_counts = side_diagnostics["final_bid_side_counts"]  # type: ignore[assignment]
                        side_counts[decision.side] = int(side_counts.get(decision.side, 0)) + 1
                elif supervisor_record.verifier_accepted is False:
                    rejected += 1
                else:
                    invalid += 1
            for record in tick_records:
                for provenance, count in record.tool_call_provenance_counts.items():
                    tool_provenance_totals[provenance] = tool_provenance_totals.get(provenance, 0) + int(count)
                handle.write(record.model_dump_json() + "\n")
            handle.flush()
            tool_cache.clear()

    elapsed = time.perf_counter() - started
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "agent_count": config.agent_count,
        "ticks": config.ticks,
        "llm_enabled": config.llm.enabled,
        "chooser_mode": config.chooser_mode,
        "forecaster_backend": config.forecaster_backend,
        "forecast_backend_by_agent": forecast_backend_by_agent,
        "forecast_routing_warnings": forecast_routing_warnings,
        "verifier_mode": config.verifier_mode,
        "final_bid_guard": config.final_bid_guard,
        "safety_toolset": config.safety_toolset,
        "asset_simulator_mode": config.asset_simulator_mode,
        "asset_proxy_style": config.asset_proxy_style,
        "candidate_sizing_mode": config.candidate_sizing_mode,
        "candidate_sizing_cap_fraction": config.candidate_sizing_cap_fraction,
        "candidate_sizing_min_mwh": config.candidate_sizing_min_mwh,
        "candidate_sizing_max_candidates": config.candidate_sizing_max_candidates,
        "deliberation_inquiry_rounds": config.deliberation_inquiry_rounds,
        "deliberation_action_rounds": config.deliberation_action_rounds,
        "deliberation_min_tool_calls": config.deliberation_min_tool_calls,
        "deliberation_require_action_probe": config.deliberation_require_action_probe,
        "deliberation_max_peer_notes": config.deliberation_max_peer_notes,
        "tool_policy": config.tool_policy,
        "market_context": config.market_context,
        "tool_mode": config.tool_mode,
        "preprobe_mode": config.preprobe_mode,
        "ablation_strategy": config.ablation_strategy,
        "persona_profile": config.persona_profile,
        "llm_model_configured": config.llm.model,
        "llm_provider": config.llm.provider,
        "llm_base_urls": config.llm.endpoint_urls,
        "llm_require_served_model_match": config.llm.require_served_model_match,
        "llm_supports_response_format": config.llm.supports_response_format,
        "llm_supports_tools": config.llm.supports_tools,
        "llm_model_served": next(iter(served_models.values()), None) if config.llm.enabled else None,
        "llm_models_served": served_models if config.llm.enabled else {},
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "watched": watched,
        "invalid": invalid,
        "trace_path": str(trace_path),
        "memory_enabled": config.memory_enabled,
        "memory_bank_path": str(config.memory_bank_path) if config.memory_bank_path is not None else None,
        "memory_items_loaded": len(memory_bank),
        "rag_enabled": config.rag.enabled,
        "rag": retriever.stats() if retriever is not None else None,
        "runtime_seconds": round(elapsed, 6),
        "runtime_seconds_per_tick": round(elapsed / max(config.ticks, 1), 6),
        "max_concurrency": config.llm.max_concurrency,
        "per_endpoint_max_concurrency": config.llm.per_endpoint_max_concurrency,
        "simulator_max_concurrency": config.simulator_max_concurrency,
        "side_diagnostics": side_diagnostics,
        "tool_call_provenance_counts": tool_provenance_totals,
        "seeded_tool_call_count": tool_provenance_totals["runner_seeded"],
        "llm_tool_call_count": tool_provenance_totals["llm_requested"],
        "forced_tool_call_count": tool_provenance_totals["forced_final"],
        "diagnostic_tool_call_count": tool_provenance_totals["runner_diagnostic"],
        "retry_tool_call_count": tool_provenance_totals["retry"],
        "unknown_tool_call_count": tool_provenance_totals["unknown"],
        "deliberation_metrics": _finalize_deliberation_totals(deliberation_totals),
        "llm_critic_metrics": _finalize_critic_totals(critic_totals),
        "llm_critic_keep_count": critic_totals["keep_count"],
        "llm_critic_veto_count": critic_totals["veto_count"],
        "llm_critic_veto_reason_counts": critic_totals["veto_reason_counts"],
        "llm_critic_mutation_attempt_count": critic_totals["mutation_attempt_count"],
        "forecast_disagreement_veto_count": critic_totals["forecast_disagreement_veto_count"],
        "fill_selector_metrics": _finalize_fill_selector_totals(fill_selector_totals),
        "fill_selector_candidate_count": fill_selector_totals["candidate_count"],
        "fill_selector_selected_clear_probability_proxy": _mean_or_none(fill_selector_totals["selected_clear_probability_proxy_values"]),
        "fill_selector_selected_worst_case_profit_eur": _mean_or_none(fill_selector_totals["selected_worst_case_profit_values"]),
        "fill_selector_mutation_attempt_count": fill_selector_totals["mutation_attempt_count"],
        "fill_selector_watch_count": fill_selector_totals["watch_count"],
        "fill_selector_no_accepted_candidate_count": fill_selector_totals["no_accepted_candidate_count"],
        "supervisor_soft_quota_per_24_ticks": config.supervisor_soft_quota_per_24_ticks,
        "supervisor_max_orders_per_tick": config.supervisor_max_orders_per_tick,
        "supervisor_metrics": _finalize_supervisor_totals(supervisor_totals, supervisor_quota_state),
        "bid_budget_enabled": config.bid_budget_enabled,
        "bid_budget_per_agent": config.bid_budget_per_agent,
        "bid_budget_scope": config.bid_budget_scope,
        "bid_budget_history_ticks": config.bid_budget_history_ticks,
        "bid_budget_exhausted_count": bid_budget_exhausted_count,
        "bid_budget_final_state_by_agent": bid_budget_states if config.bid_budget_enabled else {},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir
