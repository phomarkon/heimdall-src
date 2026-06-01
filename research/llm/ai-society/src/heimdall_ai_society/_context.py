from __future__ import annotations

from heimdall_contracts import Persona
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.memory import MemoryItem
from heimdall_ai_society.personas import build_personas
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tools import AgentToolExecutor
from heimdall_ai_society.rag import RAGRetriever
from packages.simulator import ScenarioAssetStateStore

from heimdall_ai_society._trace_helpers import _with_provenance, _with_provenance_all
from heimdall_ai_society._prompts import _opportunity_hint, _required_simulation_tool
from heimdall_ai_society._candidates import _seed_candidate_tools, _rank_seeded_candidates
from heimdall_ai_society._prompts import _compact_tool_records_for_prompt


def _filter_memory_bank(items: list[MemoryItem], scope_filter: str) -> list[MemoryItem]:
    if scope_filter == "all" or scope_filter == "synthesis":
        return items
    return [item for item in items if getattr(item, "scope", None) == scope_filter]


def _forecast_backend_by_agent(
    personas: list[Persona],
    *,
    fallback_backend: str,
    routing_mode: str = "persona",
) -> tuple[dict[str, str], list[str]]:
    if routing_mode == "run_level":
        return {persona.agent_id: fallback_backend for persona in personas}, []

    mapping = {
        "F0": "f0",
        "F1": "f1_lgbm",
        "F2": "f2_blr",
        "F3": "f3_ensemble",
        "F3_ENSEMBLE": "f3_ensemble",
        "F3_LITE": "f3_lite",
        "F4": "f4_mc_dropout",
        "F5": "f5",
        "F6": "f6",
        "F7": "f7",
        "F7_OPTUNA": "f7_optuna",
        "F8": "f8",
        "F8B": "f8b",
        "F8C": "f8c",
        "F8D": "f8d",
        "F8E": "f8e",
        "F9": "f9",
        "F10": "f10",
        "F11": "f11",
        "F13": "f13",
    }
    backends: dict[str, str] = {}
    warnings: list[str] = []
    for persona in personas:
        forecaster_id = str(persona.forecaster_id or "").upper()
        backend = mapping.get(forecaster_id)
        if backend is None:
            backend = fallback_backend
            warnings.append(
                f"{persona.agent_id} forecaster_id={persona.forecaster_id!r} fell back to run-level backend {fallback_backend}"
            )
        backends[persona.agent_id] = backend
    return backends, warnings


def _forecast_diversity_context(
    *,
    personas: list[Persona],
    persona_ticks: list[TickContext],
    forecast_backend_by_agent: dict[str, str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    by_backend: dict[str, dict[str, object]] = {}
    side_counts = {"up": 0, "down": 0, "mixed": 0}
    up_mids: list[float] = []
    down_mids: list[float] = []
    for persona, tick in zip(personas, persona_ticks, strict=True):
        backend = forecast_backend_by_agent.get(persona.agent_id, "unknown")
        up_lower, up_upper = tick.forecast.interval_for_side("up")
        down_lower, down_upper = tick.forecast.interval_for_side("down")
        up_mid = (float(up_lower) + float(up_upper)) / 2.0
        down_mid = (float(down_lower) + float(down_upper)) / 2.0
        up_mids.append(up_mid)
        down_mids.append(down_mid)
        side_signal = _forecast_side_signal(
            market_price=float(tick.market_price_eur_mwh),
            up_mid=up_mid,
            down_mid=down_mid,
        )
        side_counts[side_signal] += 1
        rows.append(
            {
                "agent_id": persona.agent_id,
                "archetype": persona.archetype.value,
                "backend": backend,
                "mfrr_up_interval_eur_mwh": [round(float(up_lower), 6), round(float(up_upper), 6)],
                "mfrr_down_interval_eur_mwh": [round(float(down_lower), 6), round(float(down_upper), 6)],
                "side_signal": side_signal,
                "forecast_hash": tick.forecast.result_hash,
            }
        )
        backend_row = by_backend.setdefault(
            backend,
            {
                "agent_count": 0,
                "side_counts": {"up": 0, "down": 0, "mixed": 0},
                "up_lower_values": [],
                "up_upper_values": [],
                "down_lower_values": [],
                "down_upper_values": [],
            },
        )
        backend_row["agent_count"] = int(backend_row["agent_count"]) + 1
        backend_side_counts = backend_row["side_counts"]
        if isinstance(backend_side_counts, dict):
            backend_side_counts[side_signal] = int(backend_side_counts.get(side_signal, 0)) + 1
        for key, value in [
            ("up_lower_values", float(up_lower)),
            ("up_upper_values", float(up_upper)),
            ("down_lower_values", float(down_lower)),
            ("down_upper_values", float(down_upper)),
        ]:
            values = backend_row[key]
            if isinstance(values, list):
                values.append(value)

    backend_summary: dict[str, dict[str, object]] = {}
    for backend, values in by_backend.items():
        up_lowers = values["up_lower_values"]
        up_uppers = values["up_upper_values"]
        down_lowers = values["down_lower_values"]
        down_uppers = values["down_upper_values"]
        backend_summary[backend] = {
            "agent_count": values["agent_count"],
            "side_counts": values["side_counts"],
            "mfrr_up_interval_range_eur_mwh": [
                round(min(up_lowers), 6),  # type: ignore[arg-type]
                round(max(up_uppers), 6),  # type: ignore[arg-type]
            ],
            "mfrr_down_interval_range_eur_mwh": [
                round(min(down_lowers), 6),  # type: ignore[arg-type]
                round(max(down_uppers), 6),  # type: ignore[arg-type]
            ],
        }
    return {
        "authority": "derived_non_leaking",
        "backend_by_agent": forecast_backend_by_agent,
        "backend_counts": {backend: int(values["agent_count"]) for backend, values in by_backend.items()},
        "interval_ranges_by_forecaster": backend_summary,
        "agent_forecasts": rows,
        "side_consensus": {
            "counts": side_counts,
            "majority_side": _majority_side(side_counts),
            "majority_fraction": round(max(side_counts.values()) / max(1, len(rows)), 6),
        },
        "interval_disagreement": {
            "up_mid_spread_eur_mwh": round(max(up_mids) - min(up_mids), 6) if up_mids else 0.0,
            "down_mid_spread_eur_mwh": round(max(down_mids) - min(down_mids), 6) if down_mids else 0.0,
            "max_mid_spread_eur_mwh": round(max(max(up_mids) - min(up_mids), max(down_mids) - min(down_mids)), 6) if up_mids and down_mids else 0.0,
        },
    }


def _forecast_diversity_for_candidate(context: dict[str, object] | None, candidate_side: str | None) -> dict[str, object]:
    if context is None:
        return {
            "authority": "derived_non_leaking",
            "candidate_side": candidate_side,
            "candidate_side_support": "unavailable",
        }
    summary = dict(context)
    summary["candidate_side"] = candidate_side
    summary["candidate_side_support"] = _candidate_side_support(context, candidate_side)
    return summary


def _forecast_side_signal(*, market_price: float, up_mid: float, down_mid: float) -> str:
    up_edge = up_mid - market_price
    down_edge = market_price - down_mid
    if up_edge >= down_edge + 2.0 and up_edge >= 5.0:
        return "up"
    if down_edge >= up_edge + 2.0 and down_edge >= 5.0:
        return "down"
    return "mixed"


def _majority_side(side_counts: dict[str, int]) -> str:
    ordered = sorted(side_counts.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] == 0:
        return "mixed"
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "mixed"
    return ordered[0][0]


def _candidate_side_support(context: dict[str, object], candidate_side: str | None) -> str:
    if candidate_side not in {"up", "down"}:
        return "none"
    consensus = context.get("side_consensus", {})
    counts = consensus.get("counts", {}) if isinstance(consensus, dict) else {}
    if not isinstance(counts, dict):
        return "unavailable"
    candidate_count = int(counts.get(candidate_side, 0) or 0)
    opposite = "down" if candidate_side == "up" else "up"
    opposite_count = int(counts.get(opposite, 0) or 0)
    mixed_count = int(counts.get("mixed", 0) or 0)
    if candidate_count > opposite_count and candidate_count >= mixed_count:
        return "majority_supported"
    if candidate_count == opposite_count or mixed_count >= max(candidate_count, opposite_count):
        return "mixed"
    return "minority_supported"


def _update_side_diagnostics(
    diagnostics: dict[str, object],
    *,
    persona: Persona,
    forecast_backend: str,
    tool_calls: list[ToolCallRecord],
) -> None:
    backend_counts = diagnostics["agent_forecast_backend_counts"]  # type: ignore[assignment]
    backend_counts[forecast_backend] = int(backend_counts.get(forecast_backend, 0)) + 1
    for record in tool_calls:
        if not record.name.startswith("simulate"):
            continue
        side = str(record.arguments.get("side") or "")
        if side not in {"up", "down"}:
            continue
        key = f"{side}:{persona.archetype.value}:{forecast_backend}"
        bucket_name = (
            "accepted_simulator_candidate_counts"
            if record.result.get("accepted") is True
            else "rejected_simulator_candidate_counts"
            if record.result.get("accepted") is False
            else None
        )
        if bucket_name is None:
            continue
        bucket = diagnostics[bucket_name]  # type: ignore[assignment]
        bucket[key] = int(bucket.get(key, 0)) + 1


def _auto_rag_query(persona: Persona, tick: TickContext) -> str:
    """Context-aware query for the seeded retrieval (zone, archetype, time of day)."""
    ts = tick.timestamp
    zone = getattr(tick.forecast, "zone", "DK1")
    arche = persona.archetype.value
    return (
        f"{zone} {arche} mFRR balancing near {ts.strftime('%H:%M')} UTC on {ts.strftime('%A')}: "
        "likely activation side (up vs down), settlement-price regime, and bid sizing/participation "
        "lesson from earlier days and past runs"
    )


def _seed_retrieval(
    persona: Persona,
    tick: TickContext,
    executor: AgentToolExecutor,
    *,
    top_k: int,
) -> list[ToolCallRecord]:
    """Seed one retrieve_knowledge call per agent/tick so the RAG treatment is always
    applied (and auditable as a runner_seeded tool call). The cutoff is forced to the
    tick inside the executor; the model may issue further targeted queries itself.
    """
    record = executor.execute("retrieve_knowledge", {"query": _auto_rag_query(persona, tick), "k": top_k})
    return _with_provenance_all([record], "runner_seeded")


def _market_intelligence_digest(
    *,
    tick: TickContext,
    personas: list[Persona],
    data_tools: object | None,
    simulator_tool: object | None,
    objective: str,
    ablation_strategy: str,
    tool_cache: dict[tuple[str, str], ToolCallRecord],
    safety_toolset: str = "full",
    preprobe_mode: str = "full",
    asset_simulator_mode: str = "proxy",
    asset_proxy_style: str = "market",
    asset_state_store: ScenarioAssetStateStore | None = None,
) -> dict[str, object]:
    asset_state_store = asset_state_store or ScenarioAssetStateStore.empty()
    anchor = personas[0] if personas else build_personas(1)[0]
    executor = AgentToolExecutor(
        persona=anchor,
        forecast=tick.forecast,
        data_tools=data_tools,
        simulator_tool=simulator_tool,  # type: ignore[arg-type]
        asset_simulator_mode=asset_simulator_mode,
        asset_proxy_style=asset_proxy_style,
        asset_state_store=asset_state_store,
        tool_cache=tool_cache,
    )
    context_records = []
    if preprobe_mode != "none":
        context_records = _with_provenance_all(
            [
                executor.execute("get_market_regime_context", {"hours": 24, "zone": tick.forecast.zone}),
                executor.execute("get_grid_constraints", {"hours": 24, "zone": tick.forecast.zone}),
                executor.execute("get_border_pressure", {"hours": 24, "zone": tick.forecast.zone, "counterparty": ""}),
                executor.execute("get_outage_impact", {"hours": 24, "zone": tick.forecast.zone}),
                executor.execute("get_uncertainty_digest", {}),
            ],
            "runner_diagnostic",
        )
    if preprobe_mode == "full" and safety_toolset != "context_only":
        context_records.append(_with_provenance(executor.execute("get_decision_trace_summary", {}), "runner_diagnostic"))
    candidate_records: list[ToolCallRecord] = []
    if preprobe_mode == "full" and safety_toolset != "context_only":
        for persona in personas:
            if persona.archetype.value not in {"p2h", "ev", "wind", "generator", "retailer", "renewables"}:
                continue
            persona_executor = AgentToolExecutor(
                persona=persona,
                forecast=tick.forecast,
                data_tools=data_tools,
                simulator_tool=simulator_tool,  # type: ignore[arg-type]
                asset_simulator_mode=asset_simulator_mode,
                asset_proxy_style=asset_proxy_style,
                asset_state_store=asset_state_store,
                tool_cache=tool_cache,
            )
            candidate_records.extend(
                _with_provenance_all(
                    _seed_candidate_tools(
                        objective=objective,
                        ablation_strategy=ablation_strategy,
                        persona=persona,
                        tick=tick,
                        executor=persona_executor,
                    ),
                    "runner_diagnostic",
                )
            )
            if len([record for record in candidate_records if record.name.startswith("simulate")]) >= 12:
                break
    reason_counts: dict[str, int] = {}
    accepted = 0
    for record in candidate_records:
        if not record.name.startswith("simulate"):
            continue
        if record.result.get("accepted") is True:
            accepted += 1
        for reason in record.result.get("reason_codes", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "kind": "shared_market_intelligence_digest",
        "authority": "derived_non_leaking",
        "timestamp": tick.timestamp.isoformat(),
        "zone": tick.forecast.zone,
        "context": _compact_tool_records_for_prompt(context_records),
        "candidate_summary": {
            "simulated_candidate_count": sum(1 for record in candidate_records if record.name.startswith("simulate")),
            "accepted_candidate_count": accepted,
            "rejection_reason_counts": reason_counts,
            "sample": _compact_tool_records_for_prompt(candidate_records[:10]),
        },
        "market_mechanics_notes": {
            "jurisdiction": "nordic_mfrr_offline_harness",
            "settlement_period": "15_minutes",
            "guardrail": (
                "pre-submit simulator/verifier hidden; final bids are shadow-scored after submission"
                if safety_toolset == "context_only"
                else "final bids must exactly match simulator-accepted candidates"
            ),
            "pnl_caveat": "reported profit is backtested realized profit under evaluator clearing assumptions, not live-market money",
            "future_data_not_in_v1": ["jao_cnec", "reserve_saturation", "intraday_order_book", "participant_behavior", "uplift_or_penalties"],
        },
    }


def _enrich_intelligence_decision(
    decision: LLMBidDecision,
    tick: TickContext,
    tool_calls: list[ToolCallRecord],
) -> LLMBidDecision:
    from heimdall_ai_society._deterministic import _watch_score_from_records
    reasons = list(decision.watch_reasons)
    accepted = any(record.name.startswith("simulate") and record.result.get("accepted") is True for record in tool_calls)
    rejected = sum(1 for record in tool_calls if record.name.startswith("simulate") and record.result.get("accepted") is False)
    watch_score = _watch_score_from_records(tool_calls)
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    width = max(up_upper - up_lower, down_upper - down_lower)
    edge = max(abs(up_lower - tick.market_price_eur_mwh), abs(tick.market_price_eur_mwh - down_upper))
    if accepted:
        reasons.append("accepted_bid_available")
    if rejected >= 2:
        reasons.append("verifier_rejection_cluster")
    if width >= 50.0:
        reasons.append("forecast_uncertainty")
    if edge >= 25.0:
        reasons.append("price_volatility")
    if watch_score >= 0.5:
        reasons.append("activation_risk")
    if decision.action == "bid" or accepted:
        watch_label = "must_watch"
    elif decision.action == "watch" or watch_score >= 0.35 or edge >= 15.0 or width >= 35.0:
        watch_label = "watch"
    else:
        watch_label = decision.watch_label
    risk_label = decision.risk_label
    if rejected >= 2 or width >= 80.0:
        risk_label = "high"
    elif rejected >= 1 or width >= 35.0 or watch_score >= 0.5:
        risk_label = "medium"
    uncertainty_label = "high" if width >= 80.0 else "medium" if width >= 35.0 else decision.uncertainty_label
    opportunity_label = "actionable" if accepted or decision.action == "bid" else "weak" if edge >= 15.0 or watch_score >= 0.35 else decision.opportunity_label
    return decision.model_copy(
        update={
            "watch_label": watch_label,
            "risk_label": risk_label,
            "uncertainty_label": uncertainty_label,
            "opportunity_label": opportunity_label,
            "watch_reasons": _merge_watch_reasons(reasons, []),
        }
    )


def _merge_watch_reasons(current: list[str], extra: list[str]) -> list[str]:
    allowed = {
        "activation_risk",
        "price_volatility",
        "forecast_uncertainty",
        "accepted_bid_available",
        "verifier_rejection_cluster",
        "cross_agent_disagreement",
    }
    return [reason for reason in dict.fromkeys([*current, *extra]) if reason in allowed]


def _communication_context(
    *,
    strategy: str,
    tick: TickContext,
    personas: list[Persona],
    peer_summaries: list[dict[str, object]],
    shared_digest: dict[str, object] | None = None,
) -> dict[str, object]:
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    digest: dict[str, object] = {
        "strategy": strategy,
        "market_digest": {
            "timestamp": tick.timestamp.isoformat(),
            "zone": tick.forecast.zone,
            "last_price_eur_mwh": tick.market_price_eur_mwh,
            "mfrr_up_interval_eur_mwh": [up_lower, up_upper],
            "mfrr_down_interval_eur_mwh": [down_lower, down_upper],
            "opportunity_hint": _opportunity_hint(
                price=tick.market_price_eur_mwh,
                up_lower=up_lower,
                up_upper=up_upper,
                down_lower=down_lower,
                down_upper=down_upper,
            ),
        },
        "roster": [
            {
                "agent_id": persona.agent_id,
                "archetype": persona.archetype.value,
                "risk_attitude": persona.risk_attitude.value,
            }
            for persona in personas
        ],
    }
    if shared_digest is not None:
        digest["shared_market_intelligence"] = shared_digest
    if strategy in {"comm_peer_signal", "comm_retry_council"}:
        digest["peer_summaries_so_far"] = peer_summaries[-4:]
    if strategy == "comm_info_then_action":
        digest["expert_summaries"] = peer_summaries
    return digest


def _peer_summary(persona: Persona, decision: LLMBidDecision, tool_calls: list[ToolCallRecord]) -> dict[str, object]:
    accepted = [
        {
            "tool": record.name,
            "candidate": record.arguments,
            "expected_profit_eur": record.result.get("rough_expected_profit_eur"),
            "worst_case_profit_eur": record.result.get("worst_case_profit_eur"),
            "risk_flags": record.result.get("risk_flags", [])[:3],
        }
        for record in tool_calls
        if record.name.startswith("simulate") and record.result.get("accepted") is True
    ]
    rejected_reasons: dict[str, int] = {}
    for record in tool_calls:
        if not record.name.startswith("simulate"):
            continue
        for reason in record.result.get("reason_codes", []):
            rejected_reasons[str(reason)] = rejected_reasons.get(str(reason), 0) + 1
    return {
        "agent_id": persona.agent_id,
        "archetype": persona.archetype.value,
        "action": decision.action,
        "watch_label": decision.watch_label,
        "risk_label": decision.risk_label,
        "uncertainty_label": decision.uncertainty_label,
        "opportunity_label": decision.opportunity_label,
        "watch_reasons": decision.watch_reasons,
        "priority_label": decision.priority_label,
        "priority_score": decision.priority_score,
        "operator_action": decision.operator_action,
        "priority_reason": decision.priority_reason,
        "side": decision.side,
        "quantity_mwh": decision.quantity_mwh,
        "limit_price_eur_mwh": decision.limit_price_eur_mwh,
        "confidence": decision.confidence,
        "accepted_candidate_count": len(accepted),
        "accepted_candidates": accepted[:2],
        "rejected_reason_counts": rejected_reasons,
        "rationale": decision.rationale[:160],
    }
