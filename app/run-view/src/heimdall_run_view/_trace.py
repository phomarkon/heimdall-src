"""Agent trace construction, tool formatting, and decision helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from heimdall_run_view._catalog import (
    FALLBACK_ARCHETYPES,
    FALLBACK_RISKS,
    FALLBACK_SOPHISTICATION,
    FOCAL_AGENT_ID,
)
from heimdall_run_view._utils import (
    RunContext,
    _float,
    _int_like,
    _iso_z,
    _optional_float,
)

# ---------------------------------------------------------------------------
# Frontend archetype normalisation
# ---------------------------------------------------------------------------


def _frontend_archetype(value: str) -> str:
    aliases = {
        "renewables": "wind",
        "grid_constraint_analyst": "grid-info",
        "outage_impact_scorer": "outage-info",
        "limit_price_specialist": "price-info",
        "candidate_sizing_specialist": "sizing-info",
        "uncertainty_auditor": "uncertainty-info",
        "decision_auditor": "decision-info",
        "trading_risk_monitor": "risk-info",
    }
    if value in aliases:
        return aliases[value]
    if value not in set(FALLBACK_ARCHETYPES):
        return "arbitrageur"
    return value


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


def _decision_action(row: dict[str, Any]) -> str:
    return str((row.get("decision") or {}).get("action") or "abstain")


def _side_to_direction(side: Any) -> str:
    return "sell" if side == "up" else "buy"


def _stage_failed(row: dict[str, Any]) -> str | None:
    reasons = ",".join(row.get("verifier_reason_codes") or []).lower()
    if any(token in reasons for token in ["conformal", "profit", "coverage"]):
        return "conformal"
    if reasons or row.get("verifier_accepted") is False:
        return "physical"
    return None


def _retry_suggestion(row: dict[str, Any]) -> str:
    reasons = ", ".join(row.get("verifier_reason_codes") or [])
    if reasons:
        return f"Adjust the bid and retry; verifier reason codes: {reasons}."
    return "Adjust quantity or price and retry against the simulator envelope."


def _worst_case_profit(row: dict[str, Any]) -> float | None:
    for call in row.get("tool_calls") or []:
        result = call.get("result") or {}
        value = result.get("worst_case_profit_eur") or result.get("rough_worst_case_profit_eur")
        if isinstance(value, int | float):
            return float(value)
    return None


def _realized_outcome(
    row: dict[str, Any],
    eval_row: dict[str, Any],
    market: dict[str, Any],
    quantity: float,
    direction: str,
    accepted: bool,
) -> dict[str, Any]:
    fill = _float(eval_row.get("cleared_mwh"), quantity if accepted else 0.0)
    price = market["mfrr_price_eur_per_mwh"]
    limit = _float((row.get("decision") or {}).get("limit_price_eur_mwh"), price)
    pnl = _float(
        eval_row.get("realized_profit_eur"),
        fill * (price - limit) if direction == "sell" else fill * (limit - price),
    )
    return {
        "fill_mw": fill,
        "realized_price_eur_per_mwh": price,
        "pnl_eur": round(pnl, 2),
    }


def _belief(row: dict[str, Any], current: dict[str, Any]) -> str:
    if row.get("llm_id") == "unavailable":
        return "Peer placeholder; no decision trace is available for this replay."
    if _decision_action(current) == "bid":
        return "Replay trace includes a concrete mFRR bid decision for this interval."
    return "Replay trace records an abstain/watch decision for this interval."


# ---------------------------------------------------------------------------
# Info digest
# ---------------------------------------------------------------------------


def _info_digest(row: dict[str, Any]) -> dict[str, Any] | None:
    from heimdall_run_view._priority import _first_number

    archetype = _frontend_archetype(str(row.get("archetype") or ""))
    if not archetype.endswith("-info"):
        return None
    decision = row.get("decision") or {}
    signals: list[dict[str, Any]] = []
    signal_labels: set[str] = set()
    watch_score = 0.0
    direction_hint = None
    for call in row.get("tool_calls") or []:
        result = call.get("result") or {}
        value = _first_number(result, "watch_score")
        if value is not None:
            watch_score = max(watch_score, value)
        direction_hint = direction_hint or result.get("direction_hint")
        for key, label in (
            ("recent_up_spread_eur_mwh", "Up spread"),
            ("mean_abs_imbalance_move_eur_mwh", "Imbalance move"),
            ("side_edge_gap_eur_mwh", "Side gap"),
            ("max_abs_price_edge_eur_mwh", "Price edge"),
            ("latest_flow_mw", "Latest flow"),
            ("flow_swing_mw", "Flow swing"),
        ):
            candidate = (result.get("signals") or {}).get(key)
            if (
                isinstance(candidate, int | float)
                and label not in signal_labels
                and len(signals) < 4
            ):
                signals.append({"label": label, "value": round(float(candidate), 2)})
                signal_labels.add(label)
    confidence = _float(decision.get("confidence"), 0.0)
    importance = max(watch_score, confidence)
    return {
        "finding": str(decision.get("rationale") or "No current finding recorded."),
        "confidence": round(confidence, 3),
        "importance": round(importance, 3),
        "risk_label": decision.get("risk_label"),
        "uncertainty_label": decision.get("uncertainty_label"),
        "opportunity_label": decision.get("opportunity_label"),
        "watch_reasons": decision.get("watch_reasons") or [],
        "direction_hint": direction_hint,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Tool call formatting
# ---------------------------------------------------------------------------


def _tool_calls(
    row: dict[str, Any], step: int, market: dict[str, Any], interval: list[Any], accepted: bool
) -> list[dict[str, Any]]:
    raw_calls = row.get("tool_calls") or []
    calls = []
    for index, call in enumerate(raw_calls):
        result = call.get("result") or {}
        name = str(call.get("name") or f"tool-{index}")
        calls.append(
            {
                "id": f"{name}-{step}-{index}",
                "kind": _tool_kind(name),
                "label": _tool_label(name),
                "status": "success" if call.get("ok", True) else "error",
                "latency_ms": 40 + index * 28,
                "summary": _tool_summary(name, call, result),
                "provenance": str(call.get("provenance") or "unknown"),
            }
        )
    if calls:
        return calls
    low = _float(interval[0] if len(interval) > 0 else None, market["mfrr_price_eur_per_mwh"])
    high = _float(interval[1] if len(interval) > 1 else None, market["mfrr_price_eur_per_mwh"])
    return [
        {
            "id": f"context-{step}",
            "kind": "forecast",
            "label": "Trace context read",
            "status": "success",
            "latency_ms": 32,
            "summary": f"Loaded replay context and forecast interval {low:.1f} to {high:.1f} EUR/MWh.",
            "provenance": "unknown",
        },
        {
            "id": f"simulate-{step}",
            "kind": "simulate",
            "label": "Simulator replay",
            "status": "success" if accepted else "error",
            "latency_ms": 52,
            "summary": "Replay accepted the proposed bid."
            if accepted
            else "Replay rejected or skipped the proposed bid.",
            "provenance": "unknown",
        },
    ]


def _tool_kind(name: str) -> str:
    if "simulate" in name:
        return "simulate"
    if "feasibility" in name or "forecast" in name:
        return "forecast"
    if "verifier" in name:
        return "verifier"
    return "simulate" if "bid" in name else "news"


def _tool_label(name: str) -> str:
    labels = {
        "run_forecaster": "Tool called: run_forecaster",
        "get_activation_context": "Tool called: get_activation_context",
        "get_opportunity_context": "Tool called: get_opportunity_context",
        "candidate_menu": "Tool called: candidate_menu",
        "rank_candidate_set": "Tool called: rank_candidate_set",
        "society_communication_context": "Tool called: society_communication_context",
        "selected_candidate_diagnostics": "Tool called: selected_candidate_diagnostics",
        "get_bid_feasibility": "Forecaster-agent: bid feasibility",
        "simulate_bid": "Risk-agent: simulator replay",
        "propose_action": "Quoter-agent: proposed action",
    }
    return labels.get(name, f"Tool called: {name}")


def _tool_summary(name: str, call: dict[str, Any], result: dict[str, Any]) -> str:
    arguments = call.get("arguments") or {}
    if name == "run_forecaster":
        up = result.get("mfrr_up_interval_eur_mwh") or []
        down = result.get("mfrr_down_interval_eur_mwh") or []
        return (
            f"Requested forecast for {result.get('zone', 'zone')} at {result.get('delivery_timestamp', 'this tick')}. "
            f"Returned up interval {_interval(up)} and down interval {_interval(down)}."
        )
    if name in {"get_activation_context", "get_opportunity_context"}:
        signals = result.get("signals") or {}
        return (
            f"Asked for {result.get('lookback_hours', arguments.get('hours', 24))}h market context. "
            f"Direction hint: {result.get('direction_hint', 'n/a')}; watch score {_fmt(result.get('watch_score'))}; "
            f"recent up spread {_fmt(signals.get('recent_up_spread_eur_mwh'))} EUR/MWh."
        )
    if name == "candidate_menu":
        candidates = result.get("candidates") or []
        best = candidates[0] if candidates else {}
        return (
            f"Requested candidate bid menu using strategy {arguments.get('strategy', 'default')}. "
            f"Returned {len(candidates)} candidates; first is {best.get('side', 'n/a')} "
            f"{_fmt(best.get('quantity_mwh'))} MWh at {_fmt(best.get('limit_price_eur_mwh'))} EUR/MWh."
        )
    if name == "rank_candidate_set":
        ranking = result.get("ranking") or []
        best = ranking[0] if ranking else {}
        best_args = best.get("arguments") or {}
        return (
            f"Ranked {len(ranking)} candidate bids. Top candidate: {best_args.get('side', 'n/a')} "
            f"{_fmt(best_args.get('quantity_mwh'))} MWh at {_fmt(best_args.get('limit_price_eur_mwh'))} EUR/MWh, "
            f"score {_fmt(best.get('score'))}, worst-case profit {_fmt(best.get('worst_case_profit_eur'))} EUR."
        )
    if name == "society_communication_context":
        context = result.get("context") or {}
        digest = context.get("market_digest") or {}
        roster = context.get("roster") or []
        hint = digest.get("opportunity_hint") or {}
        return (
            f"Loaded society broadcast context for {len(roster)} agents. "
            f"Market price {_fmt(digest.get('last_price_eur_mwh'))} EUR/MWh; suggested "
            f"{hint.get('candidate_bid_side', 'n/a')} probe at {_fmt(hint.get('suggested_limit_price_eur_mwh'))} EUR/MWh."
        )
    if name == "selected_candidate_diagnostics":
        candidate = result.get("candidate") or {}
        sim = result.get("simulator_result") or {}
        return (
            f"Checked selected candidate against {result.get('matched_tool', 'simulator')}: "
            f"{candidate.get('side', 'n/a')} {_fmt(candidate.get('quantity_mwh'))} MWh at "
            f"{_fmt(candidate.get('limit_price_eur_mwh'))} EUR/MWh. "
            f"Accepted: {sim.get('accepted', 'n/a')}; worst-case profit {_fmt(sim.get('worst_case_profit_eur'))} EUR."
        )
    if name.startswith("simulate_") or name == "simulate_bid":
        accepted = result.get("accepted")
        reasons = ", ".join(result.get("reason_codes") or [])
        return (
            f"Simulated requested {arguments.get('side', 'bid')} bid for {_fmt(arguments.get('quantity_mwh'))} MWh "
            f"at {_fmt(arguments.get('limit_price_eur_mwh'))} EUR/MWh. "
            f"{'Accepted' if accepted else 'Rejected'}"
            f"{f' ({reasons})' if reasons else ''}; worst-case profit {_fmt(result.get('worst_case_profit_eur'))} EUR."
        )
    if "feasibility" in name:
        guidance = result.get("guidance") or "guidance unavailable"
        score = result.get("score")
        return (
            f"Checked feasibility for {arguments.get('side', 'bid')} {_fmt(arguments.get('quantity_mwh'))} MWh "
            f"at {_fmt(arguments.get('limit_price_eur_mwh'))} EUR/MWh. Guidance: {guidance}; "
            f"score {_fmt(score)}; accepted: {result.get('accepted', 'n/a')}."
        )
    if name == "propose_action":
        decision = result.get("decision") or call.get("arguments") or {}
        return str(decision.get("rationale") or "Structured action was recorded in the trace.")
    return str(result or call.get("error") or "Tool call recorded.")


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    if value is None:
        return "n/a"
    return str(value)


def _interval(values: list[Any]) -> str:
    if len(values) < 2:
        return "n/a"
    return f"{_fmt(values[0])}-{_fmt(values[1])} EUR/MWh"


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


def _persona(agent_id: str, row: dict[str, Any], is_focal: bool) -> dict[str, Any]:
    raw_archetype = row.get("archetype") or ("p2h" if is_focal else "wind")
    archetype = _frontend_archetype(str(raw_archetype))
    return {
        "agent_id": agent_id,
        "display_name": f"{archetype.upper()} BRP {agent_id[-3:]}"
        if not is_focal
        else f"{archetype.upper()} focal {agent_id[-3:]}",
        "archetype": archetype,
        "risk_attitude": FALLBACK_RISKS[sum(ord(char) for char in agent_id) % len(FALLBACK_RISKS)],
        "sophistication": FALLBACK_SOPHISTICATION[
            sum(ord(char) for char in agent_id) % len(FALLBACK_SOPHISTICATION)
        ],
        "info_latency_min": 0 if is_focal else 15,
        "capacity_mw": 50 if is_focal else 10 + (sum(ord(char) for char in agent_id) % 40),
        "storage_mwh": 100 if archetype in {"p2h", "ev"} else None,
        "llm_family": str(row.get("llm_id") or "unavailable"),
        "forecaster": str(row.get("forecaster_id") or "unavailable"),
    }


# ---------------------------------------------------------------------------
# Agent trace (full per-step trace for the snapshot)
# ---------------------------------------------------------------------------


def _agent_trace(
    run_id: str,
    step: int,
    timestamp: str,
    row: dict[str, Any],
    market: dict[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    from heimdall_run_view._snapshot import _agent_cumulative_pnl

    agent_id = str(row.get("agent_id") or FOCAL_AGENT_ID)
    eval_row = context.bid_rows_by_step_agent.get((step, agent_id), {})
    decision = row.get("decision") or {}
    action = _decision_action(row)
    accepted = eval_row.get("verifier_accepted", row.get("verifier_accepted"))
    is_bid = action == "bid"
    direction = _side_to_direction(decision.get("side"))
    quantity = _float(decision.get("quantity_mwh"), 0.0)
    limit_price = _float(decision.get("limit_price_eur_mwh"), market["mfrr_price_eur_per_mwh"])
    interval = row.get("forecast_interval_eur_mwh") or [
        market["mfrr_price_eur_per_mwh"],
        market["mfrr_price_eur_per_mwh"],
    ]
    worst_case = _worst_case_profit(row)
    status = str(eval_row.get("status") or action)
    accepted_bool = bool(accepted) if accepted is not None else status in {"watch", "abstain"}
    stage_failed = None if accepted_bool else _stage_failed(row)
    return {
        "run_id": run_id,
        "step": step,
        "timestamp": timestamp,
        "agent_id": agent_id,
        "persona": _persona(agent_id, row, True),
        "state": {
            "soc_mwh": 50.0
            if _frontend_archetype(str(row.get("archetype") or "p2h")) in {"p2h", "ev"}
            else None,
            "exposure_mw": quantity if direction == "sell" else -quantity,
            "cash_eur": 50_000 + _agent_cumulative_pnl(context, agent_id, step),
        },
        "reasoning": str(
            row.get("rationale")
            or decision.get("rationale")
            or "Trace did not include free-form reasoning for this tick."
        ),
        "tool_calls": _tool_calls(row, step, market, interval, accepted_bool),
        "tool_call_provenance_counts": row.get("tool_call_provenance_counts") or {},
        "proposed_action": {
            "market": "mFRR",
            "direction": direction,
            "quantity_mw": quantity,
            "price_eur_per_mwh": limit_price,
            "delivery_quarter": timestamp,
        },
        "verifier_verdict": {
            "accepted": accepted_bool,
            "stage_failed": stage_failed,
            "physical_violation": {"reason": ",".join(row.get("verifier_reason_codes") or [])}
            if stage_failed == "physical"
            else None,
            "worst_case_profit_eur": worst_case,
            "threshold_eur": 0.0,
            "retry_suggestion": None if accepted_bool else _retry_suggestion(row),
            "conformal_interval": {
                "horizon_minutes": 15,
                "quantile_low": _float(
                    interval[0] if len(interval) > 0 else None, market["mfrr_price_eur_per_mwh"]
                ),
                "quantile_high": _float(
                    interval[1] if len(interval) > 1 else None, market["mfrr_price_eur_per_mwh"]
                ),
                "alpha": 0.1,
            },
        },
        "realized_outcome": _realized_outcome(
            row, eval_row, market, quantity, direction, accepted_bool
        )
        if is_bid
        else None,
        "info_digest": _info_digest(row),
    }


# ---------------------------------------------------------------------------
# Agent history (flat per-agent timeline)
# ---------------------------------------------------------------------------


def _agent_history_record(
    run_id: str,
    agent_id: str,
    row: dict[str, Any],
    index: int,
    context: RunContext,
) -> dict[str, Any]:
    step = _int_like(row.get("step"), index)
    timestamp = _history_timestamp(row, step)
    decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    interval = _history_interval(row.get("forecast_interval_eur_mwh"))
    accepted = row.get("verifier_accepted")
    eval_row = context.bid_rows_by_step_agent.get((step, agent_id), {})
    if eval_row.get("verifier_accepted") is not None:
        accepted = eval_row.get("verifier_accepted")
    return {
        "run_id": run_id,
        "step": step,
        "timestamp": timestamp,
        "observed_at": _iso_z(row.get("observed_at")) if row.get("observed_at") else None,
        "agent_id": agent_id,
        "zone": row.get("zone"),
        "archetype": _frontend_archetype(str(row.get("archetype") or "p2h")),
        "market_price_eur_mwh": _optional_float(row.get("market_price_eur_mwh")),
        "forecast_interval_eur_mwh": interval,
        "decision": decision,
        "rationale": str(row.get("rationale") or decision.get("rationale") or ""),
        "verifier": {
            "accepted": accepted if isinstance(accepted, bool) else None,
            "reason_codes": [str(code) for code in row.get("verifier_reason_codes") or []],
            "stage_failed": _stage_failed(row),
        },
        "realized_outcome": _history_realized_outcome(eval_row),
        "tool_calls": _history_tool_calls(row.get("tool_calls") or []),
        "tool_call_provenance_counts": row.get("tool_call_provenance_counts") or {},
    }


def _history_timestamp(row: dict[str, Any], step: int) -> str:
    value = row.get("timestamp") or row.get("utc_timestamp") or row.get("delivery_quarter")
    if value:
        return _iso_z(value)
    return _iso_z(datetime(2026, 4, 1, tzinfo=UTC) + timedelta(minutes=15 * step))


def _history_interval(value: Any) -> list[float | None] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return [_optional_float(value[0]), _optional_float(value[1])]


def _history_realized_outcome(eval_row: dict[str, Any]) -> dict[str, float] | None:
    if not eval_row:
        return None
    return {
        "fill_mw": _float(eval_row.get("cleared_mwh"), 0.0),
        "realized_price_eur_per_mwh": _float(eval_row.get("market_price_eur_mwh"), 0.0),
        "pnl_eur": round(_float(eval_row.get("realized_profit_eur"), 0.0), 2),
    }


def _history_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                "name": str(call.get("name") or f"tool-{index}"),
                "arguments": call.get("arguments") or {},
                "ok": call.get("ok") if isinstance(call.get("ok"), bool) else None,
                "result": call.get("result"),
                "error": call.get("error"),
                "provenance": str(call.get("provenance") or "unknown"),
            }
        )
    return calls
