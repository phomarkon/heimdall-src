"""Snapshot building and step-level health/coverage helpers."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from heimdall_run_view._catalog import FOCAL_AGENT_ID
from heimdall_run_view._trace import _decision_action, _frontend_archetype, _side_to_direction
from heimdall_run_view._utils import RunContext, _float, _iso_z, _parse_dt


def build_snapshot(
    *,
    run_id: str,
    step: int,
    total_steps: int,
    rows_by_step: dict[int, list[dict[str, Any]]],
    manifest: dict[str, Any],
    trace_sha256: str,
    context: RunContext,
    priority_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from heimdall_run_view._forecaster import _forecast_diagnostics
    from heimdall_run_view._graph import _nodes, _society_edges
    from heimdall_run_view._trace import _agent_trace

    safe_step = max(0, min(total_steps - 1, step))
    rows = rows_by_step.get(safe_step) or []
    focal = _select_focal(rows) or _nearest_row(rows_by_step, safe_step) or {}
    timestamp = _timestamp_for(focal, rows_by_step, safe_step)
    market = _market_tick(safe_step, timestamp, rows, focal, priority_signal)
    nodes = _nodes(rows, focal, safe_step, rows_by_step, context)
    agent_traces = {
        str(row.get("agent_id") or FOCAL_AGENT_ID): _agent_trace(
            run_id, safe_step, timestamp, row, market, context
        )
        for row in rows
    }
    selected_trace = agent_traces.get(str(focal.get("agent_id") or FOCAL_AGENT_ID)) or _agent_trace(
        run_id, safe_step, timestamp, focal, market, context
    )
    accepted_count, decision_count, cumulative_pnl = _health_inputs(
        rows_by_step, safe_step, context
    )
    acceptance_rate = accepted_count / decision_count if decision_count else 0.0
    tick_rows = context.bid_rows_by_step.get(safe_step, [])
    tick_pnl = sum(_float(row.get("realized_profit_eur"), 0.0) for row in tick_rows)
    status_counts = _status_counts(tick_rows)
    return {
        "run_id": run_id,
        "step": safe_step,
        "total_steps": total_steps,
        "nodes": nodes,
        "edges": _society_edges(safe_step, rows),
        "selected_trace": selected_trace,
        "agent_traces": agent_traces,
        "market": market,
        "forecast_diagnostics": _forecast_diagnostics(focal, market, selected_trace),
        "health": {
            "coverage": _coverage(rows_by_step, safe_step),
            "verifier_acceptance_rate": acceptance_rate,
            "cumulative_pnl_eur": cumulative_pnl,
            "gpu_utilization": 0.0,
            "wall_time_minutes": safe_step * 0.25,
            "tick_pnl_eur": round(tick_pnl, 2),
            "cleared_mwh": round(sum(_float(row.get("cleared_mwh"), 0.0) for row in tick_rows), 3),
            "filled_count": status_counts.get("filled", 0),
            "bid_count": len(tick_rows),
            "status_counts": status_counts,
        },
        "source": {
            "trace_sha256": trace_sha256,
            "manifest_schema_version": manifest.get("schema_version"),
        },
    }


# ---------------------------------------------------------------------------
# Row selection helpers
# ---------------------------------------------------------------------------


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row.get("step", index)), []).append(row)
    return grouped


def _select_focal(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return next((row for row in rows if str(row.get("archetype")) == "p2h"), rows[0])


def _nearest_row(rows_by_step: dict[int, list[dict[str, Any]]], step: int) -> dict[str, Any] | None:
    for distance in range(0, max(rows_by_step.keys(), default=0) + 1):
        for candidate in (step - distance, step + distance):
            rows = rows_by_step.get(candidate)
            if rows:
                return _select_focal(rows)
    return None


def _timestamp_for(
    row: dict[str, Any], rows_by_step: dict[int, list[dict[str, Any]]], step: int
) -> str:
    value = row.get("timestamp") or row.get("utc_timestamp")
    if value:
        return _iso_z(value)
    first = _nearest_row(rows_by_step, 0)
    if first and first.get("timestamp"):
        start = _parse_dt(first["timestamp"]) + timedelta(minutes=15 * step)
        return _iso_z(start)
    return _iso_z(datetime(2026, 4, 1, tzinfo=UTC) + timedelta(minutes=15 * step))


def _market_tick(
    step: int,
    timestamp: str,
    rows: list[dict[str, Any]],
    focal: dict[str, Any],
    priority_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    price = _float(focal.get("market_price_eur_mwh"), 0.0)
    if price == 0.0:
        price = 50.0 + math.sin(step / 3) * 8
    events: list[dict[str, str]] = []
    watch_level = _watch_level(rows)
    if watch_level == "must_watch":
        events.append(
            {"id": f"must-watch-{step}", "kind": "must_watch", "label": "Must-watch interval"}
        )
    elif watch_level == "watch":
        events.append({"id": f"watch-{step}", "kind": "watch", "label": "Watch interval"})
    for row in rows:
        if _decision_action(row) != "bid":
            continue
        accepted = row.get("verifier_accepted")
        if accepted is True:
            events.append(
                {
                    "id": f"accept-{step}-{row.get('agent_id', 'agent')}",
                    "kind": "accepted_bid",
                    "label": "Verifier accepted bid",
                }
            )
        elif accepted is False:
            events.append(
                {
                    "id": f"reject-{step}-{row.get('agent_id', 'agent')}",
                    "kind": "rejected_bid",
                    "label": "Verifier rejected bid",
                }
            )
    if price > 120:
        events.append({"id": f"spike-{step}", "kind": "price_spike", "label": "mFRR price spike"})
    return {
        "step": step,
        "timestamp": timestamp,
        "dk1_price_eur_per_mwh": price,
        "dk2_price_eur_per_mwh": price * 0.96,
        "mfrr_price_eur_per_mwh": price,
        "imbalance_mw": math.sin(step * 0.31) * 80.0,
        "gate_closure_minutes": 45 - ((step * 15) % 45),
        "events": events,
        "priority_signal": priority_signal
        or {
            "score": 0.0,
            "rank": None,
            "percentile": 0.0,
            "tier": "low",
            "label": "Low priority",
            "drivers": [],
            "risks": [],
        },
    }


def _watch_level(rows: list[dict[str, Any]]) -> str | None:
    labels = [str((row.get("decision") or {}).get("watch_label") or "") for row in rows]
    if "must_watch" in labels:
        return "must_watch"
    if "watch" in labels:
        return "watch"
    watch_actions = [row for row in rows if _decision_action(row) == "watch"]
    if not watch_actions:
        return None
    watch_share = len(watch_actions) / max(1, len(rows))
    if watch_share >= 0.75:
        return "must_watch"
    return "watch"


# ---------------------------------------------------------------------------
# Health / cumulative helpers
# ---------------------------------------------------------------------------


def _health_inputs(
    rows_by_step: dict[int, list[dict[str, Any]]], through_step: int, context: RunContext
) -> tuple[int, int, float]:
    accepted = 0
    decisions = 0
    pnl = sum(
        _float(row.get("realized_profit_eur"), 0.0)
        for step, rows in context.bid_rows_by_step.items()
        if step <= through_step
        for row in rows
    )
    for step, rows in rows_by_step.items():
        if step > through_step:
            continue
        for row in rows:
            if _decision_action(row) != "bid":
                continue
            decisions += 1
            if row.get("verifier_accepted") is True:
                accepted += 1
    return accepted, decisions, round(pnl, 2)


def _agent_cumulative_pnl(context: RunContext, agent_id: str, through_step: int) -> float:
    return round(
        sum(
            _float(row.get("realized_profit_eur"), 0.0)
            for step, rows in context.bid_rows_by_step.items()
            if step <= through_step
            for row in rows
            if row.get("agent_id") == agent_id
        ),
        2,
    )


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status")
        if status:
            counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _coverage(rows_by_step: dict[int, list[dict[str, Any]]], through_step: int) -> float:
    intervals = 0
    hits = 0
    for step, rows in rows_by_step.items():
        if step > through_step:
            continue
        for row in rows:
            interval = row.get("forecast_interval_eur_mwh") or []
            price = row.get("market_price_eur_mwh")
            if len(interval) >= 2 and isinstance(price, int | float):
                intervals += 1
                hits += int(interval[0] <= price <= interval[1])
    return hits / intervals if intervals else 0.9


def _agent_acceptance_rate(rows_by_step: dict[int, list[dict[str, Any]]], agent_id: str) -> float:
    accepted = 0
    decisions = 0
    for rows in rows_by_step.values():
        for row in rows:
            if row.get("agent_id") != agent_id or _decision_action(row) != "bid":
                continue
            decisions += 1
            accepted += int(row.get("verifier_accepted") is True)
    return accepted / decisions if decisions else 0.0
