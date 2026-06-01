"""Graph node and edge construction for the society visualization."""

from __future__ import annotations

import math
from typing import Any

from heimdall_run_view._catalog import (
    FALLBACK_ARCHETYPES,
    FOCAL_AGENT_ID,
    GRAPH_ARCHETYPE_ORDER,
)
from heimdall_run_view._snapshot import _agent_acceptance_rate, _agent_cumulative_pnl
from heimdall_run_view._trace import (
    _belief,
    _decision_action,
    _frontend_archetype,
    _persona,
    _side_to_direction,
)
from heimdall_run_view._utils import RunContext, _float


def _society_edges(step: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive interaction edges from a single interval of agent decisions.

    Real traces hold no bilateral negotiations, so the graph cannot show OTC
    deals. What it can show faithfully is (a) same-side *consensus* between
    action agents that bid the same direction this interval and (b) the society
    *broadcast* star when agents consume the shared communication digest.
    """
    edges: list[dict[str, Any]] = []
    cap = 48

    # --- consensus: action agents bidding the same side this interval ---
    by_side: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if _decision_action(row) != "bid":
            continue
        if _frontend_archetype(str(row.get("archetype") or "")).endswith("-info"):
            continue
        side = str((row.get("decision") or {}).get("side") or "")
        if side in {"up", "down"}:
            by_side.setdefault(side, []).append(row)
    for side, group in by_side.items():
        if len(group) < 2:
            continue
        strength = min(1.0, len(group) / max(2, len(rows)))
        direction = _side_to_direction(side)
        member_ids = sorted({str(row.get("agent_id") or FOCAL_AGENT_ID) for row in group})
        for i, source_id in enumerate(member_ids):
            for target_id in member_ids[i + 1 :]:
                edges.append(
                    {
                        "id": f"consensus-{step}-{side}-{source_id}-{target_id}",
                        "source": source_id,
                        "target": target_id,
                        "kind": "consensus",
                        "side": side,
                        "direction": direction,
                        "market": "mFRR",
                        "strength": round(strength, 3),
                        "label": f"Same-side {side}",
                        "detail": f"{len(group)} agents bidding {side} this interval.",
                        "started_step": step,
                        "expires_step": step,
                    }
                )

    # --- broadcast: society_communication_context consumers around a hub ---
    consumers = [
        str(row.get("agent_id") or FOCAL_AGENT_ID)
        for row in rows
        if any(
            call.get("name") == "society_communication_context"
            for call in (row.get("tool_calls") or [])
        )
    ]
    if len(consumers) >= 2:
        chair = next((r for r in rows if str(r.get("agent_role")) == "society_chair"), None)
        focal = next((r for r in rows if str(r.get("archetype")) == "p2h"), None)
        hub_id = str((chair or focal or {}).get("agent_id") or consumers[0])
        for agent_id in consumers:
            if agent_id == hub_id:
                continue
            edges.append(
                {
                    "id": f"broadcast-{step}-{hub_id}-{agent_id}",
                    "source": hub_id,
                    "target": agent_id,
                    "kind": "broadcast",
                    "side": None,
                    "direction": None,
                    "market": "mFRR",
                    "strength": 0.5,
                    "label": "Society broadcast",
                    "detail": "Shared market digest broadcast to the society this interval.",
                    "started_step": step,
                    "expires_step": step,
                }
            )

    return edges[:cap]


def _nodes(
    rows: list[dict[str, Any]],
    focal: dict[str, Any],
    step: int,
    rows_by_step: dict[int, list[dict[str, Any]]],
    context: RunContext,
) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    all_rows = [row for step_rows in rows_by_step.values() for row in step_rows]
    for row in all_rows:
        agent_id = str(row.get("agent_id") or FOCAL_AGENT_ID)
        seen.setdefault(agent_id, row)
    if not seen:
        seen[FOCAL_AGENT_ID] = focal
    desired = max(6, min(50, len(seen) if len(seen) > 1 else 12))
    while len(seen) < desired:
        index = len(seen)
        seen[f"agent-peer-{index:02d}"] = {
            "agent_id": f"agent-peer-{index:02d}",
            "archetype": FALLBACK_ARCHETYPES[index % len(FALLBACK_ARCHETYPES)],
            "llm_id": "unavailable",
            "forecaster_id": "unavailable",
        }
    nodes: list[dict[str, Any]] = []
    ids = list(seen)
    for index, agent_id in enumerate(ids):
        row = seen[agent_id]
        is_focal = agent_id == str(focal.get("agent_id") or FOCAL_AGENT_ID) or (
            index == 0 and str(row.get("archetype")) == "p2h"
        )
        archetype = _frontend_archetype(str(row.get("archetype") or "p2h"))
        point = _graph_position(index, archetype, is_focal)
        current = next(
            (candidate for candidate in rows if candidate.get("agent_id") == agent_id), row
        )
        accepted_rate = _agent_acceptance_rate(rows_by_step, agent_id) if is_focal else None
        quantity = _float((current.get("decision") or {}).get("quantity_mwh"), 0.0)
        eval_row = context.bid_rows_by_step_agent.get((step, agent_id), {})
        pnl = _agent_cumulative_pnl(context, agent_id, step)
        tick_pnl = _float(eval_row.get("realized_profit_eur"), 0.0)
        nodes.append(
            {
                "id": agent_id,
                "persona": _persona(agent_id, row, is_focal),
                "x": point[0],
                "y": point[1],
                "open_position_mw": quantity
                if _side_to_direction((current.get("decision") or {}).get("side")) == "sell"
                else -quantity,
                "pnl_eur": pnl,
                "tick_pnl_eur": round(tick_pnl, 2),
                "belief": _belief(row, current),
                "is_focal": is_focal,
                "verifier_acceptance_rate": accepted_rate,
            }
        )
    nodes.sort(key=lambda item: (not item["is_focal"], item["id"]))
    return nodes


def _graph_position(index: int, archetype: str, is_focal: bool) -> tuple[float, float]:
    if is_focal:
        return (0.08, 0.0)
    try:
        cluster = GRAPH_ARCHETYPE_ORDER.index(archetype)
    except ValueError:
        cluster = GRAPH_ARCHETYPE_ORDER.index("arbitrageur")
    cluster_angle = (cluster / len(GRAPH_ARCHETYPE_ORDER)) * math.tau
    within = (index % 9) / 9
    radius = 0.32 + within * 0.45
    jitter_x = math.cos(index * 2.31) * 0.08
    jitter_y = math.sin(index * 1.93) * 0.08
    return (
        math.cos(cluster_angle) * radius + jitter_x,
        math.sin(cluster_angle) * radius + jitter_y,
    )
