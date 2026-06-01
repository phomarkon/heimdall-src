from __future__ import annotations

from heimdall_contracts import Persona
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.schemas import ToolCallRecord
from heimdall_ai_society.tools import AgentToolExecutor

from heimdall_ai_society._trace_helpers import _with_provenance, _with_provenance_all


def _seed_candidate_tools(
    *,
    objective: str,
    ablation_strategy: str,
    persona: Persona,
    tick: TickContext,
    executor: AgentToolExecutor,
    candidate_sizing_mode: str = "current",
    candidate_sizing_cap_fraction: float = 1.0,
    candidate_sizing_min_mwh: float = 0.25,
    candidate_sizing_max_candidates: int = 8,
) -> list[ToolCallRecord]:
    from heimdall_ai_society._prompts import _opportunity_hint, _required_simulation_tool, _feasibility_tool
    if objective not in {"bid_seeking", "stress_test"}:
        return []
    if persona.archetype.value not in {"p2h", "ev", "wind", "generator", "retailer", "renewables"}:
        return []
    if ablation_strategy == "cp11_llm_suggest_candidates":
        return []
    records = []
    lower, upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    hint = _opportunity_hint(
        price=tick.market_price_eur_mwh,
        up_lower=lower,
        up_upper=upper,
        down_lower=down_lower,
        down_upper=down_upper,
    )
    candidates = _candidate_arguments(
        tick,
        hint,
        ablation_strategy,
        persona,
        sizing_mode=candidate_sizing_mode,
        cap_fraction=candidate_sizing_cap_fraction,
        min_mwh=candidate_sizing_min_mwh,
        max_candidates=candidate_sizing_max_candidates,
    )
    if not candidates:
        if objective != "stress_test":
            return []
        candidates = [{"side": "up", "quantity_mwh": 2.0, "limit_price_eur_mwh": round(max(tick.market_price_eur_mwh + 1.0, lower), 2)}]
    simulate_tool = _required_simulation_tool(persona)
    feasibility_tool = _feasibility_tool(persona)
    menu = _candidate_menu(candidates, tick)
    if menu:
        records.append(
            ToolCallRecord(
                name="candidate_menu",
                arguments={"strategy": ablation_strategy},
                ok=True,
                result={"ok": True, "authority": "advisory", "candidates": menu},
                provenance="runner_seeded",
            )
        )
    for arguments in candidates:
        records.append(_with_provenance(executor.execute(feasibility_tool, arguments), "runner_seeded"))
        sim_record = executor.execute(simulate_tool, arguments)
        records.append(_with_provenance(sim_record, "runner_seeded"))
        resized = _resize_to_physical_limit(arguments, sim_record, min_mwh=candidate_sizing_min_mwh)
        if resized is not None:
            records.append(_with_provenance(executor.execute(feasibility_tool, resized), "runner_seeded"))
            records.append(_with_provenance(executor.execute(simulate_tool, resized), "runner_seeded"))
    if ablation_strategy in {
        "ranked_candidates",
        "ranked_committee",
        "committee_vote",
        "cp04_clearprob_ranked",
        "cp05_profit_clear_tradeoff",
        "cp06_side_clear_joint",
        "comm_broadcast_digest",
        "comm_broadcast_digest_risk_filter",
        "comm_broadcast_digest_priority_calibration",
        "comm_peer_signal",
        "comm_retry_council",
    } and records:
        records.append(
            ToolCallRecord(
                name="rank_candidate_set",
                arguments={"strategy": ablation_strategy},
                ok=True,
                result={
                    "ok": True,
                    "authority": "advisory",
                    "ranking": _rank_seeded_candidates(records, tick),
                    "rankings": _rank_seeded_candidates(records, tick),
                },
                provenance="runner_diagnostic",
            )
        )
    return records


def _seed_context_tools(
    persona: Persona,
    executor: AgentToolExecutor,
    *,
    include_outages: bool = False,
) -> list[ToolCallRecord]:
    calls = [
        executor.execute("run_forecaster", {}),
        executor.execute("get_activation_context", {"hours": 24}),
        executor.execute("get_opportunity_context", {"hours": 24}),
        executor.execute("get_market_regime_context", {"hours": 24, "zone": executor._forecast.zone}),
        executor.execute("get_uncertainty_digest", {}),
    ]
    if include_outages:
        zone = executor._forecast.zone
        calls.append(executor.execute("get_outages", {"hours": 168, "zone": zone}))
        calls.append(executor.execute("get_outage_impact", {"hours": 168, "zone": zone}))
        calls.append(executor.execute("get_grid_constraints", {"hours": 24, "zone": zone}))
        calls.append(executor.execute("get_border_pressure", {"hours": 24, "zone": zone, "counterparty": ""}))
    return _with_provenance_all(calls, "runner_seeded")


def _seed_specialist_tools(
    persona: Persona,
    executor: AgentToolExecutor,
    *,
    safety_toolset: str = "full",
) -> list[ToolCallRecord]:
    archetype = persona.archetype.value
    zone = executor._forecast.zone
    if archetype == "grid_constraint_analyst":
        return _with_provenance_all([
            executor.execute("get_grid_constraints", {"hours": 24, "zone": zone}),
            executor.execute("get_border_pressure", {"hours": 24, "zone": zone, "counterparty": ""}),
        ], "runner_seeded")
    if archetype == "outage_impact_scorer":
        return _with_provenance_all([
            executor.execute("get_outages", {"hours": 168, "zone": zone}),
            executor.execute("get_outage_impact", {"hours": 168, "zone": zone}),
        ], "runner_seeded")
    if safety_toolset == "context_only":
        return []
    if archetype == "limit_price_specialist":
        return _with_provenance_all([
            executor.execute("get_limit_price_guidance", {"side": "up", "quantity_mwh": 0.25}),
            executor.execute("get_limit_price_guidance", {"side": "down", "quantity_mwh": 0.25}),
        ], "runner_seeded")
    if archetype == "candidate_sizing_specialist":
        return _with_provenance_all([
            executor.execute("get_candidate_sizing_guidance", {"archetype": "p2h"}),
            executor.execute("get_candidate_sizing_guidance", {"archetype": "ev"}),
            executor.execute("get_candidate_sizing_guidance", {"archetype": "wind"}),
        ], "runner_seeded")
    if archetype == "uncertainty_auditor":
        return _with_provenance_all([
            executor.execute("get_candidate_rejection_summary", {}),
            executor.execute("get_decision_trace_summary", {}),
        ], "runner_seeded")
    if archetype == "decision_auditor":
        return _with_provenance_all([
            executor.execute("get_candidate_rejection_summary", {}),
            executor.execute("get_decision_trace_summary", {}),
        ], "runner_seeded")
    return []


def _candidate_arguments(
    tick: TickContext,
    hint: dict[str, object],
    strategy: str,
    persona: Persona,
    *,
    sizing_mode: str = "current",
    cap_fraction: float = 1.0,
    min_mwh: float = 0.25,
    max_candidates: int = 8,
) -> list[dict[str, object]]:
    price = tick.market_price_eur_mwh
    up_lower, up_upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    base_side = hint["candidate_bid_side"]
    base_limit = hint["suggested_limit_price_eur_mwh"]
    base_quantity = float(hint["suggested_probe_quantity_mwh"])
    if persona.archetype.value == "p2h":
        base_quantity = min(base_quantity, max(0.25, persona.capacity_mw * 0.25 * 0.04))
    elif persona.archetype.value == "ev":
        energy = float(persona.storage_mwh or max(persona.capacity_mw, 1.0))
        availability_cap = persona.capacity_mw * 0.75 * 0.25
        soc_cap = energy * 0.5
        base_quantity = min(base_quantity, max(0.25, availability_cap, 0.25), soc_cap)
    if sizing_mode != "current":
        cap = _candidate_quantity_cap(persona, sizing_mode=sizing_mode, cap_fraction=cap_fraction, min_mwh=min_mwh)
        base_quantity = max(min_mwh, min(cap, max(base_quantity, cap)))
    if strategy == "diverse_action_society":
        if persona.archetype.value in {"p2h", "ev"}:
            return _specialist_candidate_arguments(
                price=price,
                up_lower=up_lower,
                up_upper=up_upper,
                down_lower=down_lower,
                down_upper=down_upper,
                base_side=str(base_side) if base_side is not None else None,
                cap_quantity=base_quantity,
                archetype=persona.archetype.value,
                sizing_mode=sizing_mode,
                min_mwh=min_mwh,
                max_candidates=max_candidates,
            )
        return _candidate_price_ablation_arguments(
            price=price,
            up_lower=up_lower,
            up_upper=up_upper,
            down_lower=down_lower,
            down_upper=down_upper,
            base_side=str(base_side) if base_side is not None else None,
            base_quantity=base_quantity,
            strategy="cp02_balanced_clear_ladder",
        )
    if strategy.startswith("comm_"):
        if persona.archetype.value in {"p2h", "ev"}:
            return _specialist_candidate_arguments(
                price=price,
                up_lower=up_lower,
                up_upper=up_upper,
                down_lower=down_lower,
                down_upper=down_upper,
                base_side=None,
                cap_quantity=base_quantity,
                archetype=persona.archetype.value,
                sizing_mode=sizing_mode,
                min_mwh=min_mwh,
                max_candidates=max_candidates,
            )
        return _candidate_price_ablation_arguments(
            price=price,
            up_lower=up_lower,
            up_upper=up_upper,
            down_lower=down_lower,
            down_upper=down_upper,
            base_side=str(base_side) if base_side is not None else None,
            base_quantity=base_quantity,
            strategy="cp02_balanced_clear_ladder",
        )
    if strategy.startswith("cp") or strategy == "deterministic_rich":
        return _candidate_price_ablation_arguments(
            price=price,
            up_lower=up_lower,
            up_upper=up_upper,
            down_lower=down_lower,
            down_upper=down_upper,
            base_side=str(base_side) if base_side is not None else None,
            base_quantity=base_quantity,
            strategy=strategy,
        )
    if strategy == "direction_prior":
        base_side = "up" if float(hint["up_edge_lower_minus_last_price"]) >= float(hint["down_edge_last_price_minus_down_upper"]) else "down"
        base_limit = round(max(price + 1.0, up_lower - 1.0), 2) if base_side == "up" else round(min(price - 1.0, down_upper + 1.0), 2)
    if strategy in {"both_side_probes", "ranked_candidates", "committee_vote", "ranked_committee"}:
        return [
            {"side": "up", "quantity_mwh": base_quantity, "limit_price_eur_mwh": round(max(price + 1.0, up_lower - 1.0), 2)},
            {"side": "down", "quantity_mwh": base_quantity, "limit_price_eur_mwh": round(min(price - 1.0, down_upper + 1.0), 2)},
        ]
    if strategy == "price_ladder":
        if base_side is None:
            return []
        if base_side == "up":
            limits = [max(price + 1.0, up_lower - offset) for offset in (10.0, 5.0, 1.0)]
        else:
            limits = [min(price - 1.0, down_upper + offset) for offset in (1.0, 5.0, 10.0)]
        return [{"side": base_side, "quantity_mwh": base_quantity, "limit_price_eur_mwh": round(limit, 2)} for limit in limits]
    if base_side is None or base_limit is None:
        return []
    return [{"side": base_side, "quantity_mwh": base_quantity, "limit_price_eur_mwh": base_limit}]


def _candidate_price_ablation_arguments(
    *,
    price: float,
    up_lower: float,
    up_upper: float,
    down_lower: float,
    down_upper: float,
    base_side: str | None,
    base_quantity: float,
    strategy: str,
) -> list[dict[str, object]]:
    up_mid = (up_lower + up_upper) / 2.0
    down_mid = (down_lower + down_upper) / 2.0

    def up(limit: float, quantity: float = base_quantity) -> dict[str, object]:
        return {"side": "up", "quantity_mwh": quantity, "limit_price_eur_mwh": round(limit, 2)}

    def down(limit: float, quantity: float = base_quantity) -> dict[str, object]:
        return {"side": "down", "quantity_mwh": quantity, "limit_price_eur_mwh": round(limit, 2)}

    aggressive = [
        up(min(price + 0.25, up_lower - 20.0)),
        up(min(price + 1.0, up_lower - 10.0)),
        down(min(price - 20.0, down_lower - 5.0)),
        down(min(price - 10.0, down_mid - 10.0)),
    ]
    balanced = [
        up(min(price + 2.0, up_lower - 5.0)),
        up(min(up_mid, up_lower + 2.0)),
        down(min(price - 5.0, down_mid)),
        down(min(down_upper, price - 2.0)),
    ]
    conservative = [
        up(max(price + 1.0, up_lower - 1.0)),
        down(min(price - 1.0, down_upper + 1.0)),
    ]
    if strategy == "cp01_aggressive_clear_ladder":
        return aggressive
    if strategy == "cp02_balanced_clear_ladder":
        return balanced
    if strategy in {
        "cp03_wide_price_ladder",
        "cp04_clearprob_ranked",
        "cp05_profit_clear_tradeoff",
        "cp06_side_clear_joint",
        "cp09_watch_threshold_low",
        "cp10_watch_threshold_high",
        "cp12_llm_suggest_plus_code_ladder",
        "cp12_delivery_risk_aware",
    }:
        return _dedupe_candidates([aggressive[0], aggressive[1], balanced[0], balanced[1], conservative[0], conservative[1]])
    if strategy == "cp07_downside_first":
        return _dedupe_candidates(
            [
                down(min(price - 25.0, down_lower - 10.0)),
                down(min(price - 10.0, down_mid - 5.0)),
                down(min(price - 1.0, down_upper + 1.0)),
                up(min(price + 1.0, up_lower - 10.0)),
            ]
        )
    if strategy == "deterministic_rich":
        half = max(0.25, round(base_quantity * 0.5, 3))
        grid = aggressive + balanced + conservative
        grid = grid + [{**candidate, "quantity_mwh": half} for candidate in (aggressive + conservative)]
        return _dedupe_candidates(grid)
    if strategy == "cp08_quantity_price_grid":
        side = base_side or "up"
        limits = (
            [min(price + 0.5, up_lower - 15.0), max(price + 1.0, up_lower - 1.0)]
            if side == "up"
            else [min(price - 20.0, down_lower - 5.0), min(price - 1.0, down_upper + 1.0)]
        )
        return [{"side": side, "quantity_mwh": quantity, "limit_price_eur_mwh": round(limit, 2)} for quantity in (1.0, 2.0) for limit in limits]
    return conservative


def _specialist_candidate_arguments(
    *,
    price: float,
    up_lower: float,
    up_upper: float,
    down_lower: float,
    down_upper: float,
    base_side: str | None,
    cap_quantity: float,
    archetype: str,
    sizing_mode: str = "current",
    min_mwh: float = 0.25,
    max_candidates: int = 8,
) -> list[dict[str, object]]:
    up_mid = (up_lower + up_upper) / 2.0
    down_mid = (down_lower + down_upper) / 2.0
    quantity_cap = max(0.25, cap_quantity)
    if sizing_mode == "current" and archetype == "ev":
        quantities = [quantity for quantity in [0.25, 0.5, 1.0] if quantity <= quantity_cap + 1e-9]
    elif sizing_mode == "current":
        quantities = [quantity for quantity in [0.25, 0.5, 1.0, 2.0] if quantity <= quantity_cap + 1e-9]
    else:
        quantities = _sizing_quantity_ladder(quantity_cap, mode=sizing_mode, min_mwh=min_mwh)
    if not quantities:
        quantities = [round(quantity_cap, 6)]
    sides = [base_side] if base_side in {"up", "down"} else ["up", "down"]
    candidates = []
    for side in sides:
        if side == "up":
            limits = [max(price + 1.0, up_lower - 1.0), min(up_mid, up_lower + 2.0)]
        else:
            limits = [min(price - 1.0, down_upper + 1.0), min(price - 5.0, down_mid)]
        for quantity in quantities:
            for limit in limits[:2]:
                candidates.append({"side": side, "quantity_mwh": quantity, "limit_price_eur_mwh": round(limit, 2)})
    return _dedupe_candidates(candidates[:max_candidates])


def _candidate_quantity_cap(
    persona: Persona,
    *,
    sizing_mode: str,
    cap_fraction: float,
    min_mwh: float,
) -> float:
    tick_capacity = max(0.0, persona.capacity_mw * 0.25)
    archetype = persona.archetype.value
    if archetype == "ev":
        energy = float(persona.storage_mwh or max(persona.capacity_mw, 1.0))
        physical_cap = min(tick_capacity * 0.75, energy * 0.5)
    elif archetype == "wind":
        physical_cap = tick_capacity * 0.45
    elif archetype == "renewables":
        physical_cap = tick_capacity * 0.55
    elif archetype == "generator":
        physical_cap = tick_capacity * 0.35
    elif archetype == "retailer":
        physical_cap = tick_capacity * 0.12
    else:
        physical_cap = tick_capacity
    if sizing_mode == "medium":
        physical_cap *= 0.5
    elif sizing_mode == "large":
        physical_cap *= cap_fraction
    else:
        physical_cap = min(2.0, physical_cap)
    return max(min_mwh, round(physical_cap, 6))


def _sizing_quantity_ladder(cap_mwh: float, *, mode: str, min_mwh: float) -> list[float]:
    cap = max(min_mwh, cap_mwh)
    seeds = [min_mwh, 2.0, 4.0, 0.5 * cap]
    if mode == "large":
        seeds.append(cap)
    out = []
    for quantity in seeds:
        clipped = round(min(max(min_mwh, quantity), cap), 6)
        if clipped not in out:
            out.append(clipped)
    return out


def _dedupe_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, object, object]] = set()
    out = []
    for candidate in candidates:
        key = (candidate["side"], candidate["quantity_mwh"], candidate["limit_price_eur_mwh"])
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _candidate_menu(candidates: list[dict[str, object]], tick: TickContext) -> list[dict[str, object]]:
    menu = []
    for idx, candidate in enumerate(candidates):
        side = str(candidate["side"])
        quantity = float(candidate["quantity_mwh"])
        limit = float(candidate["limit_price_eur_mwh"])
        lower, upper = tick.forecast.interval_for_side(side)
        median = (lower + upper) / 2.0
        if side == "up":
            expected_spread = median - tick.forecast.spot_price_eur_mwh
            worst_spread = lower - tick.forecast.spot_price_eur_mwh
        else:
            expected_spread = tick.forecast.spot_price_eur_mwh - median
            worst_spread = tick.forecast.spot_price_eur_mwh - upper
        menu.append(
            {
                "candidate_id": f"C{idx + 1:02d}",
                "side": side,
                "quantity_mwh": quantity,
                "limit_price_eur_mwh": limit,
                "clear_probability_proxy": _clear_probability_proxy(side=side, limit_price=limit, lower=lower, median=median, upper=upper),
                "expected_profit_proxy_eur": round(quantity * expected_spread, 6),
                "worst_case_profit_proxy_eur": round(quantity * worst_spread, 6),
            }
        )
    return menu


def _clear_probability_proxy(*, side: str, limit_price: float, lower: float, median: float, upper: float) -> float:
    if side == "down":
        if limit_price >= upper:
            return 0.85
        if limit_price >= median:
            return 0.60
        if limit_price >= lower:
            return 0.35
        return 0.10
    if limit_price <= lower:
        return 0.85
    if limit_price <= median:
        return 0.60
    if limit_price <= upper:
        return 0.35
    return 0.10


def _rank_seeded_candidates(records: list[ToolCallRecord], tick: TickContext) -> list[dict[str, object]]:
    rows = []
    for record in records:
        if record.name not in {"simulate_bid", "simulate_ev_bid", "simulate_wind_bid", "simulate_generator_bid", "simulate_retailer_bid", "simulate_renewables_bid"}:
            continue
        side = str(record.arguments.get("side", "up"))
        lower, upper = tick.forecast.interval_for_side(side)
        median = (lower + upper) / 2.0
        clear_probability = _clear_probability_proxy(
            side=side,
            limit_price=float(record.arguments.get("limit_price_eur_mwh", 0.0)),
            lower=lower,
            median=median,
            upper=upper,
        )
        worst_case_profit = float(record.result.get("worst_case_profit_eur") or -1000.0)
        rows.append(
            {
                "arguments": record.arguments,
                "accepted": record.result.get("accepted"),
                "worst_case_profit_eur": record.result.get("worst_case_profit_eur"),
                "clear_probability_proxy": clear_probability,
                "score": round((1.0 if record.result.get("accepted") else 0.0) + clear_probability + max(worst_case_profit, -1000.0) / 1000.0, 6),
                "reason_codes": record.result.get("reason_codes", []),
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)


def _matching_candidate_row(decision: object, candidates: list[dict[str, object]]) -> dict[str, object] | None:
    from heimdall_ai_society.schemas import LLMBidDecision
    if not isinstance(decision, LLMBidDecision):
        return None
    if decision.action != "bid" or decision.side not in {"up", "down"}:
        return None
    for row in candidates:
        args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
        if args.get("side") != decision.side:
            continue
        try:
            quantity = float(args.get("quantity_mwh"))
            limit = float(args.get("limit_price_eur_mwh"))
        except (TypeError, ValueError):
            continue
        if abs(quantity - float(decision.quantity_mwh or -1.0)) > 1e-9:
            continue
        if abs(limit - float(decision.limit_price_eur_mwh or -1e9)) > 1e-9:
            continue
        return row
    return None


def _resize_to_physical_limit(
    arguments: dict[str, object],
    sim_record: ToolCallRecord,
    *,
    min_mwh: float,
) -> dict[str, object] | None:
    """If a seeded candidate was rejected purely for exceeding the physical envelope, return a
    copy resized down to the simulator's reported ``physical_limit_mwh`` so the chooser can still
    bid the maximum feasible quantity. Without this, candidate_sizing=large makes every oversized
    candidate unbiddable under scenario_envelope (e.g. a 12.5 MWh P2H probe vs a 2-6.25 MWh ramp
    envelope), starving the deterministic/selector arms of any accepted candidate. Returns None
    when not applicable."""
    if not sim_record.ok or not isinstance(sim_record.result, dict):
        return None
    result = sim_record.result
    if result.get("accepted") is True:
        return None
    if not any("limit_exceeded" in str(code) for code in (result.get("reason_codes") or [])):
        return None
    try:
        limit = float(result.get("physical_limit_mwh"))
        requested = float(arguments.get("quantity_mwh"))
    except (TypeError, ValueError):
        return None
    if limit < min_mwh or limit >= requested:
        return None
    resized = dict(arguments)
    resized["quantity_mwh"] = round(limit, 6)
    return resized
