from __future__ import annotations

import json
from datetime import timedelta

from heimdall_contracts import Persona
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord


def _prompt(
    persona: Persona,
    tick: TickContext,
    *,
    agent_role: str = "action_agent",
    objective: str,
    ablation_strategy: str,
    safety_toolset: str = "full",
    communication_context: dict[str, object] | None = None,
    memory_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    lower, upper = tick.forecast.interval_for_side("up")
    down_lower, down_upper = tick.forecast.interval_for_side("down")
    opportunity_hint = _opportunity_hint(
        price=tick.market_price_eur_mwh,
        up_lower=lower,
        up_upper=upper,
        down_lower=down_lower,
        down_upper=down_upper,
    )
    objective_instruction = _objective_instruction(objective)
    if objective == "unverified_bid_seeking" or safety_toolset == "context_only":
        system = (
            "You are one Heimdall BRP persona in a counterfactual Nordic mFRR simulation. "
            "The primary product is auditable market intelligence: decide whether this MTU is must_watch, watch, or ignore; explain risk, uncertainty, opportunity, and evidence. "
            "No pre-submit simulator or verifier tools are available in this ablation. Use only visible forecast and market context before proposing an action. "
            "Return only JSON matching the requested schema. You may bid, watch, or abstain. "
            "Never claim to trade live markets."
        )
    else:
        system = (
            "You are one Heimdall BRP persona in a counterfactual Nordic mFRR simulation. "
            "The primary product is auditable market intelligence: decide whether this MTU is must_watch, watch, or ignore; explain risk, uncertainty, opportunity, and evidence. "
            "Use tools to inspect context and run your archetype's simulator before proposing a bid. "
            "Return only JSON matching the requested schema. You may bid, watch, or abstain. "
            "Never claim to trade live markets."
        )
    user = {
        "agent_id": persona.agent_id,
        "archetype": persona.archetype.value,
        "agent_role": agent_role,
        "risk_attitude": persona.risk_attitude.value,
        "capacity_mw": persona.capacity_mw,
        "storage_mwh": persona.storage_mwh,
        "tick_timestamp": tick.timestamp.isoformat(),
        "agent_observed_at": (tick.timestamp - timedelta(minutes=persona.info_latency_min)).isoformat(),
        "zone": tick.forecast.zone,
        "last_price_eur_mwh": tick.market_price_eur_mwh,
        "mfrr_up_interval_eur_mwh": [lower, upper],
        "mfrr_down_interval_eur_mwh": [down_lower, down_upper],
        "opportunity_hint": opportunity_hint,
        "experiment_objective": objective,
        "ablation_strategy": ablation_strategy,
        "tool_policy_guidance": _tool_policy_guidance(persona, objective=objective, safety_toolset=safety_toolset),
        "society_context": communication_context,
        "memory_context": memory_context,
        "instruction": (
            _decision_instruction(objective, safety_toolset)
            + "Memory is advisory only; ignore it when current tools or observed_at evidence disagree. "
            "Always set watch_label in {must_watch, watch, ignore}, risk_label and uncertainty_label in {low, medium, high}, opportunity_label in {none, weak, actionable}, "
            "watch_reasons from activation_risk, price_volatility, forecast_uncertainty, accepted_bid_available, verifier_rejection_cluster, cross_agent_disagreement, "
            "priority_label in {low, medium, high, critical}, priority_score from 0.0 to 1.0, operator_action in {ignore, monitor, inspect, prepare_bid, escalate}, "
            "and priority_reason in {none, activation, profit_edge, accepted_candidate, uncertainty, rejection_cluster, cross_agent_disagreement}. "
            "Use watch_label as a broad monitoring signal; use high/critical priority only when scarce operator review should go to this MTU before ordinary watch periods. "
            "Use abstain only when confidence is low after checking the available forecast/context tools. "
            + _candidate_probe_instruction(objective, safety_toolset)
            + "For watch/abstain omit bid fields; for bid include side up/down, quantity_mwh, and limit_price_eur_mwh. "
            + objective_instruction
            + _ablation_instruction(ablation_strategy, persona, objective=objective, safety_toolset=safety_toolset)
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    ]


def _decision_instruction(objective: str, safety_toolset: str) -> str:
    if objective == "unverified_bid_seeking" or safety_toolset == "context_only":
        return (
            "Choose action bid, watch, or abstain. No pre-submit verifier is available. "
            "Use watch when this MTU is worth monitoring but direction, price, or confidence is unclear. "
        )
    return "Choose action bid, watch, or abstain. Use watch when this MTU is worth monitoring but no safe bid is justified. "


def _candidate_probe_instruction(objective: str, safety_toolset: str) -> str:
    if objective == "unverified_bid_seeking" or safety_toolset == "context_only":
        return ""
    return "If opportunity_hint says candidate_bid_side is up or down, test small candidates with your feasibility and simulation tools before final action. "


def _tool_policy_guidance(
    persona: Persona,
    *,
    objective: str = "worth_bidding",
    safety_toolset: str = "full",
) -> str:
    archetype = persona.archetype.value
    if objective == "unverified_bid_seeking" or safety_toolset == "context_only":
        if archetype in {"p2h", "ev", "wind", "generator", "retailer", "renewables"}:
            return (
                f"{archetype} is bid-capable in this unverified ablation. "
                "No simulator, feasibility, or verifier tools are available before submission; use visible forecast/context only."
            )
        return f"{archetype} is context-only in this harness. Do not propose final bids; use watch or abstain."
    if archetype == "p2h":
        return "P2H is backend-gated through simulate_bid. You may propose a final bid only after simulate_bid accepted the exact candidate."
    if archetype == "ev":
        return (
            "EV is backend-gated through simulate_ev_bid. You may propose a final bid only after simulate_ev_bid accepted the exact candidate. "
            "For EV V2, low-price down-regime probes are the only current action path: if current price is <=55 EUR/MWh, a 0.25 MWh down candidate is accepted, "
            "worst_case_profit_eur >=2, expected_profit_eur >=10, and opportunity is actionable, prefer the exact accepted bid; otherwise use watch."
        )
    if archetype == "wind":
        return "Wind is proxy-simulator-backed through simulate_wind_bid. You may propose a final bid only after simulate_wind_bid accepted the exact candidate."
    if archetype == "generator":
        return "Generator is proxy-simulator-backed through simulate_generator_bid. You may propose a final bid only after simulate_generator_bid accepted the exact candidate."
    if archetype == "retailer":
        return "Retailer demand response is proxy-simulator-backed through simulate_retailer_bid. You may propose a final bid only after simulate_retailer_bid accepted the exact candidate."
    if archetype == "renewables":
        return "Aggregated renewables are proxy-simulator-backed through simulate_renewables_bid. You may propose a final bid only after simulate_renewables_bid accepted the exact candidate."
    if archetype == "market_mechanics_expert":
        return "You are context-only. Explain Nordic balancing mechanics, settlement implications, no-activation pricing, sign conventions, and evaluator caveats; do not propose final bids."
    if archetype == "imbalance_analytics_expert":
        return "You are context-only. Use forecasts, activation context, weather/load/generation/outage/flow signals, and uncertainty tools to diagnose imbalance drivers; do not propose final bids."
    if archetype == "trading_risk_monitor":
        return "You are context-only. Assess strategic behavior, tail risk, liquidity/gate-closure risk, and whether simulator-accepted candidates are prudent; do not propose final bids."
    if archetype == "grid_constraint_analyst":
        return "You are context-only. Use get_grid_constraints and flow tools to explain congestion pressure, flow direction, RAM, CNECs, and shadow-price signals; do not propose final bids."
    if archetype == "outage_impact_scorer":
        return "You are context-only. Use outage tools to score unavailable MW, duration, zone relevance, and whether outages plausibly explain activation or price tails; do not propose final bids."
    if archetype == "limit_price_specialist":
        return "You are context-only. Use get_limit_price_guidance to explain crossing-aware bid prices and clear-probability/profit tradeoffs; do not propose final bids."
    if archetype == "candidate_sizing_specialist":
        return "You are context-only. Use get_candidate_sizing_guidance and rejection summaries to recommend small safe probes, especially for EV, P2H, and wind; do not propose final bids."
    if archetype == "uncertainty_auditor":
        return "You are context-only. Use uncertainty and rejection tools to flag forecast width, side ambiguity, and when accepted candidates should stay advisory; do not propose final bids."
    if archetype == "decision_auditor":
        return "You are context-only. Use get_decision_trace_summary to explain why the society should bid, watch, or abstain and compare action-agent and information-agent evidence; do not propose final bids."
    return f"{archetype} is context-only in this harness. Do not propose final bids; use watch or abstain."


def _opportunity_hint(
    *,
    price: float,
    up_lower: float,
    up_upper: float,
    down_lower: float,
    down_upper: float,
) -> dict[str, object]:
    up_edge = up_lower - price
    down_edge = price - down_upper
    if up_edge >= max(5.0, down_edge):
        side = "up"
        limit = round(max(price + 1.0, up_lower - 1.0), 2)
    elif down_edge >= 5.0:
        side = "down"
        limit = round(min(price - 1.0, down_upper + 1.0), 2)
    else:
        side = None
        limit = None
    return {
        "candidate_bid_side": side,
        "suggested_probe_quantity_mwh": 2.0,
        "suggested_limit_price_eur_mwh": limit,
        "up_edge_lower_minus_last_price": round(up_edge, 6),
        "down_edge_last_price_minus_down_upper": round(down_edge, 6),
    }


def _ablation_instruction(
    strategy: str,
    persona: Persona,
    *,
    objective: str = "worth_bidding",
    safety_toolset: str = "full",
) -> str:
    if objective == "unverified_bid_seeking" or safety_toolset == "context_only":
        if strategy == "comm_broadcast_digest":
            return " You receive a shared society digest, but make an independent unverified decision from visible forecast and market context."
        return ""
    if strategy == "direction_prior":
        return " Prefer the side with the stronger non-leaking activation and spread prior; explicitly mention the side comparison."
    if strategy == "both_side_probes":
        return " You have pre-probed both up and down sides. Choose only a simulator-accepted side, otherwise watch."
    if strategy == "price_ladder":
        return " You have pre-probed a limit-price ladder. Prefer the accepted price most likely to clear while preserving non-negative worst-case profit."
    if strategy in {"ranked_candidates", "ranked_committee"}:
        return " Treat the ranked candidate table as the action menu. Do not invent a new bid; pick the best accepted candidate or watch."
    if strategy == "rejection_explain":
        return " Before final action, explain the simulator rejection reason codes and how they changed the action."
    if strategy == "risk_trio":
        return f" Act according to your risk attitude ({persona.risk_attitude.value}) while respecting simulator acceptance."
    if strategy == "price_styles":
        return " Your persona represents one price-placement style; use the pre-probed candidate closest to that style."
    if strategy == "committee_vote":
        return " You are part of a committee. Make an independent decision from the verified candidate set and give a concise confidence rationale."
    if strategy == "random_persona_10":
        return " Your randomized persona parameters are intentional; use them to create diverse but simulator-backed behavior."
    if strategy == "mixed_advisory":
        return " Advisory personas should contribute context only; P2H/EV must remain backend-gated for final bids."
    if strategy == "diverse_action_society":
        return " This is an action-capable diverse society. Use activation context and candidate_menu, then bid only with an exact backend-accepted candidate; otherwise use watch for important hours or abstain for low signal."
    if strategy == "comm_broadcast_digest":
        return " You receive a shared society digest, but make an independent simulator-backed decision. Use watch to mark important hours where your archetype should stay alert."
    if strategy == "comm_broadcast_digest_priority_calibration":
        return (
            " You receive a shared society digest, but make an independent simulator-backed decision. "
            "This run calibrates operator priority labels: watch_label remains broad monitoring, while priority_label/priority_score should rank scarce review attention. "
            "Reserve high or critical for MTUs with strong evidence such as activation risk, positive profit edge, accepted candidate quality, unusual uncertainty, rejection clusters, or cross-agent disagreement. "
            "For a 24-tick run, aim to mark no more than 12 ticks as high or critical priority; medium is the default for useful but non-scarce monitoring."
        )
    if strategy == "comm_broadcast_digest_risk_filter":
        return " You receive a shared society digest and make an independent simulator-backed decision. A final policy layer may downgrade low-confidence, high-risk, or side-disagreeing accepted bids to watch; keep your rationale auditable."
    if strategy == "comm_info_then_action":
        return " Market experts reason first; action agents receive their expert_summaries before deciding. Treat expert summaries as advisory market intelligence, cite where they changed your action, and bid only with an exact simulator-accepted candidate."
    if strategy == "comm_peer_signal":
        return " You see earlier peer decisions from this MTU. Treat them as advisory signals; disagree when your simulator results say so, and bid only with an exact accepted candidate."
    if strategy == "comm_retry_council":
        return " Use the shared digest, peer signals, and retry feedback. If your first decision is rejected or too cautious while an accepted candidate exists, reconsider once and choose the best exact simulator-accepted candidate or watch."
    if strategy == "comm_deliberation_protocol":
        return " Follow the bounded deliberation protocol: gather your own tool evidence, post a concise note, treat peer notes as advisory, run your own action probe before bidding, and use only your own accepted simulator result for a final bid."
    if strategy == "comm_central_supervisor":
        return " You are a specialist reporting to the central supervisor. Probe your own candidates and recommend only exact simulator-accepted bids; the execution gateway, not you, commits the final market order."
    if strategy.startswith("cp"):
        base = " Treat candidate_menu and simulation results as the action menu. You may choose any backend-accepted candidate or watch; do not invent a final bid outside simulated candidates."
        if strategy == "cp11_llm_suggest_candidates":
            return " First suggest and simulate at least three candidate prices yourself using simulate_bid, including one aggressive clear-seeking price. Then choose a simulator-accepted candidate or watch."
        if strategy == "cp12_llm_suggest_plus_code_ladder":
            return base + " You may add one extra self-suggested candidate with simulate_bid before final action."
        if strategy == "cp12_delivery_risk_aware":
            if persona.archetype.value in {"wind", "renewables", "ev"}:
                return (
                    " DELIVERY RISK IS YOUR PRIMARY CONCERN. Your variable-output asset has large real availability "
                    "uncertainty (generation forecast error CV ~0.25-0.30 from DK data): any committed MWh you cannot "
                    "physically deliver is re-settled against you at the imbalance price as a REAL LOSS that the verifier "
                    "does NOT protect against. Therefore: (1) before any bid you MUST call simulate_bid at HALF the menu "
                    "quantity and, if accepted, submit that smaller bid — committing less directly caps your worst-case "
                    "delivery loss; (2) when forecast/availability uncertainty is elevated (uncertainty_label high or "
                    "volatile regime), prefer WATCH over bidding. Accept lower expected profit in exchange for a smaller "
                    "delivery-loss tail. Only ever submit an exact simulator-accepted candidate."
                )
            return base + " You may add one extra self-suggested candidate with simulate_bid before final action."
        if strategy == "cp13_llm_probe_refine_frontier":
            return (
                " Probe-and-refine to the aggressive feasible frontier: start from an aggressive clear-seeking price, "
                "call your simulator tool, and read accepted plus worst_case_profit_eur. If rejected, adjust the price/quantity "
                "and re-simulate to find the most aggressive candidate that still returns accepted=true; the verifier floor bounds "
                "your downside, so push for higher worst-case profit rather than retreating to watch. Submit the accepted candidate "
                "with the highest worst_case_profit_eur, and only watch/abstain if repeated simulation finds no accepted candidate."
            )
        if strategy == "cp09_watch_threshold_low":
            return base + " Favor bidding when a candidate has high clear-probability proxy and non-negative simulator worst-case profit."
        if strategy == "cp10_watch_threshold_high":
            return base + " Bid only when both clear-probability proxy and simulator worst-case profit are clearly strong; otherwise watch."
        return base
    return ""


def _compact_tool_records_for_prompt(records: list[ToolCallRecord]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for record in records:
        if record.name == "candidate_menu":
            result = record.result
            candidates = result.get("candidates", []) if isinstance(result.get("candidates"), list) else []
            compact.append(
                {
                    "name": record.name,
                    "arguments": record.arguments,
                    "ok": record.ok,
                    "result": {
                        "authority": result.get("authority"),
                        "summary": result.get("summary"),
                        "candidate_count": len(candidates),
                        "candidates": [_compact_candidate(candidate) for candidate in candidates[:6]],
                    },
                    "error": record.error,
                }
            )
            continue
        if record.name == "rank_candidate_set":
            result = record.result
            rankings = result.get("rankings", []) if isinstance(result.get("rankings"), list) else []
            compact.append(
                {
                    "name": record.name,
                    "arguments": record.arguments,
                    "ok": record.ok,
                    "result": {
                        "authority": result.get("authority"),
                        "summary": result.get("summary"),
                        "ranking_count": len(rankings),
                        "rankings": [_compact_candidate(candidate) for candidate in rankings[:6]],
                    },
                    "error": record.error,
                }
            )
            continue
        if record.name.startswith("simulate") or (record.name.startswith("get_") and record.name.endswith("_bid_feasibility")):
            result = record.result
            compact.append(
                {
                    "name": record.name,
                    "arguments": record.arguments,
                    "ok": record.ok,
                    "result": {
                        "accepted": result.get("accepted"),
                        "backend": result.get("backend"),
                        "authority": result.get("authority"),
                        "controls_acceptance": result.get("controls_acceptance"),
                        "comparison": {
                            "accepted_disagreement": (result.get("comparison") or {}).get("accepted_disagreement") if isinstance(result.get("comparison"), dict) else None,
                            "proxy_false_positive": (result.get("comparison") or {}).get("proxy_false_positive") if isinstance(result.get("comparison"), dict) else None,
                        },
                        "reason_codes": result.get("reason_codes", []),
                        "worst_case_profit_eur": result.get("worst_case_profit_eur"),
                        "expected_spread_eur_mwh": result.get("expected_spread_eur_mwh"),
                        "rough_expected_profit_eur": result.get("rough_expected_profit_eur"),
                        "risk_flags": result.get("risk_flags", []),
                    },
                    "error": record.error,
                }
            )
            continue
        result = record.result
        if record.name in {"get_market_regime_context", "get_border_pressure", "get_outage_impact", "get_uncertainty_digest"}:
            compact.append(
                {
                    "name": record.name,
                    "arguments": record.arguments,
                    "ok": record.ok,
                    "result": {
                        "kind": result.get("kind"),
                        "authority": result.get("authority"),
                        "regime_label": result.get("regime_label"),
                        "pressure_label": result.get("pressure_label"),
                        "impact_label": result.get("impact_label"),
                        "uncertainty_label": result.get("uncertainty_label"),
                        "side_ambiguity": result.get("side_ambiguity"),
                        "candidate_side_hint": result.get("candidate_side_hint"),
                        "signals": result.get("signals", {}),
                    },
                    "error": record.error,
                }
            )
            continue
        compact.append(
            {
                "name": record.name,
                "arguments": record.arguments,
                "ok": record.ok,
                "result": {
                    "kind": result.get("kind"),
                    "authority": result.get("authority"),
                    "summary": result.get("summary"),
                    "rows": len(result.get("rows", [])) if isinstance(result.get("rows"), list) else None,
                    "error_code": result.get("error_code"),
                },
                "error": record.error,
            }
        )
    return compact


def _compact_candidate(candidate: object) -> object:
    if not isinstance(candidate, dict):
        return candidate
    keys = [
        "candidate_id",
        "rank",
        "side",
        "quantity_mwh",
        "limit_price_eur_mwh",
        "expected_profit_proxy_eur",
        "worst_case_profit_proxy_eur",
        "rough_expected_profit_eur",
        "worst_case_profit_eur",
        "accepted",
        "reason_codes",
        "risk_flags",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def _frontier_feedback_prompt(
    persona: Persona,
    decision: LLMBidDecision,
    records: list[ToolCallRecord],
    tick: TickContext,
) -> str:
    """Reprompt content for cp13: surface the rejection reason and the best accepted
    candidate so the agent refines toward the aggressive feasible frontier instead of
    abstaining when its bid is not yet simulator-backed."""
    from heimdall_ai_society._candidates import _rank_seeded_candidates
    from heimdall_ai_society._trace_helpers import _latest_rejected_required
    required = _required_simulation_tool(persona)
    ranked = _rank_seeded_candidates(records, tick)
    accepted = [row for row in ranked if row.get("accepted") is True]
    parts: list[str] = []
    if decision.action == "bid" and decision.limit_price_eur_mwh is not None:
        rejected = _latest_rejected_required(required, records)
        wcp = rejected.result.get("worst_case_profit_eur") if rejected is not None else None
        reasons = rejected.result.get("reason_codes", []) if rejected is not None else []
        detail = ""
        if wcp is not None:
            detail += f" worst_case_profit_eur={wcp}"
        if reasons:
            detail += f" reasons={reasons}"
        parts.append(
            f"Your proposed bid side={decision.side} quantity_mwh={decision.quantity_mwh} "
            f"limit_price_eur_mwh={decision.limit_price_eur_mwh} is not backed by an accepted {required} result"
            f"{(' (' + detail.strip() + ')') if detail else ''}."
        )
    if accepted:
        best = accepted[0].get("arguments") or {}
        parts.append(
            f"The best ACCEPTED candidate so far is side={best.get('side')} "
            f"quantity_mwh={best.get('quantity_mwh')} limit_price_eur_mwh={best.get('limit_price_eur_mwh')} "
            f"(worst_case_profit_eur={accepted[0].get('worst_case_profit_eur')})."
        )
        parts.append(
            f"Try to beat it: call {required} with a more aggressive price/quantity, keep only candidates with "
            "accepted=true, then call propose_action with the accepted candidate that has the highest "
            "worst_case_profit_eur. Do not watch or abstain while an accepted candidate exists."
        )
    else:
        parts.append(
            f"No accepted candidate yet. Call {required} with a more conservative price and/or smaller quantity "
            "to satisfy the worst-case-profit floor, then call propose_action with an exact accepted candidate. "
            "Only watch or abstain if repeated simulation finds no accepted candidate."
        )
    return " ".join(parts)


def _required_simulation_tool(persona: Persona) -> str:
    return {
        "ev": "simulate_ev_bid",
        "wind": "simulate_wind_bid",
        "generator": "simulate_generator_bid",
        "retailer": "simulate_retailer_bid",
        "renewables": "simulate_renewables_bid",
    }.get(persona.archetype.value, "simulate_bid")


def _feasibility_tool(persona: Persona) -> str:
    return {
        "ev": "get_ev_bid_feasibility",
        "wind": "get_wind_bid_feasibility",
        "generator": "get_generator_bid_feasibility",
        "retailer": "get_retailer_bid_feasibility",
        "renewables": "get_renewables_bid_feasibility",
    }.get(persona.archetype.value, "get_bid_feasibility")


def _required_feasibility_tool(persona: Persona) -> str:
    archetype = persona.archetype.value
    if archetype == "p2h":
        return "get_bid_feasibility"
    return f"get_{archetype}_bid_feasibility"


def _final_action_instruction(objective: str) -> str:
    if objective == "unverified_bid_seeking":
        return "Use action=abstain only when the visible context gives no directional edge or your confidence is very low; for abstain/watch omit bid fields."
    return "Use action=abstain when no safe bid is justified; for abstain/watch omit bid fields."


def _objective_instruction(objective: str) -> str:
    if objective == "unverified_bid_seeking":
        return (
            "This is an unverified bid-seeking ablation: no simulator or verifier can be consulted before submission. "
            "Prefer bidding when your own expected edge from visible forecast/context is positive; use watch when opportunity exists but direction or price is unclear; abstain only when there is no directional edge."
        )
    if objective == "bid_seeking":
        return (
            "This is a bid-seeking ablation: do not abstain just because activation is uncertain. "
            "If the candidate edge is positive, probe a small 2 MWh P2H bid with simulate_bid and bid if the simulator accepts it; otherwise watch."
        )
    if objective == "stress_test":
        return (
            "This is a stress-test ablation: intentionally probe borderline bids with small quantities to reveal verifier behavior. "
            "Never bypass simulate_bid for final bids."
        )
    return "This is a conservative worth-bidding run: prefer watch over abstain when the MTU may become useful."
