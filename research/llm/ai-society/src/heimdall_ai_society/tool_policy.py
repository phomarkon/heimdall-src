from __future__ import annotations

from dataclasses import dataclass

from heimdall_contracts import Persona, PersonaArchetype

CONTEXT_TOOLS = {
    "get_last_prices",
    "get_last_loads",
    "get_last_generation",
    "get_crossborder_flows",
    "get_weather_today",
    "get_weather_forecast",
    "get_outages",
    "run_forecaster",
    "run_activation_forecaster",
    "get_activation_context",
    "get_opportunity_context",
    "get_market_regime_context",
    "get_grid_constraints",
    "get_border_pressure",
    "get_outage_impact",
    "get_limit_price_guidance",
    "get_uncertainty_digest",
    "get_candidate_rejection_summary",
    "get_candidate_sizing_guidance",
    "get_decision_trace_summary",
    "retrieve_knowledge",
    "propose_action",
    "propose_bid",
    "propose_deliberation_note",
    "propose_peer_response",
}


@dataclass(frozen=True)
class ArchetypeToolPolicy:
    archetype: PersonaArchetype
    allowed_tools: frozenset[str]
    can_submit_bid: bool
    bid_requires_authoritative_simulation: bool

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


def policy_for_persona(persona: Persona) -> ArchetypeToolPolicy:
    archetype = persona.archetype
    extra: set[str] = set()
    can_submit = False
    requires_authoritative = False

    if archetype == PersonaArchetype.P2H:
        extra = {"get_bid_feasibility", "simulate_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.EV:
        extra = {"get_ev_bid_feasibility", "simulate_ev_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.WIND:
        extra = {"get_wind_bid_feasibility", "simulate_wind_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.GENERATOR:
        extra = {"get_generator_bid_feasibility", "simulate_generator_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.RENEWABLES:
        extra = {"get_renewables_bid_feasibility", "simulate_renewables_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.RETAILER:
        extra = {"get_retailer_bid_feasibility", "simulate_retailer_bid"}
        can_submit = True
        requires_authoritative = True
    elif archetype == PersonaArchetype.ARBITRAGEUR:
        extra = {"get_spread_opportunity"}
    elif archetype in {
        PersonaArchetype.MARKET_MECHANICS_EXPERT,
        PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT,
        PersonaArchetype.TRADING_RISK_MONITOR,
        PersonaArchetype.GRID_CONSTRAINT_ANALYST,
        PersonaArchetype.OUTAGE_IMPACT_SCORER,
        PersonaArchetype.LIMIT_PRICE_SPECIALIST,
        PersonaArchetype.CANDIDATE_SIZING_SPECIALIST,
        PersonaArchetype.UNCERTAINTY_AUDITOR,
        PersonaArchetype.DECISION_AUDITOR,
    }:
        extra = {"get_spread_opportunity"}

    return ArchetypeToolPolicy(
        archetype=archetype,
        allowed_tools=frozenset(CONTEXT_TOOLS | extra),
        can_submit_bid=can_submit,
        bid_requires_authoritative_simulation=requires_authoritative,
    )
