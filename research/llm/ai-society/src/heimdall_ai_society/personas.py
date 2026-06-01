from __future__ import annotations

import random

from heimdall_contracts import Persona, PersonaArchetype, RiskAttitude
from heimdall_personas.archetypes import default_persona

_ARCHETYPE_CYCLE = (
    PersonaArchetype.WIND,
    PersonaArchetype.EV,
    PersonaArchetype.RETAILER,
    PersonaArchetype.P2H,
    PersonaArchetype.GENERATOR,
    PersonaArchetype.ARBITRAGEUR,
)


def build_personas(
    agent_count: int,
    archetype_cycle: list[PersonaArchetype] | None = None,
    *,
    profile: str = "default",
    seed: int = 42,
) -> list[Persona]:
    if profile == "risk_trio":
        risks = [RiskAttitude.AVERSE, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING]
        return [
            default_persona(f"agent-{idx:03d}", PersonaArchetype.P2H).model_copy(update={"risk_attitude": risks[idx % len(risks)]})
            for idx in range(agent_count)
        ]
    if profile == "price_styles":
        risks = [RiskAttitude.AVERSE, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING]
        latencies = [0, 0, 0, 5, 15]
        return [
            default_persona(f"agent-{idx:03d}", PersonaArchetype.P2H).model_copy(
                update={"risk_attitude": risks[idx % len(risks)], "info_latency_min": latencies[idx % len(latencies)]}
            )
            for idx in range(agent_count)
        ]
    if profile == "side_specialists":
        return [
            default_persona(f"agent-{idx:03d}", PersonaArchetype.P2H).model_copy(
                update={"risk_attitude": RiskAttitude.SEEKING if idx % 3 != 2 else RiskAttitude.AVERSE}
            )
            for idx in range(agent_count)
        ]
    if profile == "committee":
        risks = [RiskAttitude.AVERSE, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING, RiskAttitude.AVERSE]
        return [
            default_persona(f"agent-{idx:03d}", PersonaArchetype.P2H).model_copy(update={"risk_attitude": risks[idx % len(risks)]})
            for idx in range(agent_count)
        ]
    if profile == "random_p2h":
        rng = random.Random(seed)
        risks = [RiskAttitude.AVERSE, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING]
        forecasters = ["F0", "F8", "F11", "AR1"]
        return [
            default_persona(f"agent-{idx:03d}", PersonaArchetype.P2H).model_copy(
                update={
                    "risk_attitude": rng.choice(risks),
                    "info_latency_min": rng.choice([0, 5, 15, 30]),
                    "capacity_mw": rng.choice([25.0, 50.0, 75.0]),
                    "storage_mwh": rng.choice([50.0, 100.0, 150.0]),
                    "forecaster_id": rng.choice(forecasters),
                }
            )
            for idx in range(agent_count)
        ]
    if profile == "mixed_advisory":
        cycle = (
            PersonaArchetype.P2H,
            PersonaArchetype.WIND,
            PersonaArchetype.RETAILER,
            PersonaArchetype.GENERATOR,
            PersonaArchetype.ARBITRAGEUR,
            PersonaArchetype.EV,
        )
        return [default_persona(f"agent-{idx:03d}", cycle[idx % len(cycle)]) for idx in range(agent_count)]
    if profile == "diverse_action":
        cycle = (
            PersonaArchetype.P2H,
            PersonaArchetype.WIND,
            PersonaArchetype.GENERATOR,
            PersonaArchetype.RENEWABLES,
            PersonaArchetype.RETAILER,
        )
        return [default_persona(f"agent-{idx:03d}", cycle[idx % len(cycle)]) for idx in range(agent_count)]
    if profile == "diverse_expert_action":
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F9", "capacity_mw": 80.0, "info_latency_min": 5}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F1", "capacity_mw": 20.0, "storage_mwh": 80.0, "info_latency_min": 15}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F10", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F12", "capacity_mw": 180.0, "info_latency_min": 5}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "all_archetypes_v1":
        # One focal verifier-guarded P2H agent (50 MW, real PyPSA-Eur-Sec spec) plus one of
        # each other action archetype with its own simulator path. Matches the thesis framing:
        # a focal P2H market-maker inside a heterogeneous society of BRP competitors. P2H is
        # placed first so a 1-agent slice is the focal alone. Capacity values mirror
        # diverse_expert_action for consistency; only P2H is capacity-grounded for the oracle.
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0, "info_latency_min": 0}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F1", "capacity_mw": 20.0, "storage_mwh": 80.0, "info_latency_min": 15}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F9", "capacity_mw": 80.0, "info_latency_min": 5}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0, "info_latency_min": 0}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "all_archetypes_double_v1":
        # Society B (12 agents): two of each action archetype, paired with DELIBERATELY
        # CONTRASTING risk attitude / forecaster / size / info-latency. The focal P2H
        # (50 MW, averse) stays first so the capacity oracle's focal is unambiguous; its
        # twin is a smaller, risk-seeking P2H competitor. Doubling the population with
        # heterogeneous aggressiveness widens the space of decision contexts and rationales
        # — the regime where a fixed hand-written rationale template can no longer
        # enumerate every case, so the LLM's zero-template-engineering auditability has the
        # most room to show, and ungrounded confabulation has more entities to invent.
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0, "info_latency_min": 0}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F1", "capacity_mw": 20.0, "storage_mwh": 80.0, "info_latency_min": 15}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F9", "capacity_mw": 80.0, "info_latency_min": 5}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0, "info_latency_min": 0}),
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F8", "capacity_mw": 30.0, "storage_mwh": 60.0, "info_latency_min": 5}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 15.0, "storage_mwh": 50.0, "info_latency_min": 5}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F10", "capacity_mw": 60.0, "info_latency_min": 10}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F12", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F10", "capacity_mw": 100.0, "info_latency_min": 10}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "all_archetypes_double_homo":
        # LOW-heterogeneity control for the count-controlled test (vs all_archetypes_double_v1).
        # Same N=12 and same archetype mix (2x each of the 6 action archetypes), but the two copies
        # of each archetype are IDENTICAL — no contrasting risk/forecaster/size. So a B-vs-this
        # comparison isolates persona/aggressiveness HETEROGENEITY at fixed agent-count + archetype
        # mix, de-confounding the "heterogeneity vs count" question for the guarded/det ratio.
        base = build_personas(6, profile="all_archetypes_v1", seed=seed)
        out = list(base)
        for i, persona in enumerate(base):
            out.append(persona.model_copy(update={"agent_id": f"agent-{6 + i:03d}"}))
        return out[:agent_count]
    if profile == "all_archetypes_plus_info_v1":
        # Society C (9 agents): the 6-archetype action society + 3 information-specialist
        # agents (market-mechanics, imbalance-analytics, trading-risk). The info agents
        # publish qualitative analysis the action agents can cite — a division of labour
        # the deterministic pipeline has no analogue for. Tests whether peer-sourced
        # evidence strengthens grounded auditability and produces rationale content no
        # fixed template enumerates (the open-endedness axis). Focal P2H stays first
        # (grounded-oracle focal); the experts are appended last.
        base = build_personas(6, profile="all_archetypes_v1", seed=seed)
        experts = [
            default_persona("agent-006", PersonaArchetype.MARKET_MECHANICS_EXPERT),
            default_persona("agent-007", PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT),
            default_persona("agent-008", PersonaArchetype.TRADING_RISK_MONITOR),
        ]
        return [*base, *experts][:agent_count]
    if profile == "all_archetypes_double_plus_info_v1":
        # Society D (15 agents): the 12-agent DOUBLED action society (2x each of the 6
        # action archetypes, with deliberately contrasting risk attitude / forecaster /
        # size) PLUS the 3 information-specialist agents (market-mechanics, imbalance-
        # analytics, trading-risk). Combines maximal within-archetype heterogeneity with a
        # division-of-labour info tier whose qualitative analysis the action agents can
        # cite. Focal P2H (50 MW, averse) stays first so the capacity oracle's focal is
        # unambiguous; the experts are appended last.
        base = build_personas(12, profile="all_archetypes_double_v1", seed=seed)
        experts = [
            default_persona("agent-012", PersonaArchetype.MARKET_MECHANICS_EXPERT),
            default_persona("agent-013", PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT),
            default_persona("agent-014", PersonaArchetype.TRADING_RISK_MONITOR),
        ]
        return [*base, *experts][:agent_count]
    if profile == "action_core_8":
        templates = [
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F10", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F12", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0, "info_latency_min": 0}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F1", "capacity_mw": 20.0, "storage_mwh": 80.0, "info_latency_min": 15}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "market_expert_panel":
        cycle = (
            PersonaArchetype.MARKET_MECHANICS_EXPERT,
            PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT,
            PersonaArchetype.TRADING_RISK_MONITOR,
        )
        return [default_persona(f"agent-{idx:03d}", cycle[idx % len(cycle)]) for idx in range(agent_count)]
    if profile == "p2h_specialist_v2":
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "capacity_mw": 50.0, "storage_mwh": 100.0, "forecaster_id": "F8"}),
            (PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT, {}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "ev_specialist_v2":
        templates = [
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.AVERSE, "capacity_mw": 8.0, "storage_mwh": 20.0, "forecaster_id": "F8", "info_latency_min": 5}),
            (PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT, {}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "p2h_info_then_action_v2":
        templates = [
            (PersonaArchetype.MARKET_MECHANICS_EXPERT, {}),
            (PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT, {}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {}),
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "capacity_mw": 50.0, "storage_mwh": 100.0, "forecaster_id": "F8"}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "ev_info_then_action_v2":
        templates = [
            (PersonaArchetype.MARKET_MECHANICS_EXPERT, {}),
            (PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT, {}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.AVERSE, "capacity_mw": 8.0, "storage_mwh": 20.0, "forecaster_id": "F8", "info_latency_min": 5}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "market_experts_plus_action_core_6":
        templates = [
            (PersonaArchetype.MARKET_MECHANICS_EXPERT, {}),
            (PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT, {}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F10", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F12", "capacity_mw": 180.0, "info_latency_min": 5}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "action_core_8_plus_market_expert":
        base = build_personas(8, profile="action_core_8", seed=seed)
        experts = [
            default_persona("agent-008", PersonaArchetype.MARKET_MECHANICS_EXPERT),
            default_persona("agent-009", PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT),
            default_persona("agent-010", PersonaArchetype.TRADING_RISK_MONITOR),
        ]
        return [*base, *experts][:agent_count]
    if profile in {
        "action_core_9_chair",
        "action_core_10_safety",
        "action_core_8_aggressive",
        "action_core_8_safety",
        "action_core_8_toolsplit",
    }:
        base = build_personas(8, profile="action_core_8", seed=seed)
        if profile == "action_core_8_aggressive":
            return [
                persona.model_copy(
                    update={"risk_attitude": RiskAttitude.SEEKING, "info_latency_min": 0}
                )
                for persona in base[:agent_count]
            ]
        if profile == "action_core_8_safety":
            return [
                persona.model_copy(
                    update={
                        "risk_attitude": RiskAttitude.AVERSE,
                        "info_latency_min": max(persona.info_latency_min, 5),
                    }
                )
                for persona in base[:agent_count]
            ]
        if profile == "action_core_8_toolsplit":
            forecasters = ["F7", "F8", "F9", "F10", "F11", "F3", "F8", "F1"]
            return [
                persona.model_copy(update={"forecaster_id": forecasters[idx % len(forecasters)]})
                for idx, persona in enumerate(base[:agent_count])
            ]
        extras = [
            default_persona("agent-008", PersonaArchetype.ARBITRAGEUR).model_copy(
                update={
                    "risk_attitude": RiskAttitude.AVERSE,
                    "forecaster_id": "F8",
                    "capacity_mw": 0.0,
                }
            ),
            default_persona("agent-009", PersonaArchetype.P2H).model_copy(
                update={
                    "risk_attitude": RiskAttitude.AVERSE,
                    "forecaster_id": "F8",
                    "capacity_mw": 25.0,
                    "storage_mwh": 50.0,
                    "info_latency_min": 10,
                }
            ),
        ]
        return [*base, *extras][:agent_count]
    if profile == "balanced_intelligence":
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F1", "capacity_mw": 20.0, "storage_mwh": 80.0, "info_latency_min": 15}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F9", "capacity_mw": 80.0, "info_latency_min": 5}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 220.0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 160.0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F11", "capacity_mw": 130.0}),
            (PersonaArchetype.ARBITRAGEUR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F8", "capacity_mw": 25.0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F3", "capacity_mw": 80.0, "info_latency_min": 10}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F10", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.WIND, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F8", "capacity_mw": 60.0, "info_latency_min": 5}),
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F3", "capacity_mw": 35.0, "storage_mwh": 70.0, "info_latency_min": 5}),
            (PersonaArchetype.ARBITRAGEUR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 0.0}),
        ]
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    if profile == "crowd_intelligence":
        rng = random.Random(seed)
        base = build_personas(max(agent_count, 12), profile="balanced_intelligence", seed=seed)
        risks = [RiskAttitude.AVERSE, RiskAttitude.NEUTRAL, RiskAttitude.SEEKING]
        forecasters = ["F0", "F3", "F8", "F9", "F11"]
        return [
            base[idx % len(base)].model_copy(
                update={
                    "agent_id": f"agent-{idx:03d}",
                    "risk_attitude": rng.choice(risks),
                    "info_latency_min": rng.choice([0, 5, 10, 15, 30]),
                    "capacity_mw": max(0.0, base[idx % len(base)].capacity_mw * rng.choice([0.75, 1.0, 1.25])),
                    "forecaster_id": rng.choice(forecasters),
                }
            )
            for idx in range(agent_count)
        ]
    if profile == "info_specialists_v1":
        cycle = _INFO_SPECIALISTS
        return [default_persona(f"agent-{idx:03d}", cycle[idx % len(cycle)]) for idx in range(agent_count)]
    if profile == "action_core_8_plus_info_specialists":
        base = build_personas(8, profile="action_core_8", seed=seed)
        experts = [default_persona(f"agent-{idx + 8:03d}", archetype) for idx, archetype in enumerate(_INFO_SPECIALISTS)]
        return [*base, *experts][:agent_count]
    if profile == "jao_grid_v1":
        base = build_personas(8, profile="action_core_8", seed=seed)
        experts = [
            default_persona("agent-008", PersonaArchetype.GRID_CONSTRAINT_ANALYST),
            default_persona("agent-009", PersonaArchetype.OUTAGE_IMPACT_SCORER),
            default_persona("agent-010", PersonaArchetype.UNCERTAINTY_AUDITOR),
            default_persona("agent-011", PersonaArchetype.DECISION_AUDITOR),
        ]
        return [*base, *experts][:agent_count]
    if profile in {"mixed_expert_18_sideaware", "mixed_expert_20_sideaware"}:
        templates = [
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 50.0, "storage_mwh": 100.0, "info_latency_min": 0}),
            (PersonaArchetype.P2H, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F3", "capacity_mw": 35.0, "storage_mwh": 70.0, "info_latency_min": 5}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F3", "capacity_mw": 12.0, "storage_mwh": 40.0, "info_latency_min": 5}),
            (PersonaArchetype.EV, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 8.0, "storage_mwh": 20.0, "info_latency_min": 10}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F8", "capacity_mw": 220.0, "info_latency_min": 0}),
            (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F7", "capacity_mw": 180.0, "info_latency_min": 5}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F8", "capacity_mw": 160.0, "info_latency_min": 0}),
            (PersonaArchetype.RENEWABLES, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F7", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 130.0, "info_latency_min": 0}),
            (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.NEUTRAL, "forecaster_id": "F3", "capacity_mw": 100.0, "info_latency_min": 10}),
            (PersonaArchetype.GRID_CONSTRAINT_ANALYST, {"forecaster_id": "F8"}),
            (PersonaArchetype.OUTAGE_IMPACT_SCORER, {"forecaster_id": "F8"}),
            (PersonaArchetype.LIMIT_PRICE_SPECIALIST, {"forecaster_id": "F8"}),
            (PersonaArchetype.CANDIDATE_SIZING_SPECIALIST, {"forecaster_id": "F7"}),
            (PersonaArchetype.UNCERTAINTY_AUDITOR, {"forecaster_id": "F3"}),
            (PersonaArchetype.DECISION_AUDITOR, {"forecaster_id": "F3"}),
            (PersonaArchetype.TRADING_RISK_MONITOR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F7"}),
            (PersonaArchetype.ARBITRAGEUR, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F8", "capacity_mw": 0.0}),
        ]
        if profile == "mixed_expert_20_sideaware":
            templates.extend(
                [
                    (PersonaArchetype.GENERATOR, {"risk_attitude": RiskAttitude.SEEKING, "forecaster_id": "F9", "capacity_mw": 120.0, "info_latency_min": 0}),
                    (PersonaArchetype.RETAILER, {"risk_attitude": RiskAttitude.AVERSE, "forecaster_id": "F3", "capacity_mw": 70.0, "info_latency_min": 15}),
                ]
            )
        return [
            default_persona(f"agent-{idx:03d}", archetype).model_copy(update=updates)
            for idx, (archetype, updates) in ((idx, templates[idx % len(templates)]) for idx in range(agent_count))
        ]
    cycle = tuple(archetype_cycle or _ARCHETYPE_CYCLE)
    return [
        default_persona(f"agent-{idx:03d}", cycle[idx % len(cycle)])
        for idx in range(agent_count)
    ]


_INFO_SPECIALISTS = (
    PersonaArchetype.GRID_CONSTRAINT_ANALYST,
    PersonaArchetype.OUTAGE_IMPACT_SCORER,
    PersonaArchetype.LIMIT_PRICE_SPECIALIST,
    PersonaArchetype.CANDIDATE_SIZING_SPECIALIST,
    PersonaArchetype.UNCERTAINTY_AUDITOR,
    PersonaArchetype.DECISION_AUDITOR,
)
