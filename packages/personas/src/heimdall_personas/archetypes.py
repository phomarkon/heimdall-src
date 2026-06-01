"""Persona archetype defaults. Stub: capacity/storage numbers are deliberately
round and will be replaced by PyPSA-Eur-Sec-derived values in Week 2 (§4.8).
"""

from __future__ import annotations

from heimdall_contracts import Persona, PersonaArchetype, RiskAttitude

ARCHETYPE_DEFAULTS: dict[PersonaArchetype, dict] = {
    PersonaArchetype.WIND: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "medium",
        "info_latency_min": 5,
        "capacity_mw": 50.0,
        "storage_mwh": None,
        "forecaster_id": "F9",
        "llm_id": "L5",
    },
    PersonaArchetype.EV: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "low",
        "info_latency_min": 15,
        "capacity_mw": 5.0,
        "storage_mwh": 20.0,
        "forecaster_id": "F1",
        "llm_id": "L1",
    },
    PersonaArchetype.RETAILER: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 100.0,
        "storage_mwh": None,
        "forecaster_id": "F11",
        "llm_id": "L4",
    },
    PersonaArchetype.P2H: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 50.0,
        "storage_mwh": 100.0,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.GENERATOR: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 200.0,
        "storage_mwh": None,
        "forecaster_id": "F7",
        "llm_id": "L5",
    },
    PersonaArchetype.RENEWABLES: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "medium",
        "info_latency_min": 5,
        "capacity_mw": 120.0,
        "storage_mwh": None,
        "forecaster_id": "F9",
        "llm_id": "L5",
    },
    PersonaArchetype.ARBITRAGEUR: {
        "risk_attitude": RiskAttitude.SEEKING,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 25.0,
        "storage_mwh": None,
        "forecaster_id": "F9",
        "llm_id": "L5",
    },
    PersonaArchetype.MARKET_MECHANICS_EXPERT: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.IMBALANCE_ANALYTICS_EXPERT: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.TRADING_RISK_MONITOR: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.GRID_CONSTRAINT_ANALYST: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.OUTAGE_IMPACT_SCORER: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.LIMIT_PRICE_SPECIALIST: {
        "risk_attitude": RiskAttitude.NEUTRAL,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.CANDIDATE_SIZING_SPECIALIST: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.UNCERTAINTY_AUDITOR: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
    PersonaArchetype.DECISION_AUDITOR: {
        "risk_attitude": RiskAttitude.AVERSE,
        "sophistication": "high",
        "info_latency_min": 0,
        "capacity_mw": 0.0,
        "storage_mwh": None,
        "forecaster_id": "F8",
        "llm_id": "L5",
    },
}


def default_persona(agent_id: str, archetype: PersonaArchetype) -> Persona:
    """Construct a default Persona for the given archetype. Used by the
    market simulator stub when no scenario file is provided."""
    return Persona(agent_id=agent_id, archetype=archetype, **ARCHETYPE_DEFAULTS[archetype])
