"""OpenAI-compatible tool specification definitions for the agent society."""

from __future__ import annotations

from typing import Any


def openai_tool_specs() -> list[dict[str, Any]]:
    def spec(
        name: str,
        description: str,
        properties: dict[str, Any],
        *,
        required: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or list(properties),
                    "additionalProperties": False,
                },
            },
        }

    zone = {"type": "string", "enum": ["DK1", "DK2"]}
    hours = {"type": "integer", "minimum": 1, "maximum": 168}
    variables = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": [
                "temperature",
                "wind_speed",
                "wind_direction",
                "wind_gusts",
                "solar_radiation",
                "cloud_cover",
                "precipitation",
                "pressure",
                "humidity",
            ],
        },
    }
    intelligence_fields = {
        "watch_label": {"type": "string", "enum": ["must_watch", "watch", "ignore"]},
        "risk_label": {"type": "string", "enum": ["low", "medium", "high"]},
        "uncertainty_label": {"type": "string", "enum": ["low", "medium", "high"]},
        "opportunity_label": {"type": "string", "enum": ["none", "weak", "actionable"]},
        "watch_reasons": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "activation_risk",
                    "price_volatility",
                    "forecast_uncertainty",
                    "accepted_bid_available",
                    "verifier_rejection_cluster",
                    "cross_agent_disagreement",
                ],
            },
        },
        "priority_label": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "priority_score": {"type": "number", "minimum": 0, "maximum": 1},
        "operator_action": {
            "type": "string",
            "enum": ["ignore", "monitor", "inspect", "prepare_bid", "escalate"],
        },
        "priority_reason": {
            "type": "string",
            "enum": [
                "none",
                "activation",
                "profit_edge",
                "accepted_candidate",
                "uncertainty",
                "rejection_cluster",
                "cross_agent_disagreement",
            ],
        },
    }
    deliberation_note_fields = {
        "side_belief": {"type": "string", "enum": ["up", "down", "mixed", "none"]},
        "price_belief": {"type": "string", "minLength": 1},
        "uncertainty_label": {"type": "string", "enum": ["low", "medium", "high"]},
        "risk_concern": {"type": "string"},
        "requested_peer_id": {"type": "string"},
        "requested_archetype": {"type": "string"},
        "requested_tool": {"type": "string"},
        "requested_candidate": {"type": "object"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string", "minLength": 1},
    }
    peer_response_fields = {
        "target_agent_id": {"type": "string"},
        "agreement": {"type": "string", "enum": ["agree", "object", "uncertain"]},
        "suggested_side": {"type": "string", "enum": ["up", "down"]},
        "suggested_quantity_mwh": {"type": "number", "exclusiveMinimum": 0},
        "suggested_limit_price_eur_mwh": {"type": "number"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string", "minLength": 1},
    }
    return [
        spec("get_last_prices", "Recent market prices before the current tick cutoff.", {"hours": hours, "zone": zone, "price_type": {"type": "string", "enum": ["day_ahead", "imbalance", "mfrr_up", "mfrr_down"]}}),
        spec("get_last_loads", "Recent actual or forecast total load before the current tick cutoff.", {"hours": hours, "zone": zone, "kind": {"type": "string", "enum": ["actual", "forecast"]}}),
        spec("get_last_generation", "Recent generation by production family before the current tick cutoff.", {"hours": hours, "zone": zone, "generation_type": {"type": "string", "enum": ["all", "wind", "solar", "hydro", "thermal"]}}),
        spec("get_crossborder_flows", "Recent cross-border flows before the current tick cutoff.", {"hours": hours, "zone": zone, "counterparty": {"type": "string"}}),
        spec("get_weather_today", "Weather forecast rows for today up to the current tick.", {"zone": zone, "variables": variables}),
        spec("get_weather_forecast", "Weather forecast rows after the current tick.", {"zone": zone, "horizon_hours": {"type": "integer", "minimum": 1, "maximum": 168}, "variables": variables}),
        spec("get_outages", "Recent or active outage reports relevant to a bidding zone.", {"hours": hours, "zone": zone}),
        spec("run_forecaster", "Return the current Heimdall price forecast interval.", {}, required=[]),
        spec("run_activation_forecaster", "Return an advisory non-verifier activation direction/volume forecast.", {}, required=[]),
        spec("get_activation_context", "Return non-leaking historical/contextual activation priors for deciding whether this MTU is worth watching.", {"hours": hours, "zone": zone}),
        spec("get_opportunity_context", "Return generic non-leaking opportunity context combining activation priors and spread/volatility hints.", {"hours": hours, "zone": zone}),
        spec("get_market_regime_context", "Classify the current market regime from non-leaking price, activation, and forecast evidence.", {"hours": hours, "zone": zone}),
        spec("get_grid_constraints", "Summarize JAO CNEC/RAM/shadow-price grid constraints visible before the current tick cutoff.", {"hours": hours, "zone": zone}),
        spec("get_border_pressure", "Summarize recent cross-border flow pressure and unusual changes before the current tick cutoff.", {"hours": hours, "zone": zone, "counterparty": {"type": "string"}}),
        spec("get_outage_impact", "Score recent or active outage reports by relevance and unavailable capacity before the current tick cutoff.", {"hours": hours, "zone": zone}),
        spec("get_limit_price_guidance", "Return crossing-aware bid price guidance balancing clear probability and expected profit.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}}),
        spec("get_uncertainty_digest", "Summarize forecast width, side ambiguity, and price-edge uncertainty for this tick.", {}, required=[]),
        spec("get_candidate_rejection_summary", "Summarize seeded candidate rejection clusters visible to this agent.", {}, required=[]),
        spec("get_candidate_sizing_guidance", "Return archetype-specific quantity guidance for safe candidate probing.", {"archetype": {"type": "string", "enum": ["p2h", "ev", "wind", "generator", "retailer", "renewables"]}}),
        spec("get_decision_trace_summary", "Summarize why current candidates should bid, watch, or abstain from action/info-agent evidence.", {}, required=[]),
        spec("get_bid_feasibility", "P2H-only cheap advisory score for a candidate bid. This does not replace simulator verification.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_ev_bid_feasibility", "EV virtual-battery advisory score for a candidate mFRR bid.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_wind_bid_feasibility", "Wind-producer advisory score for a candidate mFRR bid. Not an authoritative simulator.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_generator_bid_feasibility", "Generator advisory score for a candidate mFRR bid. Not an authoritative simulator.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_retailer_bid_feasibility", "Retailer demand-response advisory score for a candidate mFRR bid. Not an authoritative simulator.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_renewables_bid_feasibility", "Aggregated renewables advisory score for a candidate mFRR bid. Not an authoritative simulator.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("get_spread_opportunity", "Arbitrageur context-only spread opportunity score.", {"hours": hours, "zone": zone}),
        spec("simulate_bid", "Evaluate a P2H candidate bid with the simulator/verifier before proposing it.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("simulate_ev_bid", "Evaluate an EV virtual-battery candidate bid with the configured proxy, scenario-envelope, or PyPSA-background backend.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("simulate_wind_bid", "Evaluate a wind-producer candidate bid with the configured proxy, scenario-envelope, or PyPSA-background backend.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("simulate_generator_bid", "Evaluate a generator candidate bid with the configured proxy, scenario-envelope, or PyPSA-background backend.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("simulate_retailer_bid", "Evaluate a retailer demand-response candidate bid with the configured proxy, scenario-envelope, or PyPSA-background backend.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec("simulate_renewables_bid", "Evaluate an aggregated renewables candidate bid with the configured proxy, scenario-envelope, or PyPSA-background backend.", {"side": {"type": "string", "enum": ["up", "down"]}, "quantity_mwh": {"type": "number", "exclusiveMinimum": 0}, "limit_price_eur_mwh": {"type": "number"}}),
        spec(
            "propose_deliberation_note",
            "Structured inquiry note for the deliberation board. Cite visible tool evidence and request one peer, archetype, tool, or candidate probe when useful.",
            deliberation_note_fields,
            required=["side_belief", "price_belief", "uncertainty_label", "rationale"],
        ),
        spec(
            "propose_peer_response",
            "Structured response to another agent's deliberation note. Agree, object, or mark uncertain; include candidate suggestions only when evidence supports them.",
            peer_response_fields,
            required=["agreement", "rationale"],
        ),
        spec(
            "propose_action",
            "Final market action proposal. For watch or abstain, omit side, quantity_mwh, and limit_price_eur_mwh. For bid, include all bid fields.",
            {
                "action": {"type": "string", "enum": ["bid", "watch", "abstain"]},
                "side": {"type": "string", "enum": ["up", "down"]},
                "quantity_mwh": {"type": "number", "exclusiveMinimum": 0},
                "limit_price_eur_mwh": {"type": "number"},
                "rationale": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                **intelligence_fields,
            },
            required=["action", "rationale", "confidence"],
        ),
        spec(
            "propose_bid",
            "Compatibility alias for final market action proposal. For watch or abstain, omit side, quantity_mwh, and limit_price_eur_mwh. For bid, include all bid fields.",
            {
                "action": {"type": "string", "enum": ["bid", "watch", "abstain"]},
                "side": {"type": "string", "enum": ["up", "down"]},
                "quantity_mwh": {"type": "number", "exclusiveMinimum": 0},
                "limit_price_eur_mwh": {"type": "number"},
                "rationale": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                **intelligence_fields,
            },
            required=["action", "rationale", "confidence"],
        ),
    ]


def retrieve_knowledge_tool_spec() -> dict[str, Any]:
    """OpenAI-tool spec for the RAG retrieval tool.

    Kept separate from :func:`openai_tool_specs` so it is only ever offered to
    agents in runs where RAG is enabled (clean no-RAG control). The ``as_of``
    cutoff is NOT a tool argument — the runner forces it to the current tick so
    the model cannot retrieve future or same-window outcomes.
    """
    return {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "Search the leak-safe knowledge base for prior context relevant to this decision: "
                "historical market-regime statistics from earlier days, lessons from past society runs, "
                "and timeless methodology. Only documents available on or before the current market tick "
                "are returned. Use it to anticipate the likely activation side, sizing, and risk before bidding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "k": {"type": "integer", "minimum": 1, "maximum": 12},
                    "kinds": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["historical_stats", "prior_run_lesson", "methodology"],
                        },
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
