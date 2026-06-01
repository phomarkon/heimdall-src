from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pandas as pd
from heimdall_contracts import Persona

from heimdall_ai_society.schemas import (
    DeliberationNote,
    LLMBidDecision,
    PeerResponse,
    ToolCallRecord,
)
from heimdall_ai_society.tool_policy import policy_for_persona
from heimdall_ai_society._tool_specs import (
    openai_tool_specs,
    retrieve_knowledge_tool_spec,
)
from heimdall_ai_society._tool_market import MarketToolsMixin
from heimdall_ai_society._tool_simulation import (
    SimulationToolsMixin,
    _fallback_real_asset_spec,
    _stable_result_hash,
)
from heimdall_ai_society._tool_candidates import CandidateToolsMixin
from packages.pypsa_adapter import build_tiny_dk_network, extract_heimdall_scenario, solve_network
from packages.simulator import (
    Bid,
    ScenarioAssetStateStore,
    SimulatorAssetState,
    RealAssetState,
    spec_from_scenario,
)
from packages.simulator.agent_tool import AgentMFRRTool, AgentMFRRToolResult
from packages.simulator.forecast import ForecastMarketState
from packages.simulator.mfrr_engine import CalibratedMFRRPriceModel


def decision_to_bid(
    persona: Persona,
    decision: LLMBidDecision,
    forecast: ForecastMarketState,
) -> Bid | None:
    if decision.action != "bid":
        return None
    if decision.side is None or decision.quantity_mwh is None or decision.limit_price_eur_mwh is None:
        return None
    delivery = forecast.delivery_datetime()
    max_quantity = max(0.25, persona.capacity_mw * 0.25)
    return Bid(
        agent_id=persona.agent_id,
        asset_id=forecast.zone,
        zone=forecast.zone,  # type: ignore[arg-type]
        utc_timestamp=delivery,
        side=decision.side,
        quantity_mwh=min(float(decision.quantity_mwh), max_quantity),
        limit_price_eur_mwh=float(decision.limit_price_eur_mwh),
        submitted_at_utc=delivery - timedelta(minutes=50),
    )


def mock_verify(decision: LLMBidDecision) -> tuple[bool | None, list[str]]:
    if decision.action in {"abstain", "watch"}:
        return None, []
    if decision.side is None or decision.quantity_mwh is None or decision.limit_price_eur_mwh is None:
        return False, ["invalid_bid_fields"]
    return True, []


def build_simulator_tool(*, tau_eur: float = -100.0) -> AgentMFRRTool:
    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")
    scenario = extract_heimdall_scenario(network)
    return AgentMFRRTool(
        scenario,
        CalibratedMFRRPriceModel.fit(_synthetic_price_history(), min_samples=4),
        tau_eur=tau_eur,
    )


def simulate_bid(
    tool: AgentMFRRTool,
    bid: Bid,
    forecast: ForecastMarketState,
) -> AgentMFRRToolResult:
    return tool.simulate_bid_from_forecast(
        bid,
        forecast,
        asset_state=SimulatorAssetState.for_asset(
            bid.asset_id,
            electric_power_mw=8.0,
        ),
    )


def commit_asset_state_from_record(
    *,
    state_store: ScenarioAssetStateStore,
    simulator_tool: AgentMFRRTool | None,
    persona: Persona,
    forecast: ForecastMarketState,
    record: ToolCallRecord | None,
) -> None:
    if record is None:
        return
    next_state = record.result.get("next_state")
    archetype = record.result.get("archetype")
    if not isinstance(next_state, dict) or archetype not in {"ev", "wind", "generator", "renewables", "retailer"}:
        return
    if simulator_tool is not None and hasattr(simulator_tool, "_scenario"):
        try:
            spec = spec_from_scenario(simulator_tool._scenario, archetype=archetype, zone=forecast.zone)
        except (KeyError, AttributeError):
            spec = _fallback_real_asset_spec(persona, str(archetype))
    else:
        spec = _fallback_real_asset_spec(persona, str(archetype))
    state_store.commit(
        agent_id=persona.agent_id,
        spec=spec,
        state=RealAssetState(**next_state),
    )


class AgentToolExecutor(MarketToolsMixin, SimulationToolsMixin, CandidateToolsMixin):
    def __init__(
        self,
        *,
        persona: Persona,
        forecast: ForecastMarketState,
        data_tools: Any | None,
        simulator_tool: AgentMFRRTool | None,
        asset_simulator_mode: str = "proxy",
        asset_proxy_style: str = "market",
        asset_state_store: ScenarioAssetStateStore | None = None,
        tool_cache: dict[tuple[str, str], ToolCallRecord] | None = None,
        candidate_diagnostics: list[ToolCallRecord] | None = None,
        retriever: Any | None = None,
        rag_top_k: int = 4,
        rag_max_chars: int = 700,
    ) -> None:
        self._persona = persona
        self._forecast = forecast
        self._data_tools = data_tools
        self._simulator_tool = simulator_tool
        self._asset_simulator_mode = _normalize_asset_simulator_mode(asset_simulator_mode)
        self._asset_proxy_style = asset_proxy_style
        self._asset_state_store = asset_state_store or ScenarioAssetStateStore.empty()
        self._policy = policy_for_persona(persona)
        self._tool_cache = tool_cache if tool_cache is not None else {}
        self._candidate_diagnostics = candidate_diagnostics or []
        self._retriever = retriever
        self._rag_top_k = rag_top_k
        self._rag_max_chars = rag_max_chars

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        cache_key = (self._cache_scope(name), self._cache_payload(name, arguments))
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]
        try:
            if not self._policy.allows(name):
                result = {
                    "ok": False,
                    "error_code": "tool_not_allowed_for_archetype",
                    "archetype": self._persona.archetype.value,
                    "tool": name,
                }
                return ToolCallRecord(name=name, arguments=arguments, ok=False, result=result)
            result = self._dispatch(name, arguments)
            record = ToolCallRecord(name=name, arguments=arguments, ok=bool(result.get("ok", True)), result=result)
            if name not in {"propose_bid", "propose_action", "propose_deliberation_note", "propose_peer_response"}:
                self._tool_cache[cache_key] = record
            return record
        except Exception as exc:
            return ToolCallRecord(name=name, arguments=arguments, ok=False, error=str(exc))

    def _cache_scope(self, name: str) -> str:
        shared_tools = {
            "run_forecaster",
            "run_activation_forecaster",
            "get_activation_context",
            "get_opportunity_context",
            "get_market_regime_context",
            "get_border_pressure",
            "get_grid_constraints",
            "get_outage_impact",
            "get_uncertainty_digest",
            "get_limit_price_guidance",
            "get_decision_trace_summary",
        }
        if name in shared_tools:
            return f"shared:{self._forecast.delivery_timestamp}:{self._forecast.issued_at}"
        return f"{self._persona.agent_id}:{self._forecast.delivery_timestamp}:{self._forecast.issued_at}"

    def _cache_payload(self, name: str, arguments: dict[str, Any]) -> str:
        return f"{name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "run_forecaster":
            return {
                "ok": True,
                "delivery_timestamp": self._forecast.delivery_timestamp,
                "zone": self._forecast.zone,
                "mfrr_up_interval_eur_mwh": list(self._forecast.interval_for_side("up")),
                "mfrr_down_interval_eur_mwh": list(self._forecast.interval_for_side("down")),
                "forecast_hash": self._forecast.result_hash,
            }
        if name == "run_activation_forecaster":
            return self._activation_forecast()
        if name == "retrieve_knowledge":
            return self._retrieve_knowledge(arguments)
        if name == "get_activation_context":
            if self._data_tools is not None:
                method: Callable[..., dict[str, Any]] | None = getattr(self._data_tools, name, None)
                if method is not None:
                    return method(**arguments)
            return self._fallback_opportunity_context()
        if name == "get_opportunity_context":
            if self._data_tools is not None:
                method = getattr(self._data_tools, "get_activation_context", None)
                if method is not None:
                    context = method(**arguments)
                    return {**context, "kind": "opportunity_context"}
            return self._fallback_opportunity_context(kind="opportunity_context")
        if name == "get_market_regime_context":
            return self._market_regime_context(arguments)
        if name == "get_grid_constraints":
            return self._grid_constraints(arguments)
        if name == "get_border_pressure":
            return self._border_pressure(arguments)
        if name == "get_outage_impact":
            return self._outage_impact(arguments)
        if name == "get_limit_price_guidance":
            return self._limit_price_guidance(arguments)
        if name == "get_uncertainty_digest":
            return self._uncertainty_digest()
        if name == "get_candidate_rejection_summary":
            return self._candidate_rejection_summary()
        if name == "get_candidate_sizing_guidance":
            return self._candidate_sizing_guidance(str(arguments.get("archetype") or self._persona.archetype.value))
        if name == "get_spread_opportunity":
            context = self._fallback_opportunity_context(kind="spread_opportunity")
            context["authority"] = "advisory"
            return context
        if name == "get_decision_trace_summary":
            return self._decision_trace_summary()
        if name == "get_ev_bid_feasibility":
            return self._ev_bid_feasibility(arguments)
        if name == "get_wind_bid_feasibility":
            return self._wind_bid_feasibility(arguments)
        if name == "get_generator_bid_feasibility":
            return self._generator_bid_feasibility(arguments)
        if name == "get_retailer_bid_feasibility":
            return self._retailer_bid_feasibility(arguments)
        if name == "get_renewables_bid_feasibility":
            return self._renewables_bid_feasibility(arguments)
        if name == "get_bid_feasibility":
            return self._p2h_bid_feasibility(arguments)
        if name == "simulate_ev_bid":
            return self._simulate_asset_bid(arguments, archetype="ev")
        if name == "simulate_wind_bid":
            return self._simulate_asset_bid(arguments, archetype="wind")
        if name == "simulate_generator_bid":
            return self._simulate_asset_bid(arguments, archetype="generator")
        if name == "simulate_retailer_bid":
            return self._simulate_asset_bid(arguments, archetype="retailer")
        if name == "simulate_renewables_bid":
            return self._simulate_asset_bid(arguments, archetype="renewables")
        if name == "simulate_bid":
            return self._simulate_p2h_bid(arguments)
        if name in {"propose_bid", "propose_action"}:
            decision = LLMBidDecision.model_validate(arguments)
            return {"ok": True, "decision": decision.model_dump(mode="json")}
        if name == "propose_deliberation_note":
            note = DeliberationNote.model_validate(arguments)
            return {"ok": True, "note": note.model_dump(mode="json")}
        if name == "propose_peer_response":
            response = PeerResponse.model_validate(arguments)
            return {"ok": True, "response": response.model_dump(mode="json")}
        if self._data_tools is None:
            return {"ok": False, "error_code": "real_data_tools_unavailable"}
        method_fn: Callable[..., dict[str, Any]] | None = getattr(self._data_tools, name, None)
        if method_fn is None:
            return {"ok": False, "error_code": "unknown_tool"}
        return method_fn(**arguments)


def decision_from_tool_calls(records: list[ToolCallRecord]) -> LLMBidDecision | None:
    for record in reversed(records):
        if record.name in {"propose_bid", "propose_action"} and record.ok and "decision" in record.result:
            return LLMBidDecision.model_validate(record.result["decision"])
    return None


def _normalize_asset_simulator_mode(mode: str) -> str:
    from packages.simulator import SCENARIO_ENVELOPE_BACKEND
    if mode == "real":
        return SCENARIO_ENVELOPE_BACKEND
    return mode


def _synthetic_price_history() -> pd.DataFrame:
    rows = []
    for idx, volume_mwh in enumerate([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]):
        minute = (idx % 4) * 15
        hour = idx // 4
        rows.append(
            {
                "utc_timestamp": f"2025-03-04T0{hour}:{minute:02d}:00Z",
                "zone": "DK1",
                "satisfied_demand_mw": volume_mwh / 0.25,
                "imbalance_price_eur_mwh": 50.0 + 8.0 + 2.0 * volume_mwh,
                "spot_price_eur_mwh": 50.0,
                "mfrr_marginal_price_up_eur_mwh": 50.0 + 8.0 + 2.0 * volume_mwh,
                "mfrr_marginal_price_down_eur_mwh": 50.0,
            }
        )
    return pd.DataFrame(rows)
