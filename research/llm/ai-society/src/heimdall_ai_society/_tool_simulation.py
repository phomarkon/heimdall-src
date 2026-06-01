"""Simulation gateway tool methods (mixin for AgentToolExecutor)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord

from packages.simulator import (
    SCENARIO_ENVELOPE_BACKEND,
    EVFleetState,
    EVVirtualBatterySimulator,
    RealAssetSpec,
    RealAssetState,
    SimulatorAssetState,
    pypsa_background_from_p2h_mfrr,
    simulate_p2h_scenario_envelope_bid,
    simulate_pypsa_background_asset_bid,
    simulate_real_asset_bid,
    spec_from_scenario,
)


def _stable_result_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backend_comparison(
    *,
    proxy: dict[str, Any],
    scenario_envelope: dict[str, Any],
    pypsa_background: dict[str, Any],
) -> dict[str, Any]:
    return {
        "proxy": proxy,
        SCENARIO_ENVELOPE_BACKEND: scenario_envelope,
        "pypsa_background": pypsa_background,
        "accepted_disagreement": len({
            proxy.get("accepted"),
            scenario_envelope.get("accepted"),
            pypsa_background.get("accepted"),
        }) > 1,
        "proxy_false_positive": (
            proxy.get("accepted") is True
            and (
                scenario_envelope.get("accepted") is not True
                or pypsa_background.get("accepted") is not True
            )
        ),
        "scenario_envelope_false_positive": (
            scenario_envelope.get("accepted") is True
            and pypsa_background.get("accepted") is not True
        ),
    }


def _fallback_real_asset_spec(persona: Any, archetype: str) -> RealAssetSpec:
    if archetype == "ev":
        return RealAssetSpec(
            archetype="ev",
            capacity_mw=persona.capacity_mw,
            storage_mwh=float(persona.storage_mwh or max(persona.capacity_mw, 1.0)),
            availability_share=0.75,
        )
    if archetype == "wind":
        return RealAssetSpec(archetype="wind", capacity_mw=persona.capacity_mw, availability_share=0.45)
    if archetype == "renewables":
        return RealAssetSpec(archetype="renewables", capacity_mw=persona.capacity_mw, availability_share=0.55)
    if archetype == "generator":
        return RealAssetSpec(
            archetype="generator",
            capacity_mw=persona.capacity_mw,
            current_dispatch_share=0.55,
            ramp_share_per_tick=0.35,
            marginal_cost_eur_mwh=55.0,
        )
    if archetype == "retailer":
        return RealAssetSpec(archetype="retailer", capacity_mw=persona.capacity_mw, availability_share=0.12)
    raise ValueError(f"unsupported real asset archetype: {archetype}")


def _remaining_charge_proxy(state: EVFleetState) -> float:
    return max(0.0, state.energy_mwh - state.soc_mwh)


def _remaining_discharge_proxy(state: EVFleetState) -> float:
    return max(0.0, state.soc_mwh)


class SimulationToolsMixin:
    """Simulator gateway methods mixed into :class:`AgentToolExecutor`.

    Expects ``self._persona``, ``self._forecast``, ``self._simulator_tool``,
    ``self._asset_simulator_mode``, ``self._asset_proxy_style``,
    ``self._asset_state_store``, ``self._data_tools`` on the instance, plus
    ``self._call_data_tool`` from :class:`MarketToolsMixin`.
    """

    # -- Shared helpers ------------------------------------------------------

    def _real_asset_spec(self, archetype: str) -> RealAssetSpec:
        if self._simulator_tool is not None and hasattr(self._simulator_tool, "_scenario"):  # type: ignore[attr-defined]
            scenario = self._simulator_tool._scenario  # type: ignore[attr-defined]
            try:
                return spec_from_scenario(scenario, archetype=archetype, zone=self._forecast.zone)  # type: ignore[attr-defined]
            except (KeyError, AttributeError):
                pass
        return _fallback_real_asset_spec(self._persona, archetype)  # type: ignore[attr-defined]

    def commit_asset_state(self, record: ToolCallRecord) -> None:
        next_state = record.result.get("next_state")
        if not isinstance(next_state, dict):
            return
        archetype = record.result.get("archetype")
        if archetype not in {"ev", "wind", "generator", "renewables", "retailer"}:
            return
        spec = self._real_asset_spec(str(archetype))
        self._asset_state_store.commit(  # type: ignore[attr-defined]
            agent_id=self._persona.agent_id,  # type: ignore[attr-defined]
            spec=spec,
            state=RealAssetState(**next_state),
        )

    def _as_proxy_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if not result.get("ok", True):
            return result
        return {
            **result,
            "backend": "proxy",
            "authority": "proxy_comparison",
        }

    def _ev_state(self) -> EVFleetState:
        energy = float(self._persona.storage_mwh or max(self._persona.capacity_mw, 1.0))  # type: ignore[attr-defined]
        return EVFleetState(
            asset_id=self._persona.agent_id,  # type: ignore[attr-defined]
            capacity_mw=self._persona.capacity_mw,  # type: ignore[attr-defined]
            energy_mwh=energy,
            soc_mwh=energy * 0.5,
            availability_share=0.75,
        )

    def _proxy_max_quantity(self, *, archetype: str, side: str) -> float:
        tick_capacity = self._persona.capacity_mw * 0.25  # type: ignore[attr-defined]
        if archetype == "wind":
            return max(0.25, tick_capacity * (0.08 if side == "up" else 0.18))
        if archetype == "renewables":
            return max(0.25, tick_capacity * (0.10 if side == "up" else 0.20))
        if archetype == "generator":
            return max(0.25, tick_capacity * 0.35)
        if archetype == "retailer":
            return max(0.25, tick_capacity * 0.12)
        return max(0.25, tick_capacity * 0.10)

    # -- P2H simulation backends ---------------------------------------------

    def _simulate_p2h_bid(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid, simulate_bid

        proxy = self._simulate_proxy_bid(arguments, archetype="p2h")
        proxy = self._as_proxy_result(proxy)
        if self._asset_simulator_mode == "proxy":  # type: ignore[attr-defined]
            return {**proxy, "controls_acceptance": True}
        if self._asset_simulator_mode == SCENARIO_ENVELOPE_BACKEND:  # type: ignore[attr-defined]
            return {**self._simulate_p2h_scenario_envelope_bid(arguments), "controls_acceptance": True}
        if self._asset_simulator_mode == "pypsa_background":  # type: ignore[attr-defined]
            return {**self._simulate_pypsa_p2h_bid(arguments), "controls_acceptance": True}
        scenario_envelope = self._simulate_p2h_scenario_envelope_bid(arguments)
        pypsa = self._simulate_pypsa_p2h_bid(arguments)
        return self._select_backend_result(
            proxy=proxy,
            scenario_envelope=scenario_envelope,
            pypsa_background=pypsa,
        )

    def _simulate_p2h_scenario_envelope_bid(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale="p2h scenario-envelope simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        if self._simulator_tool is None:  # type: ignore[attr-defined]
            return {"ok": False, "error_code": "simulator_unavailable"}
        result = simulate_p2h_scenario_envelope_bid(
            scenario=self._simulator_tool._scenario,  # type: ignore[attr-defined]
            bid=bid,
            forecast=self._forecast,  # type: ignore[attr-defined]
            tau_eur=float(getattr(self._simulator_tool, "_tau_eur", 0.0)),  # type: ignore[attr-defined]
        )
        payload = asdict(result)
        payload["forecast_interval_eur_mwh"] = list(result.forecast_interval_eur_mwh)
        return {"ok": True, **payload}

    def _apply_pypsa_market_clearing(self, payload: dict[str, Any], bid: Any) -> dict[str, Any]:
        market = getattr(self._simulator_tool, "_market", None)  # type: ignore[attr-defined]
        if market is None or payload.get("accepted") is not True:
            return payload
        market_result = market.clear(self._forecast.to_market_row(), [bid])  # type: ignore[attr-defined]
        reason_codes = [
            decision.reason_code or "market_rejected"
            for decision in market_result.rejected_bids
        ]
        payload["market_cleared"] = bool(market_result.accepted_bids)
        payload["market_reason_codes"] = reason_codes
        if not market_result.accepted_bids:
            payload["accepted"] = False
            payload["reason_codes"] = list(dict.fromkeys(list(payload.get("reason_codes", [])) + reason_codes))
            payload["failed_stage"] = "market"
        payload["result_hash"] = _stable_result_hash(
            {key: value for key, value in payload.items() if key != "result_hash"}
        )
        return payload

    def _simulate_pypsa_p2h_bid(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid, simulate_bid

        decision = LLMBidDecision(action="bid", rationale="p2h PyPSA-background simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        if self._simulator_tool is None:  # type: ignore[attr-defined]
            return {"ok": False, "error_code": "simulator_unavailable"}
        result = simulate_bid(self._simulator_tool, bid, self._forecast)  # type: ignore[attr-defined]
        lower, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        median = (lower + upper) / 2.0
        expected_spread = median - self._forecast.spot_price_eur_mwh if bid.side == "up" else self._forecast.spot_price_eur_mwh - median  # type: ignore[attr-defined]
        return pypsa_background_from_p2h_mfrr(
            scenario=self._simulator_tool._scenario,  # type: ignore[attr-defined]
            bid=bid,
            forecast=self._forecast,  # type: ignore[attr-defined]
            accepted=result.accepted,
            reason_codes=result.reason_codes,
            expected_profit_eur=bid.quantity_mwh * expected_spread,
            worst_case_profit_eur=result.worst_case_profit_eur,
            forecast_interval_eur_mwh=result.forecast_interval_eur_mwh,
            result_hash=result.result_hash,
        )

    # -- Generic asset simulation backends -----------------------------------

    def _simulate_asset_bid(self, arguments: dict[str, Any], *, archetype: str) -> dict[str, Any]:
        if archetype == "ev" and self._asset_proxy_style == "asset_light":  # type: ignore[attr-defined]
            proxy = self._simulate_ev_bid(arguments)
        else:
            proxy = self._simulate_proxy_bid(arguments, archetype=archetype)
        proxy = self._as_proxy_result(proxy)
        if self._asset_simulator_mode == "proxy":  # type: ignore[attr-defined]
            return {**proxy, "controls_acceptance": True}
        if self._asset_simulator_mode == SCENARIO_ENVELOPE_BACKEND:  # type: ignore[attr-defined]
            return {**self._simulate_real_asset_bid(arguments, archetype=archetype), "controls_acceptance": True}
        if self._asset_simulator_mode == "pypsa_background":  # type: ignore[attr-defined]
            return {**self._simulate_pypsa_asset_bid(arguments, archetype=archetype), "controls_acceptance": True}
        scenario_envelope = self._simulate_real_asset_bid(arguments, archetype=archetype)
        pypsa = self._simulate_pypsa_asset_bid(arguments, archetype=archetype)
        return self._select_backend_result(
            proxy=proxy,
            scenario_envelope=scenario_envelope,
            pypsa_background=pypsa,
        )

    def _simulate_real_asset_bid(self, arguments: dict[str, Any], *, archetype: str) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale=f"{archetype} scenario-envelope simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        spec = self._real_asset_spec(archetype)
        state = self._asset_state_store.get(agent_id=self._persona.agent_id, spec=spec)  # type: ignore[attr-defined]
        result = simulate_real_asset_bid(
            spec=spec,
            bid=bid,
            forecast=self._forecast,  # type: ignore[attr-defined]
            state=state,
            tau_eur=float(getattr(self._simulator_tool, "_tau_eur", 0.0) if self._simulator_tool is not None else 0.0),  # type: ignore[attr-defined]
        )
        payload = asdict(result)
        payload["forecast_interval_eur_mwh"] = list(result.forecast_interval_eur_mwh)
        return {"ok": True, **payload}

    def _simulate_pypsa_asset_bid(self, arguments: dict[str, Any], *, archetype: str) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale=f"{archetype} PyPSA-background simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        if self._simulator_tool is None or not hasattr(self._simulator_tool, "_scenario"):  # type: ignore[attr-defined]
            return {"ok": False, "error_code": "simulator_unavailable"}
        spec = self._real_asset_spec(archetype)
        state = self._asset_state_store.get(agent_id=self._persona.agent_id, spec=spec)  # type: ignore[attr-defined]
        result = simulate_pypsa_background_asset_bid(
            scenario=self._simulator_tool._scenario,  # type: ignore[attr-defined]
            spec=spec,
            bid=bid,
            forecast=self._forecast,  # type: ignore[attr-defined]
            state=state,
            tau_eur=float(getattr(self._simulator_tool, "_tau_eur", 0.0)),  # type: ignore[attr-defined]
        )
        payload = asdict(result)
        payload["forecast_interval_eur_mwh"] = list(result.forecast_interval_eur_mwh)
        return self._apply_pypsa_market_clearing({"ok": True, **payload}, bid)

    def _select_backend_result(
        self,
        *,
        proxy: dict[str, Any],
        scenario_envelope: dict[str, Any],
        pypsa_background: dict[str, Any],
    ) -> dict[str, Any]:
        comparison = _backend_comparison(
            proxy=proxy,
            scenario_envelope=scenario_envelope,
            pypsa_background=pypsa_background,
        )
        selected = {
            "proxy": proxy,
            SCENARIO_ENVELOPE_BACKEND: scenario_envelope,
            "pypsa_background": pypsa_background,
            "dual_compare_proxy_controls": proxy,
            "dual_compare_real_controls": scenario_envelope,
            "dual_compare_pypsa_controls": pypsa_background,
            "dual_compare_real_vs_pypsa": pypsa_background,
        }.get(self._asset_simulator_mode, proxy)  # type: ignore[attr-defined]
        payload = {**selected, "controls_acceptance": True}
        if self._asset_simulator_mode.startswith("dual_compare_"):  # type: ignore[attr-defined]
            payload["comparison"] = comparison
        return payload

    # -- EV-specific simulation ----------------------------------------------

    def _simulate_ev_bid(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale="ev simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        activation_context = self._call_data_tool("get_activation_context", {"hours": 24, "zone": bid.zone})  # type: ignore[attr-defined]
        direction_hint = str((activation_context or {}).get("direction_hint") or "neutral")
        if direction_hint in {"up", "down"} and direction_hint != bid.side:
            state = self._ev_state()
            available_quantity_mwh = state.capacity_mw * state.availability_share * 0.25
            return {
                "ok": True,
                "simulator_kind": "ev_virtual_battery",
                "archetype": "ev",
                "authority": "authoritative",
                "accepted": False,
                "reason_codes": ["activation_prior_side_mismatch"],
                "projected_soc_mwh": round(state.soc_mwh, 6),
                "remaining_charge_mwh": round(_remaining_charge_proxy(state), 6),
                "remaining_discharge_mwh": round(_remaining_discharge_proxy(state), 6),
                "failed_stage": "activation",
                "available_quantity_mwh": round(available_quantity_mwh, 6),
                "energy_mwh": round(state.energy_mwh, 6),
                "soc_mwh": round(state.soc_mwh, 6),
                "availability_share": round(state.availability_share, 6),
                "activation_direction_hint": direction_hint,
                "result_hash": "",
            }
        lower, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        if bid.side == "up":
            expected_spread = ((lower + upper) / 2.0) - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
            worst_spread = lower - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
        else:
            expected_spread = self._forecast.spot_price_eur_mwh - ((lower + upper) / 2.0)  # type: ignore[attr-defined]
            worst_spread = self._forecast.spot_price_eur_mwh - upper  # type: ignore[attr-defined]
        if expected_spread <= 0.0 or worst_spread < 0.0:
            state = self._ev_state()
            available_quantity_mwh = state.capacity_mw * state.availability_share * 0.25
            reason_codes = []
            if expected_spread <= 0.0:
                reason_codes.append("weak_expected_spread")
            if worst_spread < 0.0:
                reason_codes.append("negative_worst_case_spread")
            return {
                "ok": True,
                "simulator_kind": "ev_virtual_battery",
                "archetype": "ev",
                "authority": "authoritative",
                "accepted": False,
                "reason_codes": reason_codes,
                "expected_profit_eur": round(bid.quantity_mwh * expected_spread, 6),
                "worst_case_profit_eur": round(bid.quantity_mwh * worst_spread, 6),
                "projected_soc_mwh": round(state.soc_mwh, 6),
                "remaining_charge_mwh": round(_remaining_charge_proxy(state), 6),
                "remaining_discharge_mwh": round(_remaining_discharge_proxy(state), 6),
                "failed_stage": "price_risk",
                "available_quantity_mwh": round(available_quantity_mwh, 6),
                "energy_mwh": round(state.energy_mwh, 6),
                "soc_mwh": round(state.soc_mwh, 6),
                "availability_share": round(state.availability_share, 6),
                "activation_direction_hint": direction_hint,
                "forecast_interval_eur_mwh": [round(lower, 6), round(upper, 6)],
                "result_hash": "",
            }
        state = self._ev_state()
        result = EVVirtualBatterySimulator(state).simulate_bid(bid)
        available_quantity_mwh = state.capacity_mw * state.availability_share * 0.25
        return {
            "ok": True,
            "simulator_kind": result.simulator_kind,
            "archetype": result.archetype,
            "authority": result.authority,
            "accepted": result.accepted,
            "reason_codes": result.reason_codes,
            "projected_soc_mwh": result.projected_soc_mwh,
            "remaining_charge_mwh": result.remaining_charge_mwh,
            "remaining_discharge_mwh": result.remaining_discharge_mwh,
            "failed_stage": result.failed_stage,
            "expected_profit_eur": round(bid.quantity_mwh * expected_spread, 6),
            "worst_case_profit_eur": round(bid.quantity_mwh * worst_spread, 6),
            "available_quantity_mwh": round(available_quantity_mwh, 6),
            "energy_mwh": round(state.energy_mwh, 6),
            "soc_mwh": round(state.soc_mwh, 6),
            "availability_share": round(state.availability_share, 6),
            "activation_direction_hint": direction_hint,
            "forecast_interval_eur_mwh": [round(lower, 6), round(upper, 6)],
            "result_hash": result.result_hash,
        }

    # -- Proxy simulation (all archetypes) -----------------------------------

    def _simulate_proxy_bid(self, arguments: dict[str, Any], *, archetype: str) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale=f"{archetype} proxy simulation", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        lower, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        median = (lower + upper) / 2.0
        if bid.side == "up":
            expected_spread = median - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
            worst_spread = lower - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
            price_crosses_proxy = bid.limit_price_eur_mwh <= median
        else:
            expected_spread = self._forecast.spot_price_eur_mwh - median  # type: ignore[attr-defined]
            worst_spread = self._forecast.spot_price_eur_mwh - upper  # type: ignore[attr-defined]
            price_crosses_proxy = bid.limit_price_eur_mwh >= median
        max_quantity = self._proxy_max_quantity(archetype=archetype, side=bid.side)
        reason_codes: list[str] = []
        if bid.quantity_mwh > max_quantity + 1e-9:
            reason_codes.append(f"{archetype}_proxy_quantity_exceeded")
        if expected_spread <= 0:
            reason_codes.append("weak_expected_spread")
        if worst_spread < 0:
            reason_codes.append("negative_worst_case_spread")
        if not price_crosses_proxy:
            reason_codes.append("proxy_price_not_clearable")
        accepted = not reason_codes
        payload = {
            "ok": True,
            "simulator_kind": f"{archetype}_proxy_mfrr",
            "archetype": archetype,
            "backend": "proxy",
            "authority": "proxy_comparison",
            "accepted": accepted,
            "reason_codes": reason_codes,
            "expected_profit_eur": round(bid.quantity_mwh * expected_spread, 6),
            "worst_case_profit_eur": round(bid.quantity_mwh * worst_spread, 6),
            "max_quantity_mwh": round(max_quantity, 6),
            "physical_limit_mwh": round(max_quantity, 6),
            "forecast_interval_eur_mwh": [round(lower, 6), round(upper, 6)],
        }
        payload["result_hash"] = _stable_result_hash(
            {
                "simulator_kind": payload["simulator_kind"],
                "archetype": archetype,
                "side": bid.side,
                "quantity_mwh": round(bid.quantity_mwh, 6),
                "limit_price_eur_mwh": round(bid.limit_price_eur_mwh, 6),
                "accepted": accepted,
                "reason_codes": reason_codes,
                "expected_profit_eur": payload["expected_profit_eur"],
                "worst_case_profit_eur": payload["worst_case_profit_eur"],
                "forecast_interval_eur_mwh": payload["forecast_interval_eur_mwh"],
                "forecast_hash": self._forecast.result_hash,  # type: ignore[attr-defined]
            }
        )
        return payload
