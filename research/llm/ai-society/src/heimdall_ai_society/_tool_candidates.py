"""Candidate evaluation tool methods (mixin for AgentToolExecutor)."""

from __future__ import annotations

from typing import Any

from heimdall_ai_society.schemas import LLMBidDecision
from heimdall_ai_society._tool_simulation import _remaining_charge_proxy, _remaining_discharge_proxy


def _quantity_ladder(cap_mwh: float) -> list[float]:
    cap = max(0.0, cap_mwh)
    return [quantity for quantity in [0.25, 0.5, 1.0, 2.0] if quantity <= cap + 1e-9] or ([round(cap, 6)] if cap > 0 else [])


class CandidateToolsMixin:
    """Candidate evaluation / feasibility / sizing methods.

    Mixed into :class:`AgentToolExecutor` — expects ``self._persona``,
    ``self._forecast``, ``self._candidate_diagnostics``, plus simulation
    methods from :class:`SimulationToolsMixin`.
    """

    def _candidate_rejection_summary(self) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        accepted = 0
        simulated = 0
        for record in self._candidate_diagnostics:  # type: ignore[attr-defined]
            if not record.name.startswith("simulate"):
                continue
            simulated += 1
            if record.result.get("accepted") is True:
                accepted += 1
            for reason in record.result.get("reason_codes", []):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        return {
            "ok": True,
            "kind": "candidate_rejection_summary",
            "authority": "derived_from_seeded_candidates",
            "simulated_candidate_count": simulated,
            "accepted_candidate_count": accepted,
            "rejection_reason_counts": reason_counts,
        }

    def _candidate_sizing_guidance(self, archetype: str) -> dict[str, Any]:
        tick_capacity = self._persona.capacity_mw * 0.25  # type: ignore[attr-defined]
        if archetype == "ev":
            state = self._ev_state()  # type: ignore[attr-defined]
            available_power_mw = state.capacity_mw * state.availability_share
            available_quantity = available_power_mw * 0.25
            up_cap = min(available_quantity, _remaining_discharge_proxy(state))
            down_cap = min(available_quantity, _remaining_charge_proxy(state))
            return {
                "ok": True,
                "kind": "candidate_sizing_guidance",
                "authority": "derived_from_ev_virtual_battery",
                "archetype": "ev",
                "suggested_quantities_mwh": _quantity_ladder(min(up_cap, down_cap)),
                "side_caps_mwh": {"up": round(up_cap, 6), "down": round(down_cap, 6)},
                "signals": {
                    "available_quantity_mwh": round(available_quantity, 6),
                    "soc_mwh": round(state.soc_mwh, 6),
                    "energy_mwh": round(state.energy_mwh, 6),
                    "availability_share": round(state.availability_share, 6),
                },
            }
        share = {"p2h": 0.04, "wind": 0.08, "generator": 0.35, "retailer": 0.12, "renewables": 0.10}.get(archetype, 0.10)
        cap = max(0.25, tick_capacity * share)
        return {
            "ok": True,
            "kind": "candidate_sizing_guidance",
            "authority": "derived_from_persona_capacity",
            "archetype": archetype,
            "suggested_quantities_mwh": _quantity_ladder(cap),
            "side_caps_mwh": {"up": round(cap, 6), "down": round(cap, 6)},
        }

    def _p2h_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale="tool feasibility", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        lower, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        adverse = lower if bid.side == "up" else upper
        median = (lower + upper) / 2.0
        if bid.side == "up":
            expected_spread = median - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
            worst_spread = adverse - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
            price_edge = median - bid.limit_price_eur_mwh
        else:
            expected_spread = self._forecast.spot_price_eur_mwh - median  # type: ignore[attr-defined]
            worst_spread = self._forecast.spot_price_eur_mwh - adverse  # type: ignore[attr-defined]
            price_edge = bid.limit_price_eur_mwh - median
        quantity_ratio = bid.quantity_mwh / max(self._persona.capacity_mw * 0.25, 1e-9)  # type: ignore[attr-defined]
        interval_width = max(0.0, upper - lower)
        risk_penalty = min(0.4, quantity_ratio * 0.2) + min(0.35, interval_width / 200.0)
        score = max(0.0, min(1.0, 0.5 + expected_spread / 80.0 + price_edge / 100.0 - risk_penalty))
        flags: list[str] = []
        if quantity_ratio > 0.8:
            flags.append("large_quantity_vs_persona_capacity")
        if worst_spread < 0:
            flags.append("negative_worst_case_spread")
        if expected_spread <= 0:
            flags.append("weak_expected_spread")
        if interval_width > 80:
            flags.append("wide_forecast_interval")
        guidance = "worth_simulating" if score >= 0.55 and "negative_worst_case_spread" not in flags else "watch_or_resize"
        return {
            "ok": True,
            "kind": "bid_feasibility",
            "archetype": "p2h",
            "authority": "advisory",
            "accepted": score >= 0.55,
            "score": round(score, 6),
            "guidance": guidance,
            "risk_flags": flags,
            "expected_spread_eur_mwh": round(expected_spread, 6),
            "worst_case_spread_eur_mwh": round(worst_spread, 6),
            "rough_expected_profit_eur": round(bid.quantity_mwh * expected_spread, 6),
            "rough_worst_case_profit_eur": round(bid.quantity_mwh * worst_spread, 6),
            "forecast_interval_eur_mwh": [round(lower, 6), round(upper, 6)],
            "quantity_ratio_to_tool_cap": round(quantity_ratio, 6),
        }

    def _ev_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._simulate_asset_bid(arguments, archetype="ev")  # type: ignore[attr-defined]
        score = 0.75 if result.get("accepted") else 0.25
        if "ev_capacity_exceeded" in result.get("reason_codes", []):
            score -= 0.1
        if "ev_soc_lower_exceeded" in result.get("reason_codes", []) or "ev_soc_upper_exceeded" in result.get("reason_codes", []):
            score -= 0.15
        return {
            **result,
            "kind": "ev_bid_feasibility",
            "authority": "advisory",
            "score": round(max(0.0, min(1.0, score)), 6),
            "guidance": "worth_simulating" if result.get("accepted") else "resize_or_watch",
        }

    def _wind_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale="wind feasibility", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        availability_mwh = self._persona.capacity_mw * 0.25 * 0.45  # type: ignore[attr-defined]
        _, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        lower, _ = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        interval_width = max(0.0, upper - lower)
        flags = []
        if bid.quantity_mwh > availability_mwh:
            flags.append("quantity_above_wind_availability_proxy")
        if interval_width > 80:
            flags.append("high_forecast_uncertainty")
        if bid.side == "down":
            flags.append("wrong_side_risk_for_wind_curtailment")
        score = max(0.0, min(1.0, 0.7 - 0.25 * len(flags) - min(interval_width / 300.0, 0.25)))
        return {
            "ok": True,
            "kind": "wind_bid_feasibility",
            "archetype": "wind",
            "authority": "advisory",
            "accepted": score >= 0.55,
            "score": round(score, 6),
            "risk_flags": flags,
            "availability_mwh_proxy": round(availability_mwh, 6),
            "guidance": "watch_or_small_bid" if score >= 0.55 else "watch_only",
        }

    def _generator_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from heimdall_ai_society.tools import decision_to_bid

        decision = LLMBidDecision(action="bid", rationale="generator feasibility", confidence=0.5, **arguments)
        bid = decision_to_bid(self._persona, decision, self._forecast)  # type: ignore[attr-defined]
        if bid is None:
            return {"ok": False, "error_code": "invalid_bid_fields"}
        max_tick_mwh = self._persona.capacity_mw * 0.25  # type: ignore[attr-defined]
        ramp_proxy_mwh = max_tick_mwh * 0.35
        lower, upper = self._forecast.interval_for_side(bid.side)  # type: ignore[attr-defined]
        midpoint = (lower + upper) / 2.0
        spread = midpoint - self._forecast.spot_price_eur_mwh if bid.side == "up" else self._forecast.spot_price_eur_mwh - midpoint  # type: ignore[attr-defined]
        flags = []
        if bid.quantity_mwh > ramp_proxy_mwh:
            flags.append("ramp_proxy_exceeded")
        if spread < 5.0:
            flags.append("weak_marginal_spread")
        score = max(0.0, min(1.0, 0.65 + spread / 120.0 - 0.25 * len(flags)))
        return {
            "ok": True,
            "kind": "generator_bid_feasibility",
            "archetype": "generator",
            "authority": "advisory",
            "accepted": score >= 0.55,
            "score": round(score, 6),
            "risk_flags": flags,
            "ramp_proxy_mwh": round(ramp_proxy_mwh, 6),
            "marginal_spread_proxy_eur_mwh": round(spread, 6),
            "guidance": "watch_or_small_bid" if score >= 0.55 else "watch_only",
        }

    def _retailer_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._proxy_bid_feasibility(arguments, archetype="retailer", max_share=0.12)

    def _renewables_bid_feasibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._proxy_bid_feasibility(arguments, archetype="renewables", max_share=0.10)

    def _proxy_bid_feasibility(self, arguments: dict[str, Any], *, archetype: str, max_share: float) -> dict[str, Any]:
        result = self._simulate_proxy_bid(arguments, archetype=archetype)  # type: ignore[attr-defined]
        quantity = float(arguments.get("quantity_mwh", 0.0) or 0.0)
        max_quantity = max(self._persona.capacity_mw * 0.25 * max_share, 0.25)  # type: ignore[attr-defined]
        score = 0.7 if result.get("accepted") else 0.25
        if quantity > max_quantity:
            score -= 0.2
        if float(result.get("worst_case_profit_eur") or -1.0) < 0:
            score -= 0.15
        return {
            **result,
            "kind": f"{archetype}_bid_feasibility",
            "authority": "advisory",
            "score": round(max(0.0, min(1.0, score)), 6),
            "max_quantity_mwh_proxy": round(max_quantity, 6),
            "guidance": "worth_simulating" if score >= 0.55 else "resize_or_watch",
        }
