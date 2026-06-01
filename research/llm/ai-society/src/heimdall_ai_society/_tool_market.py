"""Market context accessor tool methods (mixin for AgentToolExecutor)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from heimdall_contracts import ActivationForecast, QuantileForecast

from heimdall_ai_society.schemas import ToolCallRecord


class MarketToolsMixin:
    """Methods that read market data and derive non-leaking context signals.

    Mixed into :class:`AgentToolExecutor` — expects ``self._forecast``,
    ``self._data_tools``, ``self._persona``, ``self._retriever``,
    ``self._rag_top_k``, ``self._rag_max_chars``, and
    ``self._candidate_diagnostics`` to be present on the instance.
    """

    # -- RAG retrieval -------------------------------------------------------

    def _retrieve_knowledge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._retriever is None:  # type: ignore[attr-defined]
            return {"ok": False, "error_code": "retrieval_not_configured"}
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"ok": False, "error_code": "empty_query"}
        try:
            k = int(arguments.get("k", self._rag_top_k))  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            k = self._rag_top_k  # type: ignore[attr-defined]
        k = max(1, min(12, k))
        kinds = arguments.get("kinds")
        kinds_t = tuple(kinds) if isinstance(kinds, list) and kinds else None
        cutoff = self._forecast.delivery_timestamp  # type: ignore[attr-defined]
        results = self._retriever.retrieve(  # type: ignore[attr-defined]
            query, as_of=cutoff, k=k, max_chars=self._rag_max_chars, kinds=kinds_t  # type: ignore[attr-defined]
        )
        return {
            "ok": True,
            "authority": "advisory",
            "as_of": str(cutoff),
            "query": query,
            "result_count": len(results),
            "results": results,
            "note": (
                "Knowledge base is filtered to documents available on or before this market tick; "
                "future or same-window outcomes are never returned."
            ),
        }

    # -- Data tool helper ----------------------------------------------------

    def _call_data_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if self._data_tools is None:  # type: ignore[attr-defined]
            return None
        method: Callable[..., dict[str, Any]] | None = getattr(self._data_tools, name, None)  # type: ignore[attr-defined]
        if method is None:
            return None
        return method(**arguments)

    # -- Fallback / derived context ------------------------------------------

    def _fallback_opportunity_context(self, *, kind: str = "activation_context") -> dict[str, Any]:
        lower, upper = self._forecast.interval_for_side("up")  # type: ignore[attr-defined]
        spread = ((lower + upper) / 2.0) - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
        return {
            "ok": True,
            "kind": kind,
            "zone": self._forecast.zone,  # type: ignore[attr-defined]
            "observed_at": self._forecast.issued_at,  # type: ignore[attr-defined]
            "watch_score": round(max(0.0, min(1.0, abs(spread) / 50.0)), 6),
            "direction_hint": "up" if spread >= 0 else "down",
            "signals": {"forecast_mid_spread_eur_mwh": round(spread, 6)},
        }

    def _activation_forecast(self) -> dict[str, Any]:
        context = self._call_data_tool(
            "get_activation_context",
            {"hours": 24, "zone": self._forecast.zone},  # type: ignore[attr-defined]
        )
        hint = str((context or {}).get("direction_hint") or self._forecast.activation_direction)  # type: ignore[attr-defined]
        watch_score = float((context or {}).get("watch_score", 0.0) or 0.0)
        base = max(0.0, min(0.85, 0.35 + 0.5 * watch_score))
        if hint == "up":
            p_up, p_down = base, max(0.05, (1.0 - base) * 0.35)
        elif hint == "down":
            p_down, p_up = base, max(0.05, (1.0 - base) * 0.35)
        else:
            p_up = p_down = max(0.05, (1.0 - base) * 0.5)
        p_neutral = max(0.0, 1.0 - p_up - p_down)
        total = p_up + p_down + p_neutral
        p_up, p_down, p_neutral = p_up / total, p_down / total, p_neutral / total
        volume = max(0.0, float(self._forecast.activation_volume_mwh))  # type: ignore[attr-defined]
        if volume <= 0.0:
            volume = max(0.0, watch_score * 4.0)
        q = QuantileForecast(
            horizon_minutes=15,
            levels=(0.1, 0.5, 0.9),
            values=(round(0.25 * volume, 6), round(volume, 6), round(1.75 * volume, 6)),
        )
        forecast = ActivationForecast(
            issued_at=self._forecast.issued_datetime(),  # type: ignore[attr-defined]
            zone=self._forecast.zone,  # type: ignore[arg-type, attr-defined]
            horizon_minutes=15,
            p_up=round(p_up, 6),
            p_down=round(p_down, 6),
            p_neutral=round(1.0 - round(p_up, 6) - round(p_down, 6), 6),
            volume_quantiles_mwh=q,
            source_model="activation-context-f0",
            leakage_guard="agent_visible_context",
        )
        payload = forecast.model_dump(mode="json")
        payload.update(
            {
                "ok": True,
                "kind": "activation_forecast",
                "authority": "advisory_not_verifier",
                "context_hash": hashlib.sha256(
                    json.dumps(context or {}, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
            }
        )
        return payload

    def _market_regime_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        activation = self._call_data_tool("get_activation_context", arguments) or self._fallback_opportunity_context()
        uncertainty = self._uncertainty_digest()
        watch_score = float(activation.get("watch_score", 0.0) or 0.0)
        interval_width = float(uncertainty["signals"]["max_interval_width_eur_mwh"])
        edge = float(uncertainty["signals"]["max_abs_price_edge_eur_mwh"])
        if interval_width >= 80.0 or edge >= 40.0:
            regime = "volatile"
        elif watch_score >= 0.6:
            regime = "high_activation_watch"
        elif edge <= 5.0 and interval_width <= 25.0:
            regime = "quiet"
        else:
            regime = "normal_watch"
        return {
            "ok": True,
            "kind": "market_regime_context",
            "authority": "derived_non_leaking",
            "zone": self._forecast.zone,  # type: ignore[attr-defined]
            "observed_at": self._forecast.issued_at,  # type: ignore[attr-defined]
            "regime_label": regime,
            "watch_score": round(watch_score, 6),
            "signals": {
                **uncertainty["signals"],
                "activation_direction_hint": activation.get("direction_hint", "neutral"),
            },
            "unavailable_future_enrichments": ["jao_cnec", "reserve_saturation", "intraday_order_book"],
        }

    def _border_pressure(self, arguments: dict[str, Any]) -> dict[str, Any]:
        context = self._call_data_tool("get_crossborder_flows", arguments)
        rows = (context or {}).get("rows", []) if isinstance(context, dict) else []
        flows = [float(row.get("flow_mw", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
        if not flows:
            return {
                "ok": True,
                "kind": "border_pressure",
                "authority": "derived_non_leaking",
                "zone": self._forecast.zone,  # type: ignore[attr-defined]
                "row_count": 0,
                "pressure_label": "unavailable",
                "signals": {"reason": "missing_flow_context"},
                "unavailable_future_enrichments": ["jao_ram", "jao_shadow_price", "jao_domain_contraction"],
            }
        latest = flows[-1]
        mean_abs = sum(abs(value) for value in flows) / len(flows)
        swing = max(flows) - min(flows)
        label = "high_export_pressure" if latest > mean_abs * 0.75 else "high_import_pressure" if latest < -mean_abs * 0.75 else "balanced_or_mixed"
        return {
            "ok": True,
            "kind": "border_pressure",
            "authority": "derived_non_leaking",
            "zone": self._forecast.zone,  # type: ignore[attr-defined]
            "row_count": len(flows),
            "pressure_label": label,
            "signals": {
                "latest_flow_mw": round(latest, 6),
                "mean_abs_flow_mw": round(mean_abs, 6),
                "flow_swing_mw": round(swing, 6),
            },
            "unavailable_future_enrichments": ["jao_ram", "jao_shadow_price", "jao_domain_contraction"],
        }

    def _grid_constraints(self, arguments: dict[str, Any]) -> dict[str, Any]:
        context = self._call_data_tool("get_grid_constraints", arguments)
        if context is not None:
            return context
        return {
            "ok": True,
            "kind": "grid_constraints",
            "authority": "jao_optional",
            "zone": self._forecast.zone,  # type: ignore[attr-defined]
            "row_count": 0,
            "pressure_label": "unavailable",
            "signals": {"reason": "missing_jao_constraints"},
        }

    def _outage_impact(self, arguments: dict[str, Any]) -> dict[str, Any]:
        context = self._call_data_tool("get_outages", arguments)
        rows = (context or {}).get("rows", []) if isinstance(context, dict) else []
        scored = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            capacity = float(row.get("max_unavailable_capacity_mw") or row.get("unavailable_capacity_mw") or 0.0)
            score = min(1.0, capacity / 1000.0)
            scored.append(
                {
                    "summary": row.get("event_type") or row.get("reason") or row.get("message") or "outage",
                    "impact_score": round(score, 6),
                    "max_unavailable_capacity_mw": capacity,
                    "published_at_utc": row.get("published_at_utc"),
                    "time_start_utc": row.get("time_start_utc"),
                    "time_end_utc": row.get("time_end_utc"),
                }
            )
        scored = sorted(scored, key=lambda item: float(item["impact_score"]), reverse=True)
        label = "high" if scored and scored[0]["impact_score"] >= 0.5 else "medium" if scored else "none"
        return {
            "ok": True,
            "kind": "outage_impact",
            "authority": "derived_non_leaking",
            "zone": self._forecast.zone,  # type: ignore[attr-defined]
            "impact_label": label,
            "row_count": len(scored),
            "top_events": scored[:5],
        }

    def _uncertainty_digest(self) -> dict[str, Any]:
        up_lower, up_upper = self._forecast.interval_for_side("up")  # type: ignore[attr-defined]
        down_lower, down_upper = self._forecast.interval_for_side("down")  # type: ignore[attr-defined]
        up_edge = up_lower - self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
        down_edge = self._forecast.spot_price_eur_mwh - down_upper  # type: ignore[attr-defined]
        width = max(up_upper - up_lower, down_upper - down_lower)
        side_gap = abs(up_edge - down_edge)
        if width >= 80.0 or side_gap <= 5.0:
            label = "high"
        elif width >= 35.0 or side_gap <= 15.0:
            label = "medium"
        else:
            label = "low"
        return {
            "ok": True,
            "kind": "uncertainty_digest",
            "authority": "derived_non_leaking",
            "uncertainty_label": label,
            "side_ambiguity": side_gap <= 15.0,
            "candidate_side_hint": "up" if up_edge >= max(5.0, down_edge) else "down" if down_edge >= 5.0 else None,
            "signals": {
                "up_edge_lower_minus_spot_eur_mwh": round(up_edge, 6),
                "down_edge_spot_minus_upper_eur_mwh": round(down_edge, 6),
                "max_interval_width_eur_mwh": round(width, 6),
                "max_abs_price_edge_eur_mwh": round(max(abs(up_edge), abs(down_edge)), 6),
                "side_edge_gap_eur_mwh": round(side_gap, 6),
            },
        }

    def _limit_price_guidance(self, arguments: dict[str, Any]) -> dict[str, Any]:
        side = str(arguments.get("side") or "up")
        quantity = float(arguments.get("quantity_mwh") or 0.25)
        lower, upper = self._forecast.interval_for_side(side)  # type: ignore[attr-defined]
        median = (lower + upper) / 2.0
        spot = self._forecast.spot_price_eur_mwh  # type: ignore[attr-defined]
        if side == "up":
            crossing = max(spot + 1.0, lower - 1.0)
            balanced = median
            profit_seek = max(lower, median - 0.25 * (upper - lower))
            worst_spread = lower - spot
            expected_spread = median - spot
        else:
            crossing = min(spot - 1.0, upper + 1.0)
            balanced = median
            profit_seek = min(upper, median + 0.25 * (upper - lower))
            worst_spread = spot - upper
            expected_spread = spot - median
        return {
            "ok": True,
            "kind": "limit_price_guidance",
            "authority": "derived_from_forecast_interval",
            "side": side,
            "quantity_mwh": round(quantity, 6),
            "recommended_limit_price_eur_mwh": round(crossing, 2),
            "price_ladder": [
                {"style": "clear_probability", "limit_price_eur_mwh": round(crossing, 2), "clear_probability_proxy": 0.85},
                {"style": "balanced", "limit_price_eur_mwh": round(balanced, 2), "clear_probability_proxy": 0.6},
                {"style": "profit_seeking", "limit_price_eur_mwh": round(profit_seek, 2), "clear_probability_proxy": 0.4},
            ],
            "signals": {
                "expected_profit_proxy_eur": round(quantity * expected_spread, 6),
                "worst_case_profit_proxy_eur": round(quantity * worst_spread, 6),
                "forecast_lower_eur_mwh": round(lower, 6),
                "forecast_upper_eur_mwh": round(upper, 6),
                "spot_price_eur_mwh": round(spot, 6),
            },
        }

    def _decision_trace_summary(self) -> dict[str, Any]:
        rejection = self._candidate_rejection_summary()  # type: ignore[attr-defined]
        uncertainty = self._uncertainty_digest()
        accepted = int(rejection.get("accepted_candidate_count", 0) or 0)
        simulated = int(rejection.get("simulated_candidate_count", 0) or 0)
        if accepted:
            recommendation = "bid_or_chair_review"
        elif simulated:
            recommendation = "watch"
        else:
            recommendation = "advisory"
        return {
            "ok": True,
            "kind": "decision_trace_summary",
            "authority": "derived_from_visible_tool_trace",
            "recommendation": recommendation,
            "summary": (
                f"{accepted}/{simulated} simulated candidates accepted; "
                f"uncertainty={uncertainty.get('uncertainty_label')}"
            ),
            "candidate_summary": rejection,
            "uncertainty_summary": uncertainty,
        }
