from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from packages.pypsa_adapter import HeimdallScenario

from .forecast import ForecastMarketState
from .market import MFRRMarketClock
from .mfrr_engine import (
    CalibratedMFRRPriceModel,
    MFRRClearingEngine,
    MFRRClearingResult,
    PriceModelQuality,
)
from .models import Bid, MarketState, SimulatorAssetState
from .physical import PhysicalConstraintProvider


@dataclass(frozen=True)
class AgentMFRRToolResult:
    accepted: bool
    agent_summary: str
    reason_codes: list[str]
    submission_gate_utc: str
    acceptance_notice_utc: str
    physical_projected_power_mw: float | None
    physical_projected_thermal_soc_mwh: float | None
    physical_remaining_capacity_mw: float | None
    model_quality: PriceModelQuality | None
    market_result: MFRRClearingResult | None
    forecast_interval_eur_mwh: tuple[float, float] | None
    worst_case_profit_eur: float | None
    required_profit_improvement_eur: float | None
    verifier_stage_failed: str | None
    result_hash: str


class AgentMFRRTool:
    def __init__(
        self,
        scenario: HeimdallScenario,
        price_model: CalibratedMFRRPriceModel,
        *,
        clock: MFRRMarketClock | None = None,
        tau_eur: float = 0.0,
    ) -> None:
        self._scenario = scenario
        self._clock = clock or MFRRMarketClock()
        self._physical = PhysicalConstraintProvider(scenario)
        self._market = MFRRClearingEngine(price_model)
        self._tau_eur = tau_eur

    def simulate_bid(
        self,
        bid: Bid,
        market_row: dict[str, Any],
        *,
        asset_state: SimulatorAssetState | None = None,
    ) -> AgentMFRRToolResult:
        state = self._resolve_state(bid, market_row, asset_state)
        submission_gate = self._clock.submission_gate_timestamp(state)
        acceptance_notice = self._clock.acceptance_timestamp(state)

        if not self._clock.is_gate_open(bid, state):
            return _tool_result(
                accepted=False,
                reason_codes=["gate_closed"],
                submission_gate_utc=_iso_z(submission_gate),
                acceptance_notice_utc=_iso_z(acceptance_notice),
                agent_summary="rejected:gate_closed",
            )

        return self._run_pipeline(bid, state, market_row, submission_gate, acceptance_notice)

    def simulate_bid_from_forecast(
        self,
        bid: Bid,
        forecast: ForecastMarketState,
        *,
        mode: str = "forecast",
        asset_state: SimulatorAssetState | None = None,
    ) -> AgentMFRRToolResult:
        market_row = forecast.to_market_row()
        state = self._resolve_state(bid, market_row, asset_state)
        submission_gate = self._clock.submission_gate_timestamp(state)
        acceptance_notice = self._clock.acceptance_timestamp(state)

        if mode not in {"forecast", "replay", "oracle_test"}:
            return _tool_result(
                accepted=False,
                reason_codes=["unknown_agent_tool_mode"],
                submission_gate_utc=_iso_z(submission_gate),
                acceptance_notice_utc=_iso_z(acceptance_notice),
                agent_summary="rejected:unknown_agent_tool_mode",
            )
        if mode == "forecast" and forecast.source.kind == "oracle_actual":
            return _tool_result(
                accepted=False,
                reason_codes=["oracle_source_forbidden_in_forecast_mode"],
                submission_gate_utc=_iso_z(submission_gate),
                acceptance_notice_utc=_iso_z(acceptance_notice),
                verifier_stage_failed="leakage",
                agent_summary="rejected:oracle_source_forbidden_in_forecast_mode",
            )
        if forecast.issued_datetime() > submission_gate:
            return _tool_result(
                accepted=False,
                reason_codes=["forecast_after_submission_gate"],
                submission_gate_utc=_iso_z(submission_gate),
                acceptance_notice_utc=_iso_z(acceptance_notice),
                verifier_stage_failed="leakage",
                agent_summary="rejected:forecast_after_submission_gate",
            )
        if not self._clock.is_gate_open(bid, state):
            return _tool_result(
                accepted=False,
                reason_codes=["gate_closed"],
                submission_gate_utc=_iso_z(submission_gate),
                acceptance_notice_utc=_iso_z(acceptance_notice),
                verifier_stage_failed="timing",
                agent_summary="rejected:gate_closed",
            )

        return self._run_pipeline(
            bid,
            state,
            market_row,
            submission_gate,
            acceptance_notice,
            forecast=forecast,
        )

    # -- shared pipeline (physical → conformal → market → accept) ------------

    def _resolve_state(
        self,
        bid: Bid,
        market_row: dict[str, Any],
        asset_state: SimulatorAssetState | None,
    ) -> MarketState:
        return _state_from_market_row(
            market_row,
            self._scenario,
            asset_state
            or SimulatorAssetState.for_asset(
                bid.asset_id,
                thermal_soc_mwh=self._scenario.thermal_storage[bid.asset_id].initial_soc_mwh,
            ),
        )

    def _run_pipeline(
        self,
        bid: Bid,
        state: MarketState,
        market_row: dict[str, Any],
        submission_gate: Any,
        acceptance_notice: Any,
        *,
        forecast: ForecastMarketState | None = None,
    ) -> AgentMFRRToolResult:
        gate_utc = _iso_z(submission_gate)
        notice_utc = _iso_z(acceptance_notice)
        has_forecast = forecast is not None

        physical_decision = self._physical.validate_bid(bid, state.asset_states[bid.asset_id])
        physical_fields = {
            "physical_projected_power_mw": round(physical_decision.projected_power_mw, 6),
            "physical_projected_thermal_soc_mwh": round(
                physical_decision.projected_thermal_soc_mwh, 6
            ),
            "physical_remaining_capacity_mw": round(physical_decision.remaining_capacity_mw, 6),
        }
        if not physical_decision.accepted:
            reason = physical_decision.reason_code or "physical_rejected"
            return _tool_result(
                accepted=False,
                reason_codes=[reason],
                submission_gate_utc=gate_utc,
                acceptance_notice_utc=notice_utc,
                **physical_fields,
                verifier_stage_failed="physical" if has_forecast else None,
                agent_summary=f"rejected:{reason}",
            )

        interval: tuple[float, float] | None = None
        worst_case_profit_eur: float | None = None
        if forecast is not None:
            interval = forecast.interval_for_side(bid.side)
            worst_case_profit = _worst_case_profit(
                bid,
                spot_price_eur_mwh=forecast.spot_price_eur_mwh,
                price_interval_eur_mwh=interval,
            )
            worst_case_profit_eur = round(worst_case_profit, 6)
            if worst_case_profit + 1e-9 < self._tau_eur:
                required = round(self._tau_eur - worst_case_profit, 6)
                return _tool_result(
                    accepted=False,
                    reason_codes=["conformal_profit_below_threshold"],
                    submission_gate_utc=gate_utc,
                    acceptance_notice_utc=notice_utc,
                    **physical_fields,
                    forecast_interval_eur_mwh=interval,
                    worst_case_profit_eur=worst_case_profit_eur,
                    required_profit_improvement_eur=required,
                    verifier_stage_failed="conformal",
                    agent_summary="rejected:conformal_profit_below_threshold",
                )

        market_result = self._market.clear(market_row, [bid])
        if not market_result.accepted_bids:
            reason_codes = [d.reason_code or "market_rejected" for d in market_result.rejected_bids]
            return _tool_result(
                accepted=False,
                reason_codes=reason_codes,
                submission_gate_utc=gate_utc,
                acceptance_notice_utc=notice_utc,
                **physical_fields,
                model_quality=market_result.model_quality,
                market_result=market_result,
                forecast_interval_eur_mwh=interval,
                worst_case_profit_eur=worst_case_profit_eur,
                verifier_stage_failed="market" if has_forecast else None,
                agent_summary=f"rejected:{','.join(reason_codes)}",
            )

        return _tool_result(
            accepted=True,
            reason_codes=[],
            submission_gate_utc=gate_utc,
            acceptance_notice_utc=notice_utc,
            **physical_fields,
            model_quality=market_result.model_quality,
            market_result=market_result,
            forecast_interval_eur_mwh=interval,
            worst_case_profit_eur=worst_case_profit_eur,
            verifier_stage_failed=None if has_forecast else None,
            agent_summary="accepted",
        )


def _state_from_market_row(
    market_row: dict[str, Any],
    scenario: HeimdallScenario,
    asset_state: SimulatorAssetState,
) -> MarketState:
    timestamp = Bid(
        agent_id="_clock",
        asset_id=asset_state.asset_id,
        zone=market_row["zone"],
        utc_timestamp=market_row["utc_timestamp"],
        side="up",
        quantity_mwh=1.0,
        limit_price_eur_mwh=0.0,
    ).utc_timestamp
    return MarketState(
        utc_timestamp=timestamp,
        zones=scenario.zones,
        markets={str(market_row["zone"]): dict(market_row)},
        asset_states={asset_state.asset_id: asset_state},
    )


def _tool_result(
    *,
    accepted: bool,
    agent_summary: str,
    reason_codes: list[str],
    submission_gate_utc: str,
    acceptance_notice_utc: str,
    physical_projected_power_mw: float | None = None,
    physical_projected_thermal_soc_mwh: float | None = None,
    physical_remaining_capacity_mw: float | None = None,
    model_quality: PriceModelQuality | None = None,
    market_result: MFRRClearingResult | None = None,
    forecast_interval_eur_mwh: tuple[float, float] | None = None,
    worst_case_profit_eur: float | None = None,
    required_profit_improvement_eur: float | None = None,
    verifier_stage_failed: str | None = None,
) -> AgentMFRRToolResult:
    result = AgentMFRRToolResult(
        accepted=accepted,
        agent_summary=agent_summary,
        reason_codes=reason_codes,
        submission_gate_utc=submission_gate_utc,
        acceptance_notice_utc=acceptance_notice_utc,
        physical_projected_power_mw=physical_projected_power_mw,
        physical_projected_thermal_soc_mwh=physical_projected_thermal_soc_mwh,
        physical_remaining_capacity_mw=physical_remaining_capacity_mw,
        model_quality=model_quality,
        market_result=market_result,
        forecast_interval_eur_mwh=forecast_interval_eur_mwh,
        worst_case_profit_eur=worst_case_profit_eur,
        required_profit_improvement_eur=required_profit_improvement_eur,
        verifier_stage_failed=verifier_stage_failed,
        result_hash="",
    )
    return replace(result, result_hash=_hash_payload(asdict(result)))


def _iso_z(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _worst_case_profit(
    bid: Bid,
    *,
    spot_price_eur_mwh: float,
    price_interval_eur_mwh: tuple[float, float],
) -> float:
    lower, upper = price_interval_eur_mwh
    adverse_price = lower if bid.side == "up" else upper
    if bid.side == "up":
        return round(bid.quantity_mwh * (adverse_price - spot_price_eur_mwh), 6)
    return round(bid.quantity_mwh * (spot_price_eur_mwh - adverse_price), 6)


def _hash_payload(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, default=str, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
