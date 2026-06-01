from __future__ import annotations

import pandas as pd

from packages.simulator import Bid, SimulatorAssetState
from packages.simulator.agent_tool import AgentMFRRTool
from packages.simulator.forecast import ForecastMarketState, ForecastSource
from packages.simulator.mfrr_engine import CalibratedMFRRPriceModel


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "utc_timestamp": f"2025-03-04T00:{idx * 15:02d}:00Z",
                "zone": "DK1",
                "satisfied_demand_mw": volume_mwh / 0.25,
                "imbalance_price_eur_mwh": 50.0 + 8.0 + 2.0 * volume_mwh,
                "spot_price_eur_mwh": 50.0,
                "mfrr_marginal_price_up_eur_mwh": 50.0 + 8.0 + 2.0 * volume_mwh,
                "mfrr_marginal_price_down_eur_mwh": 50.0,
            }
            for idx, volume_mwh in enumerate([2.0, 4.0, 6.0, 8.0])
        ]
    )


def _forecast(
    *,
    issued_at: str = "2025-03-04T02:00:00Z",
    lower: float = 70.0,
    median: float = 78.0,
    upper: float = 90.0,
    source_kind: str = "baseline_conformal",
) -> ForecastMarketState:
    return ForecastMarketState(
        delivery_timestamp="2025-03-04T03:00:00Z",
        zone="DK1",
        issued_at=issued_at,
        activation_direction="up",
        activation_volume_mwh=8.0,
        spot_price_eur_mwh=50.0,
        imbalance_price_lower_eur_mwh=lower,
        imbalance_price_median_eur_mwh=median,
        imbalance_price_upper_eur_mwh=upper,
        mfrr_up_price_lower_eur_mwh=lower,
        mfrr_up_price_median_eur_mwh=median,
        mfrr_up_price_upper_eur_mwh=upper,
        mfrr_down_price_lower_eur_mwh=40.0,
        mfrr_down_price_median_eur_mwh=45.0,
        mfrr_down_price_upper_eur_mwh=50.0,
        source=ForecastSource(kind=source_kind, window_start="2025-03-04T00:00:00Z"),
    )


def _tool(scenario) -> AgentMFRRTool:
    return AgentMFRRTool(
        scenario,
        CalibratedMFRRPriceModel.fit(_history(), min_samples=4),
        tau_eur=20.0,
    )


def test_agent_tool_accepts_forecast_bid_with_positive_conformal_worst_case_profit(tiny_dk_scenario) -> None:
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T02:10:00Z",
    )

    asset_state = SimulatorAssetState.for_asset("DK1", electric_power_mw=8.0)
    first = _tool(tiny_dk_scenario).simulate_bid_from_forecast(
        bid,
        _forecast(),
        asset_state=asset_state,
    )
    second = _tool(tiny_dk_scenario).simulate_bid_from_forecast(
        bid,
        _forecast(),
        asset_state=asset_state,
    )

    assert first.accepted is True
    assert first.verifier_stage_failed is None
    assert first.worst_case_profit_eur == 40.0
    assert first.forecast_interval_eur_mwh == (70.0, 90.0)
    assert first.market_result is not None
    assert first.result_hash == second.result_hash


def test_agent_tool_rejects_forecast_bid_when_conformal_profit_is_too_low(tiny_dk_scenario) -> None:
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T02:10:00Z",
    )

    result = _tool(tiny_dk_scenario).simulate_bid_from_forecast(
        bid,
        _forecast(lower=55.0, median=78.0, upper=90.0),
        asset_state=SimulatorAssetState.for_asset("DK1", electric_power_mw=8.0),
    )

    assert result.accepted is False
    assert result.verifier_stage_failed == "conformal"
    assert result.reason_codes == ["conformal_profit_below_threshold"]
    assert result.worst_case_profit_eur == 10.0
    assert result.required_profit_improvement_eur == 10.0


def test_agent_tool_rejects_forecast_issued_after_submission_gate(tiny_dk_scenario) -> None:
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T02:10:00Z",
    )

    result = _tool(tiny_dk_scenario).simulate_bid_from_forecast(
        bid,
        _forecast(issued_at="2025-03-04T02:16:00Z"),
    )

    assert result.accepted is False
    assert result.reason_codes == ["forecast_after_submission_gate"]
    assert result.market_result is None


def test_agent_tool_rejects_oracle_source_in_forecast_mode_but_allows_replay_mode(tiny_dk_scenario) -> None:
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T02:10:00Z",
    )
    oracle = _forecast(source_kind="oracle_actual")

    forecast_result = _tool(tiny_dk_scenario).simulate_bid_from_forecast(bid, oracle)
    replay_result = _tool(tiny_dk_scenario).simulate_bid_from_forecast(
        bid,
        oracle,
        mode="replay",
        asset_state=SimulatorAssetState.for_asset("DK1", electric_power_mw=8.0),
    )

    assert forecast_result.accepted is False
    assert forecast_result.reason_codes == ["oracle_source_forbidden_in_forecast_mode"]
    assert replay_result.accepted is True
