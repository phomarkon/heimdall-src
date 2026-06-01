from __future__ import annotations

import os

import pytest

from packages.data import fetch_eds_dataset, normalize_eds_imbalance_price
from packages.pypsa_adapter import (
    build_tiny_dk_network,
    extract_heimdall_scenario,
    solve_network,
)
from packages.simulator import (
    AgentMFRRTool,
    BaselineMFRRForecaster,
    Bid,
    CalibratedMFRRPriceModel,
    SimulatorAssetState,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_HEIMDALL_INTEGRATION") != "1",
    reason="set RUN_HEIMDALL_INTEGRATION=1 to run live EDS forecast-agent validation",
)
def test_live_eds_forecast_agent_tool_does_not_need_future_actuals() -> None:
    raw = fetch_eds_dataset(
        "ImbalancePrice",
        start="2025-03-04",
        end="2025-04-10",
        price_areas=["DK1", "DK2"],
    )
    frame = normalize_eds_imbalance_price(raw)
    forecaster = BaselineMFRRForecaster.fit(frame, lookback_days=21)
    price_model = CalibratedMFRRPriceModel.fit(frame, min_samples=20)
    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")
    tool = AgentMFRRTool(
        extract_heimdall_scenario(network),
        price_model,
        tau_eur=-100.0,
    )
    forecast = forecaster.forecast(
        delivery_timestamp="2025-04-09T12:00:00Z",
        zone="DK1",
        issued_at="2025-04-09T10:45:00Z",
    )
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp=forecast.delivery_timestamp,
        side=forecast.activation_direction if forecast.activation_direction != "neutral" else "up",
        quantity_mwh=1.0,
        limit_price_eur_mwh=forecast.mfrr_up_price_median_eur_mwh,
        submitted_at_utc="2025-04-09T10:55:00Z",
    )

    result = tool.simulate_bid_from_forecast(
        bid,
        forecast,
        asset_state=SimulatorAssetState.for_asset("DK1", electric_power_mw=8.0),
    )

    assert forecast.source.kind == "baseline_conformal"
    assert result.forecast_interval_eur_mwh is not None
    assert result.worst_case_profit_eur is not None
    assert result.reason_codes != ["oracle_source_forbidden_in_forecast_mode"]
    assert result.result_hash
