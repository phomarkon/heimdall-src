from __future__ import annotations

import pandas as pd

from packages.simulator import Bid, SimulatorAssetState
from packages.simulator.agent_tool import AgentMFRRTool
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
        + [
            {
                "utc_timestamp": f"2025-03-04T01:{idx * 15:02d}:00Z",
                "zone": "DK2",
                "satisfied_demand_mw": -(volume_mwh / 0.25),
                "imbalance_price_eur_mwh": 55.0 - (7.0 + 2.5 * volume_mwh),
                "spot_price_eur_mwh": 55.0,
                "mfrr_marginal_price_up_eur_mwh": 55.0,
                "mfrr_marginal_price_down_eur_mwh": 55.0 - (7.0 + 2.5 * volume_mwh),
            }
            for idx, volume_mwh in enumerate([2.0, 4.0, 6.0, 8.0])
        ]
    )


def test_agent_tool_returns_market_physical_and_quality_signal_for_feasible_bid(tiny_dk_scenario) -> None:
    scenario = tiny_dk_scenario
    tool = AgentMFRRTool(
        scenario,
        CalibratedMFRRPriceModel.fit(_history(), min_samples=4),
    )
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
    market_row = {
        "utc_timestamp": "2025-03-04T03:00:00Z",
        "zone": "DK1",
        "satisfied_demand_mw": 32.0,
        "imbalance_price_eur_mwh": 74.0,
        "spot_price_eur_mwh": 50.0,
        "mfrr_marginal_price_up_eur_mwh": 74.0,
        "mfrr_marginal_price_down_eur_mwh": 50.0,
    }

    result = tool.simulate_bid(
        bid,
        market_row,
        asset_state=SimulatorAssetState.for_asset("DK1", electric_power_mw=8.0),
    )

    assert result.accepted is True
    assert result.reason_codes == []
    assert result.submission_gate_utc == "2025-03-04T02:15:00Z"
    assert result.acceptance_notice_utc == "2025-03-04T02:52:30Z"
    assert result.physical_projected_power_mw == 0.0
    assert result.market_result is not None
    assert result.market_result.price_delta_eur_mwh == -4.0
    assert result.model_quality is not None
    assert result.model_quality.fallback_level == "zone_direction"
    assert result.agent_summary == "accepted"
    assert result.result_hash


def test_agent_tool_rejects_late_bid_before_market_clearing(tiny_dk_scenario) -> None:
    tool = AgentMFRRTool(tiny_dk_scenario, CalibratedMFRRPriceModel.fit(_history(), min_samples=4))
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T02:16:00Z",
    )

    result = tool.simulate_bid(
        bid,
        {
            "utc_timestamp": "2025-03-04T03:00:00Z",
            "zone": "DK1",
            "satisfied_demand_mw": 32.0,
            "imbalance_price_eur_mwh": 74.0,
            "spot_price_eur_mwh": 50.0,
            "mfrr_marginal_price_up_eur_mwh": 74.0,
            "mfrr_marginal_price_down_eur_mwh": 50.0,
        },
    )

    assert result.accepted is False
    assert result.reason_codes == ["gate_closed"]
    assert result.market_result is None
    assert result.agent_summary == "rejected:gate_closed"


def test_agent_tool_rejects_physical_violation_before_market_clearing(tiny_dk_scenario) -> None:
    tool = AgentMFRRTool(tiny_dk_scenario, CalibratedMFRRPriceModel.fit(_history(), min_samples=4))
    bid = Bid(
        agent_id="focal",
        asset_id="DK2",
        zone="DK2",
        utc_timestamp="2025-03-04T03:00:00Z",
        side="down",
        quantity_mwh=20.0,
        limit_price_eur_mwh=20.0,
        submitted_at_utc="2025-03-04T02:10:00Z",
    )

    result = tool.simulate_bid(
        bid,
        {
            "utc_timestamp": "2025-03-04T03:00:00Z",
            "zone": "DK2",
            "satisfied_demand_mw": -32.0,
            "imbalance_price_eur_mwh": 28.0,
            "spot_price_eur_mwh": 55.0,
            "mfrr_marginal_price_up_eur_mwh": 55.0,
            "mfrr_marginal_price_down_eur_mwh": 28.0,
        },
    )

    assert result.accepted is False
    assert result.reason_codes == ["capacity_exceeded"]
    assert result.market_result is None
    assert result.physical_remaining_capacity_mw == 50.0
