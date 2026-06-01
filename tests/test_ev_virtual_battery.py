from __future__ import annotations

from packages.simulator import Bid, EVFleetState, EVVirtualBatterySimulator


def test_ev_virtual_battery_accepts_feasible_bid() -> None:
    simulator = EVVirtualBatterySimulator(
        EVFleetState(
            asset_id="ev-fleet",
            capacity_mw=8.0,
            energy_mwh=20.0,
            soc_mwh=10.0,
            availability_share=1.0,
        )
    )
    result = simulator.simulate_bid(
        Bid(
            agent_id="agent-ev",
            asset_id="ev-fleet",
            zone="DK1",
            utc_timestamp="2025-03-04T03:00:00Z",
            side="up",
            quantity_mwh=1.0,
            limit_price_eur_mwh=80.0,
        )
    )
    assert result.accepted is True
    assert result.authority == "authoritative"
    assert result.simulator_kind == "ev_virtual_battery"


def test_ev_virtual_battery_rejects_capacity_violation() -> None:
    simulator = EVVirtualBatterySimulator(
        EVFleetState(
            asset_id="ev-fleet",
            capacity_mw=4.0,
            energy_mwh=20.0,
            soc_mwh=10.0,
            availability_share=0.5,
        )
    )
    result = simulator.simulate_bid(
        Bid(
            agent_id="agent-ev",
            asset_id="ev-fleet",
            zone="DK1",
            utc_timestamp="2025-03-04T03:00:00Z",
            side="up",
            quantity_mwh=1.0,
            limit_price_eur_mwh=80.0,
        )
    )
    assert result.accepted is False
    assert result.failed_stage == "capacity"
    assert result.reason_codes == ["ev_capacity_exceeded"]


def test_ev_virtual_battery_rejects_soc_violation() -> None:
    simulator = EVVirtualBatterySimulator(
        EVFleetState(
            asset_id="ev-fleet",
            capacity_mw=8.0,
            energy_mwh=20.0,
            soc_mwh=0.2,
            availability_share=1.0,
        )
    )
    result = simulator.simulate_bid(
        Bid(
            agent_id="agent-ev",
            asset_id="ev-fleet",
            zone="DK1",
            utc_timestamp="2025-03-04T03:00:00Z",
            side="up",
            quantity_mwh=1.0,
            limit_price_eur_mwh=80.0,
        )
    )
    assert result.accepted is False
    assert result.failed_stage == "soc"
    assert result.reason_codes == ["ev_soc_lower_exceeded"]
