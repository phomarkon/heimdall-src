import pytest

from packages.pypsa_adapter import (
    build_tiny_dk_network,
    extract_heimdall_scenario,
    solve_network,
)
from packages.simulator import Bid, PhysicalConstraintProvider, SimulatorAssetState


def _provider() -> PhysicalConstraintProvider:
    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")
    return PhysicalConstraintProvider(extract_heimdall_scenario(network))


def test_physical_verifier_rejects_capacity_violation() -> None:
    provider = _provider()
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=13.0,
        limit_price_eur_mwh=40.0,
    )

    decision = provider.validate_bid(bid, SimulatorAssetState.for_asset("DK1"))

    assert not decision.accepted
    assert decision.reason_code == "capacity_exceeded"


def test_physical_verifier_rejects_ramp_violation() -> None:
    provider = _provider()
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=7.0,
        limit_price_eur_mwh=40.0,
    )

    decision = provider.validate_bid(bid, SimulatorAssetState.for_asset("DK1"))

    assert not decision.accepted
    assert decision.reason_code == "ramp_exceeded"


def test_physical_verifier_rejects_storage_violation() -> None:
    provider = _provider()
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
    )
    state = SimulatorAssetState.for_asset("DK1", thermal_soc_mwh=95.0)

    decision = provider.validate_bid(bid, state)

    assert not decision.accepted
    assert decision.reason_code == "storage_exceeded"


def test_physical_verifier_accepts_feasible_bid_and_reports_remaining_envelope() -> None:
    provider = _provider()
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
    )

    decision = provider.validate_bid(bid, SimulatorAssetState.for_asset("DK1"))

    assert decision.accepted
    assert decision.reason_code is None
    assert decision.remaining_capacity_mw == pytest.approx(34.0)
    assert decision.projected_thermal_soc_mwh == pytest.approx(51.996)


def test_physical_verifier_applies_thermal_loss_before_charging() -> None:
    provider = _provider()
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
    )

    decision = provider.validate_bid(
        bid, SimulatorAssetState.for_asset("DK1", thermal_soc_mwh=80.0)
    )

    assert decision.accepted
    assert decision.projected_thermal_soc_mwh == pytest.approx(91.992)
