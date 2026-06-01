from pathlib import Path

import pandas as pd

from packages.pypsa_adapter import (
    build_tiny_dk_network,
    export_network,
    load_network,
    network_snapshots_utc,
    solve_network,
)


def test_tiny_dk_network_has_required_utc_snapshots_and_components() -> None:
    network = build_tiny_dk_network()

    snapshots = network_snapshots_utc(network)

    assert len(snapshots) == 16
    assert snapshots.tz is not None
    assert snapshots[0] == pd.Timestamp("2025-03-04T00:00:00Z")
    assert snapshots.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert {"DK1 electricity", "DK2 electricity", "DK1 heat", "DK2 heat"}.issubset(
        set(network.buses.index)
    )
    assert {"DK1 P2H", "DK2 P2H", "DK1-DK2"}.issubset(set(network.links.index))
    assert {"DK1 wind", "DK2 solar"}.issubset(set(network.generators.index))
    assert {"DK1 thermal store", "DK2 thermal store"}.issubset(set(network.stores.index))


def test_tiny_dk_network_solves_with_highs() -> None:
    network = build_tiny_dk_network()

    result = solve_network(network, solver_name="highs")

    assert result.status == "ok"
    assert result.condition == "optimal"
    assert not network.generators_t.p.empty
    assert not network.links_t.p0.empty


def test_network_round_trips_through_netcdf(tmp_path: Path) -> None:
    network = build_tiny_dk_network()
    path = tmp_path / "tiny-dk.nc"

    export_network(network, path)
    reloaded = load_network(path)

    assert list(network_snapshots_utc(reloaded)) == list(network_snapshots_utc(network))
    assert set(reloaded.buses.index) == set(network.buses.index)
    assert set(reloaded.links.index) == set(network.links.index)
    assert set(reloaded.stores.index) == set(network.stores.index)
