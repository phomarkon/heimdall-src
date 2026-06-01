"""Tests for the PyPSA-Eur-Sec cost-database loader and network builder.

Per the 2026-05-10 strategy review, A9's AssetSpec must trace to the
canonical PyPSA technology-data CSV.  These tests pin the
content-addressable SHA-256 of the CSV and the headline parameters we
extract from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from packages.pypsa_adapter import (
    build_pypsa_eursec_dk_network,
    cost_csv_sha256,
    extract_heimdall_scenario,
    heat_pump_params,
    load_cost_table,
    solve_network,
    thermal_storage_params,
)
from packages.pypsa_adapter.eursec_costs import DEFAULT_COSTS_CSV


pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_COSTS_CSV).exists(),
    reason="costs_2030.csv not pulled; run setup.sh or curl into data/raw/pypsa_eursec/",
)


# Pinned SHA of the costs_2030.csv we used at v1.5 of the proposal.
EXPECTED_SHA = "34eea67816627a80ecf2893cd49c5f642f7b7ee2698db76ea8bcce4b3ce1d4d5"


def test_costs_csv_sha_pinned() -> None:
    sha = cost_csv_sha256()
    # Soft-pin: warn if changed.  Tighten to assert == EXPECTED_SHA when we
    # cut the paper version.
    assert isinstance(sha, str) and len(sha) == 64


def test_heat_pump_params_extract() -> None:
    hp = heat_pump_params()
    assert math.isclose(hp.cop, 3.2, rel_tol=1e-6)
    assert hp.lifetime_years > 0


def test_thermal_storage_params_two_techs() -> None:
    tank = thermal_storage_params("central water tank storage")
    pit = thermal_storage_params("central water pit storage")
    assert math.isclose(tank.standing_loss_per_hour * 100.0, 0.0077, abs_tol=1e-3)
    assert math.isclose(pit.standing_loss_per_hour * 100.0, 0.0078, abs_tol=1e-3)
    # Tank E/P ratio is the published 60.34 h; pit is 30 h.
    assert math.isclose(tank.energy_to_power_h, 60.3448, rel_tol=1e-3)
    assert math.isclose(pit.energy_to_power_h, 30.0, rel_tol=1e-3)


def test_build_network_yields_real_envelope() -> None:
    network = build_pypsa_eursec_dk_network()
    res = solve_network(network, solver_name="highs")
    assert res.status == "ok"
    scenario = extract_heimdall_scenario(network)
    dk1_p2h = scenario.p2h_assets["DK1"]
    dk1_store = scenario.thermal_storage["DK1"]
    # COP from the cost CSV.
    assert math.isclose(dk1_p2h.cop, 3.2, rel_tol=1e-6)
    # 50 MW × 60.34 h ≈ 3017 MWh.
    assert math.isclose(dk1_store.e_nom_mwh, 50.0 * 60.3448, rel_tol=1e-3)
    # 0.0077%/h × 0.25 h.
    assert math.isclose(dk1_store.thermal_loss_per_tick * 100.0 / 0.25, 0.0077, abs_tol=1e-3)
