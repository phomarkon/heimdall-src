"""Tests for the pypsa-scenario service."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from heimdall_pypsa_scenario import get_assetspec, get_scenario

from packages.pypsa_adapter.eursec_costs import DEFAULT_COSTS_CSV

# The service composes the AssetSpec from the PyPSA-Eur-Sec cost CSV. Skip
# cleanly when that external artefact has not been pulled (matches the pattern
# in tests/test_pypsa_eursec_costs.py) so CPU CI stays green.
pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_COSTS_CSV).exists(),
    reason="costs_2030.csv not pulled; run setup.sh or curl into data/raw/pypsa_eursec/",
)


def test_assetspec_dk1_matches_pypsa_eursec_costs() -> None:
    spec = get_assetspec("DK1")
    assert math.isclose(spec.cop, 3.2, rel_tol=1e-6)
    assert math.isclose(spec.q_max_mw, 50.0, rel_tol=1e-6)
    # 50 MW × 60.34 h E/P ratio.
    assert math.isclose(spec.storage_mwh, 50.0 * 60.3448, rel_tol=1e-3)
    assert spec.provenance["source"].startswith("PyPSA-Eur-Sec")
    assert len(spec.provenance["csv_sha256"]) == 64


def test_scenario_dk1_dk2_round_trips() -> None:
    s = get_scenario("DK1")
    assert "p2h_assets" in s.payload
    assert "thermal_storage" in s.payload
    assert "DK1" in s.payload["p2h_assets"]
    assert "DK2" in s.payload["p2h_assets"]
