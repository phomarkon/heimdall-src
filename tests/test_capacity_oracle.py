"""Tests for the grounded capacity-capped fair oracle in tools/evaluation/rescore_runs.py.

The capacity oracle's only spec-grounded input is P2H power (PyPSA-Eur-Sec DK1 p_nom = 50 MW).
These tests pin that grounding, the P2H-only grounded-detection, and the cap behaviour against
the real system activation volume.
"""

from __future__ import annotations

import pandas as pd
import pytest
from heimdall_contracts import PersonaArchetype
from heimdall_personas.archetypes import ARCHETYPE_DEFAULTS

from tools.evaluation.rescore_runs import (
    MTU_HOURS,
    P2H_CAPACITY_MW,
    P2H_RAMP_LIMIT_MW_PER_TICK,
    _p2h_focal_capacity_mwh_per_mtu,
    _per_run_metrics,
)


def _bids(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _p2h_bid(agent_id: str, *, side="up", status="filled", quantity=2.0, cleared=2.0, profit=40.0) -> dict:
    return {
        "timestamp_utc": "2026-04-02T05:30:00Z",
        "zone": "DK1",
        "agent_id": agent_id,
        "archetype": "p2h",
        "side": side,
        "status": status,
        "quantity_mwh": quantity,
        "cleared_mwh": cleared,
        "realized_profit_eur": profit,
    }


def _truth(direction="up", volume=10.0, settlement=120.0, spot=100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-04-02T05:30:00Z",
                "zone": "DK1",
                "activation_direction": direction,
                "activated_volume_mwh": volume,
                "settlement_price_eur_mwh": settlement,
                "spot_price_eur_mwh": spot,
                "imbalance_price_eur_mwh": settlement,
            }
        ]
    )


def test_p2h_capacity_constant_matches_canonical_spec() -> None:
    # Single source of truth: the rescorer nominal constant must equal the canonical archetype spec.
    assert P2H_CAPACITY_MW == float(ARCHETYPE_DEFAULTS[PersonaArchetype.P2H]["capacity_mw"])
    assert P2H_CAPACITY_MW == 50.0


@pytest.mark.slow
def test_p2h_ramp_constant_matches_scenario() -> None:
    # The oracle's ramp-limited deliverable must equal the actual scenario asset spec ramp,
    # not a hardcoded guess. Builds the p2h_dk1_pypsa scenario (HiGHS solve ~2s).
    from packages.pypsa_adapter import (  # noqa: PLC0415  (lazy: heavy pypsa deps only for this slow test)
        build_tiny_dk_network,
        extract_heimdall_scenario,
        solve_network,
    )

    net = build_tiny_dk_network()
    solve_network(net, solver_name="highs")
    scenario = extract_heimdall_scenario(net)
    assert P2H_RAMP_LIMIT_MW_PER_TICK == scenario.p2h_assets["DK1"].ramp_limit_mw_per_tick == 25.0


def test_p2h_focal_capacity_counts_p2h_agents() -> None:
    bids = _bids([_p2h_bid("a0"), _p2h_bid("a1")])
    cap, grounded, n_p2h = _p2h_focal_capacity_mwh_per_mtu(bids)
    assert grounded is True
    assert n_p2h == 2
    assert cap == 2 * P2H_RAMP_LIMIT_MW_PER_TICK * MTU_HOURS  # 2 * 25 * 0.25 = 12.5 MWh/MTU


def test_p2h_focal_capacity_grounded_in_heterogeneous_society() -> None:
    # One P2H focal among non-P2H competitors: grounded on the P2H focal only; non-P2H MW
    # (assumed, not specs) are excluded from the denominator, never fabricated.
    bids = _bids([_p2h_bid("a0"), {**_p2h_bid("a1"), "archetype": "generator"}])
    cap, grounded, n_p2h = _p2h_focal_capacity_mwh_per_mtu(bids)
    assert grounded is True
    assert n_p2h == 1
    assert cap == 1 * P2H_RAMP_LIMIT_MW_PER_TICK * MTU_HOURS  # only the P2H focal: 25 * 0.25 = 6.25


def test_p2h_focal_capacity_suppressed_without_p2h() -> None:
    bids = _bids([{**_p2h_bid("a0"), "archetype": "generator"}, {**_p2h_bid("a1"), "archetype": "wind"}])
    cap, grounded, n_p2h = _p2h_focal_capacity_mwh_per_mtu(bids)
    assert grounded is False
    assert n_p2h == 0
    assert cap == 0.0


def test_capacity_oracle_caps_at_system_volume_and_capture_in_unit_range() -> None:
    # 1 P2H agent -> ramp-limited cap = 6.25 MWh/MTU. System activated 100 MWh, spread 20 -> bound.
    bids = _bids([_p2h_bid("a0", profit=40.0)])
    metrics = _per_run_metrics(bids, _truth(volume=100.0), [0.0], [0.0], {}, 42)
    assert metrics["capacity_grounded"] is True
    # capacity oracle = spread * min(volume, cap) = 20 * min(100, 6.25) = 125
    assert metrics["oracle_capacity_eur"] == 125.0
    # strictly below the (physically unreachable) uncapped oracle = 20 * 100 = 2000
    assert metrics["oracle_uncapped_eur"] == 2000.0
    assert metrics["capture_capacity"] == round(40.0 / 125.0, 6)
    assert 0.0 <= metrics["capture_capacity"] <= 1.0


def test_capacity_oracle_equals_uncapped_when_volume_below_capacity() -> None:
    # System volume (5) below ramp-limited capacity (6.25) -> capacity cap does not bind.
    bids = _bids([_p2h_bid("a0", profit=40.0)])
    metrics = _per_run_metrics(bids, _truth(volume=5.0), [0.0], [0.0], {}, 42)
    assert metrics["oracle_capacity_eur"] == metrics["oracle_uncapped_eur"] == 100.0


def test_capacity_capture_suppressed_without_p2h() -> None:
    bids = _bids([{**_p2h_bid("a0"), "archetype": "generator"}, {**_p2h_bid("a1"), "archetype": "wind"}])
    metrics = _per_run_metrics(bids, _truth(volume=100.0), [0.0], [0.0], {}, 42)
    assert metrics["capacity_grounded"] is False
    assert metrics["oracle_capacity_eur"] is None
    assert metrics["capture_capacity"] is None


def test_capture_capacity_is_p2h_focal_excludes_non_p2h_profit() -> None:
    # Heterogeneous society: P2H focal earns 40, a generator earns 1000. The focal capacity
    # capture must use ONLY the P2H 40 in the numerator (the focal is the thesis subject).
    bids = _bids([
        _p2h_bid("a0", profit=40.0),
        {**_p2h_bid("a1", profit=1000.0), "archetype": "generator"},
    ])
    metrics = _per_run_metrics(bids, _truth(volume=100.0), [0.0], [0.0], {}, 42)
    assert metrics["capacity_grounded"] is True
    assert metrics["n_p2h_agents"] == 1
    assert metrics["realized_p2h_eur"] == 40.0
    # oracle = spread 20 * min(100, 6.25) = 125; focal capture = 40/125, NOT 1040/125
    assert metrics["oracle_capacity_eur"] == 125.0
    assert metrics["capture_capacity"] == round(40.0 / 125.0, 6)
