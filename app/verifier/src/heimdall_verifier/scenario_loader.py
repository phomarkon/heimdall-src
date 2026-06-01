"""PyPSA-Eur-Sec → verifier AssetSpec mapping (per A9, docs/RESEARCH-PROPOSAL.md §5.4).

Tim's `packages/pypsa_adapter/` produces a typed `HeimdallScenario` containing
`P2HAsset` and `ThermalStorage` records sourced from a PyPSA-Eur-Sec network.
The verifier consumes a flat `AssetSpec` (q_max_mw, ramp_mw_per_min,
storage_mwh, cop, loss_per_quarter, bid_tick_eur).  This module wires the two
together, and is the *only* code A9 needs beyond the existing adapter.

Mapping (see `notes/findings/2026-05-09-pypsa-adapter-review.md`):
    q_max_mw          ← p2h.p_nom_mw
    ramp_mw_per_min   ← p2h.ramp_limit_mw_per_tick / 15.0  (15-min tick)
    storage_mwh       ← thermal_storage.e_nom_mwh
    cop               ← p2h.cop
    loss_per_quarter  ← thermal_storage.thermal_loss_per_tick
    bid_tick_eur      ← market-rule (not asset-derived); kept at default
"""

from __future__ import annotations

from pathlib import Path

from packages.pypsa_adapter import (
    HeimdallScenario,
    build_pypsa_eursec_dk_network,
    build_tiny_dk_network,
    extract_heimdall_scenario,
    load_heimdall_scenario,
    solve_network,
)

from .physical import AssetSpec


def assetspec_from_scenario(
    scenario: HeimdallScenario,
    *,
    zone: str = "DK1",
    bid_tick_eur: float = 0.01,
) -> AssetSpec:
    """Map a `HeimdallScenario`'s P2H + thermal-storage entries for ``zone``
    onto a flat verifier `AssetSpec`."""
    if zone not in scenario.p2h_assets:
        raise KeyError(f"scenario has no P2H asset for zone {zone!r}")
    if zone not in scenario.thermal_storage:
        raise KeyError(f"scenario has no thermal storage for zone {zone!r}")
    p2h = scenario.p2h_assets[zone]
    storage = scenario.thermal_storage[zone]
    return AssetSpec(
        q_max_mw=float(p2h.p_nom_mw),
        ramp_mw_per_min=float(p2h.ramp_limit_mw_per_tick) / 15.0,
        storage_mwh=float(storage.e_nom_mwh),
        cop=float(p2h.cop),
        loss_per_quarter=float(storage.thermal_loss_per_tick),
        bid_tick_eur=bid_tick_eur,
    )


def assetspec_from_bundle(path: Path | str, *, zone: str = "DK1") -> AssetSpec:
    """Load a `HeimdallScenarioBundle` JSON and produce an `AssetSpec`."""
    scenario = load_heimdall_scenario(Path(path))
    return assetspec_from_scenario(scenario, zone=zone)


def assetspec_from_tiny_dk_network(*, zone: str = "DK1") -> AssetSpec:
    """Legacy: solve the synthetic DK1+DK2 reference network and extract.

    Kept for unit-test continuity. Prefer
    :func:`assetspec_from_pypsa_eursec_dk_network` for any cell that
    feeds the verifier; that path uses the real PyPSA technology-data
    cost CSV.
    """
    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")
    scenario = extract_heimdall_scenario(network)
    return assetspec_from_scenario(scenario, zone=zone)


def assetspec_from_pypsa_eursec_dk_network(
    *,
    zone: str = "DK1",
    p_nom_p2h_mw: float = 50.0,
    storage_technology: str = "central water tank storage",
) -> AssetSpec:
    """Build the DK1+DK2 PyPSA-Eur-Sec-grounded network and extract a spec.

    All P2H and thermal-storage parameters are read from
    `data/raw/pypsa_eursec/costs_2030.csv`
    (PyPSA/technology-data on GitHub, SHA-256 pinned).  This is the
    A9-and-onwards default; the synthetic builder is retained only for
    backwards-compatible unit tests.
    """
    network = build_pypsa_eursec_dk_network(
        p_nom_p2h_mw=p_nom_p2h_mw,
        storage_technology=storage_technology,
    )
    solve_network(network, solver_name="highs")
    scenario = extract_heimdall_scenario(network)
    return assetspec_from_scenario(scenario, zone=zone)
