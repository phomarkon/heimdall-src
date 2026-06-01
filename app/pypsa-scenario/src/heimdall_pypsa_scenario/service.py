"""PyPSA-scenario FastAPI service. docs/RESEARCH-PROPOSAL.md §6.2 service #11.

Wraps `packages.pypsa_adapter` so the verifier and Tim's focal-orchestrator
can fetch the PyPSA-Eur-Sec-derived ``AssetSpec`` over HTTP rather than
importing the heavy PyPSA stack themselves.

API:
  GET /assetspec?zone=DK1                  AssetSpec JSON (verifier-compatible)
  GET /scenario?zone=DK1                   Full HeimdallScenario.to_dict()
  GET /healthz

Cached process-locally; the underlying CSV is content-addressable so a
single load + solve at start-up is enough.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


class AssetSpecResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone: str
    q_max_mw: float
    ramp_mw_per_min: float
    storage_mwh: float
    cop: float
    loss_per_quarter: float
    bid_tick_eur: float
    provenance: dict[str, Any]


class ScenarioResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict[str, Any]


@lru_cache(maxsize=4)
def _cached_scenario(p_nom_mw: float = 50.0,
                     storage_technology: str = "central water tank storage"):
    from packages.pypsa_adapter import (
        build_pypsa_eursec_dk_network,
        cost_csv_sha256,
        extract_heimdall_scenario,
        solve_network,
    )
    network = build_pypsa_eursec_dk_network(
        p_nom_p2h_mw=p_nom_mw, storage_technology=storage_technology
    )
    solve_network(network, solver_name="highs")
    return extract_heimdall_scenario(network), cost_csv_sha256()


def get_assetspec(zone: str = "DK1") -> AssetSpecResponse:
    scenario, sha = _cached_scenario()
    p2h = scenario.p2h_assets[zone]
    storage = scenario.thermal_storage[zone]
    return AssetSpecResponse(
        zone=zone,
        q_max_mw=float(p2h.p_nom_mw),
        ramp_mw_per_min=float(p2h.ramp_limit_mw_per_tick) / 15.0,
        storage_mwh=float(storage.e_nom_mwh),
        cop=float(p2h.cop),
        loss_per_quarter=float(storage.thermal_loss_per_tick),
        bid_tick_eur=0.01,
        provenance={
            "source": "PyPSA-Eur-Sec via PyPSA/technology-data costs_2030.csv",
            "csv_sha256": sha,
        },
    )


def get_scenario(zone: str = "DK1") -> ScenarioResponse:
    scenario, _ = _cached_scenario()
    return ScenarioResponse(payload=scenario.to_dict())


app = FastAPI(title="heimdall-pypsa-scenario", version="0.1.0")


@app.get("/assetspec", response_model=AssetSpecResponse)
def assetspec_route(zone: Literal["DK1", "DK2"] = "DK1") -> AssetSpecResponse:
    return get_assetspec(zone)


@app.get("/scenario", response_model=ScenarioResponse)
def scenario_route(zone: Literal["DK1", "DK2"] = "DK1") -> ScenarioResponse:
    return get_scenario(zone)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
