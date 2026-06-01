from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from packages.data import file_sha256

from ._pypsa import pypsa
from .network import network_snapshots_utc


@dataclass(frozen=True)
class P2HAsset:
    asset_id: str
    zone: str
    electric_bus: str
    heat_bus: str
    p_nom_mw: float
    ramp_limit_mw_per_tick: float
    cop: float


@dataclass(frozen=True)
class ThermalStorage:
    zone: str
    asset_id: str
    e_nom_mwh: float
    initial_soc_mwh: float
    thermal_loss_per_tick: float = 0.0001


@dataclass(frozen=True)
class Interconnector:
    link_id: str
    from_zone: str
    to_zone: str
    capacity_mw: float


@dataclass(frozen=True)
class EVFleetAsset:
    asset_id: str
    zone: str
    capacity_mw: float
    energy_mwh: float
    initial_soc_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    availability_share: float


@dataclass(frozen=True)
class WindAsset:
    asset_id: str
    zone: str
    generator_id: str
    p_nom_mw: float
    availability_share: float
    ramp_limit_mw_per_tick: float
    marginal_cost_eur_mwh: float
    down_regulation_supported: bool = False


@dataclass(frozen=True)
class GeneratorAsset:
    asset_id: str
    zone: str
    generator_id: str
    p_nom_mw: float
    initial_dispatch_mw: float
    ramp_limit_mw_per_tick: float
    marginal_cost_eur_mwh: float
    availability_share: float = 1.0


@dataclass(frozen=True)
class RenewablesAsset:
    asset_id: str
    zone: str
    component_ids: list[str]
    p_nom_mw: float
    availability_share: float
    wind_share: float
    solar_share: float
    ramp_limit_mw_per_tick: float
    marginal_cost_eur_mwh: float
    down_regulation_supported: bool = False


@dataclass(frozen=True)
class RetailerAsset:
    asset_id: str
    zone: str
    flexible_load_mw: float
    max_event_duration_ticks: int
    rebound_ticks: int
    rebound_fraction: float
    availability_share: float


@dataclass(frozen=True)
class HeimdallScenario:
    zones: list[str]
    snapshots_utc: list[str]
    p2h_assets: dict[str, P2HAsset]
    thermal_storage: dict[str, ThermalStorage]
    interconnectors: dict[str, Interconnector]
    ev_fleets: dict[str, EVFleetAsset] | None = None
    wind_assets: dict[str, WindAsset] | None = None
    generator_assets: dict[str, GeneratorAsset] | None = None
    renewables_assets: dict[str, RenewablesAsset] | None = None
    retailer_assets: dict[str, RetailerAsset] | None = None
    asset_provenance: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        physical_envelope = {
            zone: {
                "asset_capacity_mw": asset.p_nom_mw,
                "ramp_mw_per_tick": asset.ramp_limit_mw_per_tick,
                "cop_profile": {
                    "type": "constant",
                    "value": asset.cop,
                    "source": "PyPSA p2h link efficiency",
                },
                "thermal_storage_mwh": self.thermal_storage[zone].e_nom_mwh,
                "thermal_loss_per_tick": self.thermal_storage[
                    zone
                ].thermal_loss_per_tick,
                "interconnector_capacity_mw": _zone_interconnector_capacity(
                    zone, self.interconnectors
                ),
            }
            for zone, asset in self.p2h_assets.items()
        }
        return {
            "schema_version": "1.0.0",
            "zones": self.zones,
            "snapshots_utc": self.snapshots_utc,
            "provenance": self.provenance or _default_provenance(),
            "physical_envelope": physical_envelope,
            "p2h_assets": {
                zone: asdict(asset) for zone, asset in self.p2h_assets.items()
            },
            "thermal_storage": {
                zone: asdict(storage)
                for zone, storage in self.thermal_storage.items()
            },
            "interconnectors": {
                link_id: asdict(link)
                for link_id, link in self.interconnectors.items()
            },
            "ev_fleets": {
                zone: asdict(asset) for zone, asset in (self.ev_fleets or {}).items()
            },
            "wind_assets": {
                zone: asdict(asset) for zone, asset in (self.wind_assets or {}).items()
            },
            "generator_assets": {
                zone: asdict(asset) for zone, asset in (self.generator_assets or {}).items()
            },
            "renewables_assets": {
                zone: asdict(asset) for zone, asset in (self.renewables_assets or {}).items()
            },
            "retailer_assets": {
                zone: asdict(asset) for zone, asset in (self.retailer_assets or {}).items()
            },
            "asset_provenance": self.asset_provenance or _default_asset_provenance(),
        }


@dataclass(frozen=True)
class HeimdallScenarioBundle:
    path: Path
    scenario_path: Path
    dispatch_path: Path
    manifest_path: Path


def _zone(name: str) -> str:
    return name.split(" ", 1)[0].split("-", 1)[0]


def _iso_z(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _schema_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_provenance() -> dict[str, Any]:
    return {
        "scenario_id": "dk1-dk2-p2h-v0",
        "physical_source": "compact PyPSA-derived DK1/DK2 P2H scenario",
        "parameter_basis": [
            "Heimdall proposal counterfactual: 50 MW P2H + 100 MWh thermal storage",
            "PyPSA link/store executable sanity check",
        ],
        "full_pypsa_eur_required": False,
    }


def _default_asset_provenance() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "basis": "compact PyPSA scenario plus documented thesis envelope assumptions",
        "notes": [
            "P2H parameters are extracted from PyPSA links/stores.",
            "Wind/generator/renewables capacities are extracted from PyPSA generator components.",
            "EV and retailer assets are explicit scenario assumptions until a public fleet/flex-load dataset is wired.",
        ],
    }


def _zone_interconnector_capacity(
    zone: str, interconnectors: dict[str, Interconnector]
) -> float:
    capacities = [
        link.capacity_mw
        for link in interconnectors.values()
        if link.from_zone == zone or link.to_zone == zone
    ]
    return max(capacities) if capacities else 0.0


def _extract_p2h_assets(network: pypsa.Network, zones: list[str] | None = None) -> dict[str, P2HAsset]:
    p2h_assets: dict[str, P2HAsset] = {}
    p2h_links = network.links[network.links["carrier"] == "p2h"]
    for link_id, row in p2h_links.iterrows():
        zone = _zone(str(link_id))
        ramp_limit = row.get("ramp_limit_up")
        ramp_limit_mw = (
            float(row.p_nom)
            if pd.isna(ramp_limit)
            else float(row.p_nom) * float(ramp_limit)
        )
        p2h_assets[zone] = P2HAsset(
            asset_id=zone,
            zone=zone,
            electric_bus=str(row.bus0),
            heat_bus=str(row.bus1),
            p_nom_mw=float(row.p_nom),
            ramp_limit_mw_per_tick=ramp_limit_mw,
            cop=float(row.efficiency),
        )
    return p2h_assets


def _extract_thermal_storage(network: pypsa.Network, zones: list[str] | None = None) -> dict[str, ThermalStorage]:
    thermal_storage: dict[str, ThermalStorage] = {}
    stores = network.stores[network.stores["carrier"] == "thermal-storage"]
    for store_id, row in stores.iterrows():
        zone = _zone(str(store_id))
        loss = getattr(row, "standing_loss", 0.0001)
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            loss = 0.0001
        if loss == 0.0:
            loss = 0.0001  # synthetic toy default for legacy networks
        thermal_storage[zone] = ThermalStorage(
            zone=zone,
            asset_id=str(store_id),
            e_nom_mwh=float(row.e_nom),
            initial_soc_mwh=float(row.e_initial),
            thermal_loss_per_tick=loss,
        )
    return thermal_storage


def _extract_interconnectors(network: pypsa.Network) -> dict[str, Interconnector]:
    interconnectors: dict[str, Interconnector] = {}
    links = network.links[network.links["carrier"] == "interconnector"]
    for link_id, row in links.iterrows():
        interconnectors[str(link_id)] = Interconnector(
            link_id=str(link_id),
            from_zone=_zone(str(row.bus0)),
            to_zone=_zone(str(row.bus1)),
            capacity_mw=float(row.p_nom),
        )
    return interconnectors


def _extract_wind_assets(network: pypsa.Network, zone_generators: pd.DataFrame, zone: str) -> WindAsset | None:
    wind_rows = zone_generators[zone_generators["carrier"] == "wind"]
    if wind_rows.empty:
        return None
    wind_id, wind = next(iter(wind_rows.iterrows()))
    return WindAsset(
        asset_id=zone,
        zone=zone,
        generator_id=str(wind_id),
        p_nom_mw=float(wind.p_nom),
        availability_share=_availability_share(network, str(wind_id), default=0.45),
        ramp_limit_mw_per_tick=float(wind.p_nom),
        marginal_cost_eur_mwh=float(getattr(wind, "marginal_cost", 0.0) or 0.0),
    )


def _extract_generator_assets(zone_generators: pd.DataFrame, zone: str) -> GeneratorAsset | None:
    dispatchable_rows = zone_generators[~zone_generators["carrier"].isin(["wind", "solar"])]
    if dispatchable_rows.empty:
        return None
    gen_id, gen = next(iter(dispatchable_rows.iterrows()))
    return GeneratorAsset(
        asset_id=zone,
        zone=zone,
        generator_id=str(gen_id),
        p_nom_mw=float(gen.p_nom),
        initial_dispatch_mw=float(gen.p_nom) * 0.55,
        ramp_limit_mw_per_tick=float(gen.p_nom) * 0.35,
        marginal_cost_eur_mwh=float(getattr(gen, "marginal_cost", 55.0) or 55.0),
    )


def _extract_renewables_assets(network: pypsa.Network, zone_generators: pd.DataFrame, zone: str) -> RenewablesAsset | None:
    wind_rows = zone_generators[zone_generators["carrier"] == "wind"]
    solar_rows = zone_generators[zone_generators["carrier"] == "solar"]
    portfolio = pd.concat([wind_rows, solar_rows])
    if portfolio.empty:
        return None
    p_nom = float(portfolio["p_nom"].sum())
    wind_nom = float(wind_rows["p_nom"].sum()) if not wind_rows.empty else 0.0
    solar_nom = float(solar_rows["p_nom"].sum()) if not solar_rows.empty else 0.0
    return RenewablesAsset(
        asset_id=zone,
        zone=zone,
        component_ids=[str(idx) for idx in portfolio.index],
        p_nom_mw=p_nom,
        availability_share=_portfolio_availability_share(network, portfolio.index, default=0.5),
        wind_share=wind_nom / p_nom if p_nom > 0 else 0.0,
        solar_share=solar_nom / p_nom if p_nom > 0 else 0.0,
        ramp_limit_mw_per_tick=p_nom,
        marginal_cost_eur_mwh=float(portfolio["marginal_cost"].mean()) if "marginal_cost" in portfolio else 0.0,
    )


def extract_heimdall_scenario(network: pypsa.Network) -> HeimdallScenario:
    snapshots = [_iso_z(ts) for ts in network_snapshots_utc(network)]
    p2h_assets = _extract_p2h_assets(network)
    thermal_storage = _extract_thermal_storage(network)
    interconnectors = _extract_interconnectors(network)
    ev_fleets: dict[str, EVFleetAsset] = {}
    wind_assets: dict[str, WindAsset] = {}
    generator_assets: dict[str, GeneratorAsset] = {}
    renewables_assets: dict[str, RenewablesAsset] = {}
    retailer_assets: dict[str, RetailerAsset] = {}

    zones = sorted(set(p2h_assets) | set(thermal_storage))
    for zone in zones:
        zone_generators = network.generators[
            network.generators["bus"].astype(str).str.startswith(zone)
        ]

        wind_asset = _extract_wind_assets(network, zone_generators, zone)
        if wind_asset is not None:
            wind_assets[zone] = wind_asset

        gen_asset = _extract_generator_assets(zone_generators, zone)
        if gen_asset is not None:
            generator_assets[zone] = gen_asset

        ren_asset = _extract_renewables_assets(network, zone_generators, zone)
        if ren_asset is not None:
            renewables_assets[zone] = ren_asset

        ev_fleets[zone] = EVFleetAsset(
            asset_id=zone,
            zone=zone,
            capacity_mw=20.0,
            energy_mwh=80.0,
            initial_soc_mwh=40.0,
            charge_efficiency=0.92,
            discharge_efficiency=0.92,
            availability_share=0.75,
        )
        retailer_assets[zone] = RetailerAsset(
            asset_id=zone,
            zone=zone,
            flexible_load_mw=12.0,
            max_event_duration_ticks=4,
            rebound_ticks=4,
            rebound_fraction=0.35,
            availability_share=0.8,
        )

    return HeimdallScenario(
        zones=zones,
        snapshots_utc=snapshots,
        p2h_assets={zone: p2h_assets[zone] for zone in zones},
        thermal_storage={zone: thermal_storage[zone] for zone in zones},
        interconnectors=interconnectors,
        ev_fleets=ev_fleets,
        wind_assets=wind_assets,
        generator_assets=generator_assets,
        renewables_assets=renewables_assets,
        retailer_assets=retailer_assets,
        asset_provenance=_default_asset_provenance(),
        provenance=_default_provenance(),
    )


def _availability_share(network: pypsa.Network, component_id: str, *, default: float) -> float:
    if component_id in getattr(network.generators_t, "p_max_pu", pd.DataFrame()).columns:
        series = network.generators_t.p_max_pu[component_id]
        if not series.empty:
            return round(float(series.mean()), 6)
    return default


def _portfolio_availability_share(network: pypsa.Network, component_ids: pd.Index, *, default: float) -> float:
    p_max = getattr(network.generators_t, "p_max_pu", pd.DataFrame())
    weighted = []
    weights = []
    for component_id in component_ids:
        if str(component_id) not in p_max.columns:
            continue
        p_nom = float(network.generators.loc[component_id, "p_nom"])
        weighted.append(float(p_max[str(component_id)].mean()) * p_nom)
        weights.append(p_nom)
    return round(sum(weighted) / sum(weights), 6) if weights and sum(weights) > 0 else default


def _dispatch_rows(network: pypsa.Network, scenario: HeimdallScenario) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshots = network_snapshots_utc(network)
    for raw_snapshot, utc_snapshot in zip(network.snapshots, snapshots, strict=True):
        for zone in scenario.zones:
            generator_names = network.generators.index[
                network.generators["bus"].astype(str).str.startswith(zone)
            ]
            load_names = network.loads.index[
                network.loads["bus"].astype(str).str.startswith(zone)
                & network.loads.index.astype(str).str.contains("electric")
            ]
            p2h_name = f"{zone} P2H"
            generation_mw = (
                float(network.generators_t.p.loc[raw_snapshot, generator_names].sum())
                if not network.generators_t.p.empty
                else 0.0
            )
            load_mw = float(network.loads_t.p_set.loc[raw_snapshot, load_names].sum())
            p2h_mw = (
                float(max(0.0, network.links_t.p0.loc[raw_snapshot, p2h_name]))
                if p2h_name in network.links_t.p0.columns
                else 0.0
            )
            rows.append(
                {
                    "utc_timestamp": _iso_z(pd.Timestamp(utc_snapshot)),
                    "zone": zone,
                    "generation_mw": round(generation_mw, 6),
                    "load_mw": round(load_mw, 6),
                    "p2h_electric_mw": round(p2h_mw, 6),
                }
            )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_heimdall_scenario_bundle(
    network: pypsa.Network, output_dir: Path, *, source: str
) -> HeimdallScenarioBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = extract_heimdall_scenario(network)
    scenario_path = output_dir / "scenario.json"
    dispatch_path = output_dir / "dispatch.json"
    manifest_path = output_dir / "manifest.json"

    scenario_payload = scenario.to_dict()
    dispatch_rows = _dispatch_rows(network, scenario)
    dispatch_payload = {
        "schema_version": "1.0.0",
        "rows": dispatch_rows,
    }
    _write_json(scenario_path, scenario_payload)
    _write_json(dispatch_path, dispatch_payload)

    files = {
        "scenario.json": {
            "sha256": file_sha256(scenario_path),
            "schema_hash": _schema_hash(list(scenario_payload.keys())),
        },
        "dispatch.json": {
            "sha256": file_sha256(dispatch_path),
            "schema_hash": _schema_hash(
                list(dispatch_rows[0].keys()) if dispatch_rows else []
            ),
        },
    }
    manifest = {
        "schema_version": "1.0.0",
        "source": source,
        "scenario_id": scenario_payload["provenance"]["scenario_id"],
        "provenance": scenario_payload["provenance"],
        "zones": scenario.zones,
        "tick_count": len(scenario.snapshots_utc),
        "files": files,
    }
    _write_json(manifest_path, manifest)
    return HeimdallScenarioBundle(
        path=output_dir.resolve(),
        scenario_path=scenario_path.resolve(),
        dispatch_path=dispatch_path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def load_heimdall_scenario(path: Path) -> HeimdallScenario:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HeimdallScenario(
        zones=list(payload["zones"]),
        snapshots_utc=list(payload["snapshots_utc"]),
        p2h_assets={
            zone: P2HAsset(**asset)
            for zone, asset in payload["p2h_assets"].items()
        },
        thermal_storage={
            zone: ThermalStorage(**storage)
            for zone, storage in payload["thermal_storage"].items()
        },
        interconnectors={
            link_id: Interconnector(**link)
            for link_id, link in payload["interconnectors"].items()
        },
        ev_fleets={
            zone: EVFleetAsset(**asset)
            for zone, asset in payload.get("ev_fleets", {}).items()
        },
        wind_assets={
            zone: WindAsset(**asset)
            for zone, asset in payload.get("wind_assets", {}).items()
        },
        generator_assets={
            zone: GeneratorAsset(**asset)
            for zone, asset in payload.get("generator_assets", {}).items()
        },
        renewables_assets={
            zone: RenewablesAsset(**asset)
            for zone, asset in payload.get("renewables_assets", {}).items()
        },
        retailer_assets={
            zone: RetailerAsset(**asset)
            for zone, asset in payload.get("retailer_assets", {}).items()
        },
        asset_provenance=payload.get("asset_provenance", _default_asset_provenance()),
        provenance=payload.get("provenance", _default_provenance()),
    )
