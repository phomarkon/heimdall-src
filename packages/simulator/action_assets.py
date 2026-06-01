from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from packages.pypsa_adapter import HeimdallScenario

from .ev import EVFleetState, EVVirtualBatterySimulator
from .forecast import ForecastMarketState
from .models import Bid

AssetArchetype = Literal["ev", "wind", "generator", "renewables", "retailer"]
SCENARIO_ENVELOPE_BACKEND = "scenario_envelope"


@dataclass(frozen=True)
class RealAssetBidResult:
    accepted: bool
    simulator_kind: str
    archetype: AssetArchetype
    backend: str
    authority: str
    reason_codes: list[str]
    expected_profit_eur: float
    worst_case_profit_eur: float
    forecast_interval_eur_mwh: tuple[float, float]
    physical_limit_mwh: float
    projected_state: dict[str, float | str]
    failed_stage: str | None
    scenario_asset_id: str
    provenance: dict[str, str | float | bool]
    next_state: RealAssetState | None
    result_hash: str


@dataclass(frozen=True)
class RealAssetSpec:
    archetype: AssetArchetype
    capacity_mw: float
    storage_mwh: float | None = None
    availability_share: float = 1.0
    current_dispatch_share: float = 0.5
    ramp_share_per_tick: float = 0.35
    marginal_cost_eur_mwh: float = 0.0
    scenario_asset_id: str | None = None
    down_regulation_supported: bool = False
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    rebound_fraction: float = 0.0
    max_event_duration_ticks: int = 1
    provenance: dict[str, str | float | bool] | None = None


@dataclass(frozen=True)
class RealAssetState:
    archetype: AssetArchetype
    asset_id: str
    soc_mwh: float | None = None
    dispatch_mw: float | None = None
    rebound_obligation_mwh: float = 0.0
    consecutive_event_ticks: int = 0


@dataclass
class ScenarioAssetStateStore:
    states: dict[tuple[str, AssetArchetype, str], RealAssetState]

    @classmethod
    def empty(cls) -> ScenarioAssetStateStore:
        return cls(states={})

    def get(self, *, agent_id: str, spec: RealAssetSpec) -> RealAssetState:
        key = (agent_id, spec.archetype, spec.scenario_asset_id or spec.archetype)
        if key not in self.states:
            self.states[key] = initial_state_from_spec(spec)
        return self.states[key]

    def commit(self, *, agent_id: str, spec: RealAssetSpec, state: RealAssetState) -> None:
        key = (agent_id, spec.archetype, spec.scenario_asset_id or spec.archetype)
        self.states[key] = state


def initial_state_from_spec(spec: RealAssetSpec) -> RealAssetState:
    if spec.archetype == "ev":
        return RealAssetState(
            archetype="ev",
            asset_id=spec.scenario_asset_id or "ev",
            soc_mwh=float(spec.storage_mwh or max(spec.capacity_mw, 1.0)) * 0.5,
        )
    if spec.archetype == "generator":
        return RealAssetState(
            archetype="generator",
            asset_id=spec.scenario_asset_id or "generator",
            dispatch_mw=spec.capacity_mw * spec.current_dispatch_share,
        )
    if spec.archetype == "retailer":
        return RealAssetState(archetype="retailer", asset_id=spec.scenario_asset_id or "retailer")
    return RealAssetState(archetype=spec.archetype, asset_id=spec.scenario_asset_id or spec.archetype)


def _extract_ev(asset, provenance: dict) -> dict:
    return dict(
        capacity_mw=asset.capacity_mw,
        storage_mwh=asset.energy_mwh,
        availability_share=asset.availability_share,
        charge_efficiency=asset.charge_efficiency,
        discharge_efficiency=asset.discharge_efficiency,
        scenario_asset_id=asset.asset_id,
        provenance=provenance,
    )


def _extract_wind(asset, provenance: dict) -> dict:
    return dict(
        capacity_mw=asset.p_nom_mw,
        availability_share=asset.availability_share,
        ramp_share_per_tick=asset.ramp_limit_mw_per_tick / max(asset.p_nom_mw, 1e-9),
        marginal_cost_eur_mwh=asset.marginal_cost_eur_mwh,
        down_regulation_supported=asset.down_regulation_supported,
        scenario_asset_id=asset.asset_id,
        provenance={**provenance, "component_id": asset.generator_id},
    )


def _extract_generator(asset, provenance: dict) -> dict:
    return dict(
        capacity_mw=asset.p_nom_mw,
        availability_share=asset.availability_share,
        current_dispatch_share=asset.initial_dispatch_mw / max(asset.p_nom_mw, 1e-9),
        ramp_share_per_tick=asset.ramp_limit_mw_per_tick / max(asset.p_nom_mw, 1e-9),
        marginal_cost_eur_mwh=asset.marginal_cost_eur_mwh,
        scenario_asset_id=asset.asset_id,
        provenance={**provenance, "component_id": asset.generator_id},
    )


def _extract_renewables(asset, provenance: dict) -> dict:
    return dict(
        capacity_mw=asset.p_nom_mw,
        availability_share=asset.availability_share,
        ramp_share_per_tick=asset.ramp_limit_mw_per_tick / max(asset.p_nom_mw, 1e-9),
        marginal_cost_eur_mwh=asset.marginal_cost_eur_mwh,
        down_regulation_supported=asset.down_regulation_supported,
        scenario_asset_id=asset.asset_id,
        provenance={**provenance, "wind_share": asset.wind_share, "solar_share": asset.solar_share},
    )


def _extract_retailer(asset, provenance: dict) -> dict:
    return dict(
        capacity_mw=asset.flexible_load_mw,
        availability_share=asset.availability_share,
        rebound_fraction=asset.rebound_fraction,
        max_event_duration_ticks=asset.max_event_duration_ticks,
        scenario_asset_id=asset.asset_id,
        provenance=provenance,
    )


_SCENARIO_EXTRACTORS: dict[AssetArchetype, tuple[str, callable]] = {
    "ev": ("ev_fleets", _extract_ev),
    "wind": ("wind_assets", _extract_wind),
    "generator": ("generator_assets", _extract_generator),
    "renewables": ("renewables_assets", _extract_renewables),
    "retailer": ("retailer_assets", _extract_retailer),
}


def spec_from_scenario(scenario: HeimdallScenario, *, archetype: AssetArchetype, zone: str) -> RealAssetSpec:
    if archetype not in _SCENARIO_EXTRACTORS:
        raise ValueError(f"unsupported scenario asset archetype: {archetype}")
    field_name, extract = _SCENARIO_EXTRACTORS[archetype]
    asset = (getattr(scenario, field_name) or {})[zone]
    provenance = {
        "source": "HeimdallScenario",
        "scenario_id": str((scenario.provenance or {}).get("scenario_id", "unknown")),
        "zone": zone,
    }
    return RealAssetSpec(archetype=archetype, **extract(asset, provenance))


def simulate_real_asset_bid(
    *,
    spec: RealAssetSpec,
    bid: Bid,
    forecast: ForecastMarketState,
    state: RealAssetState | None = None,
    tau_eur: float = 0.0,
) -> RealAssetBidResult:
    state = state or initial_state_from_spec(spec)
    if spec.archetype == "ev":
        return _simulate_ev(spec=spec, state=state, bid=bid, forecast=forecast, tau_eur=tau_eur)
    if spec.archetype == "wind":
        return _simulate_wind_like(
            spec=replace(spec, availability_share=min(spec.availability_share, 0.45)),
            state=state,
            bid=bid,
            forecast=forecast,
            tau_eur=tau_eur,
            down_allowed=False,
        )
    if spec.archetype == "renewables":
        return _simulate_wind_like(
            spec=replace(spec, availability_share=min(spec.availability_share, 0.55)),
            state=state,
            bid=bid,
            forecast=forecast,
            tau_eur=tau_eur,
            down_allowed=False,
        )
    if spec.archetype == "generator":
        return _simulate_generator(spec=spec, state=state, bid=bid, forecast=forecast, tau_eur=tau_eur)
    if spec.archetype == "retailer":
        return _simulate_retailer(spec=spec, state=state, bid=bid, forecast=forecast, tau_eur=tau_eur)
    raise ValueError(f"unsupported real asset archetype: {spec.archetype}")


def _simulate_ev(
    *,
    spec: RealAssetSpec,
    state: RealAssetState,
    bid: Bid,
    forecast: ForecastMarketState,
    tau_eur: float,
) -> RealAssetBidResult:
    energy = float(spec.storage_mwh or max(spec.capacity_mw, 1.0))
    state = EVFleetState(
        asset_id=bid.asset_id,
        capacity_mw=spec.capacity_mw,
        energy_mwh=energy,
        soc_mwh=state.soc_mwh if state.soc_mwh is not None else energy * 0.5,
        charge_efficiency=spec.charge_efficiency,
        discharge_efficiency=spec.discharge_efficiency,
        availability_share=min(max(spec.availability_share, 0.0), 1.0),
    )
    physical = EVVirtualBatterySimulator(state).simulate_bid(bid)
    interval = forecast.interval_for_side(bid.side)
    expected, worst = _profit_pair(bid, forecast=forecast, interval=interval)
    reasons = list(physical.reason_codes)
    failed_stage = physical.failed_stage
    if physical.accepted and worst + 1e-9 < tau_eur:
        reasons.append("conformal_profit_below_threshold")
        failed_stage = "conformal"
    accepted = physical.accepted and not reasons
    return _result(
        archetype="ev",
        accepted=accepted,
        reason_codes=reasons,
        expected_profit_eur=expected,
        worst_case_profit_eur=worst,
        forecast_interval_eur_mwh=interval,
        physical_limit_mwh=state.capacity_mw * state.availability_share * 0.25,
        projected_state={
            "state_asset_id": state.asset_id,
            "soc_mwh": physical.projected_soc_mwh,
            "remaining_charge_mwh": physical.remaining_charge_mwh,
            "remaining_discharge_mwh": physical.remaining_discharge_mwh,
            "availability_share": state.availability_share,
        },
        failed_stage=failed_stage,
        next_state=RealAssetState(
            archetype="ev",
            asset_id=state.asset_id,
            soc_mwh=physical.projected_soc_mwh,
        ) if accepted else None,
        spec=spec,
    )


def _simulate_wind_like(
    *,
    spec: RealAssetSpec,
    state: RealAssetState,
    bid: Bid,
    forecast: ForecastMarketState,
    tau_eur: float,
    down_allowed: bool,
) -> RealAssetBidResult:
    interval = forecast.interval_for_side(bid.side)
    expected, worst = _profit_pair(bid, forecast=forecast, interval=interval)
    available_mwh = max(0.0, spec.capacity_mw * spec.availability_share * 0.25)
    reasons: list[str] = []
    if bid.side == "down" and not (down_allowed or spec.down_regulation_supported):
        reasons.append(f"{spec.archetype}_down_bid_not_physically_supported")
    if bid.quantity_mwh > available_mwh + 1e-9:
        reasons.append(f"{spec.archetype}_availability_exceeded")
    if worst + 1e-9 < tau_eur:
        reasons.append("conformal_profit_below_threshold")
    failed = "physical" if any(reason.endswith(("supported_v1", "exceeded")) for reason in reasons) else "conformal" if reasons else None
    return _result(
        archetype=spec.archetype,
        accepted=not reasons,
        reason_codes=reasons,
        expected_profit_eur=expected,
        worst_case_profit_eur=worst,
        forecast_interval_eur_mwh=interval,
        physical_limit_mwh=available_mwh,
        projected_state={
            "available_generation_mwh": available_mwh,
            "state_asset_id": state.asset_id,
            "availability_share": spec.availability_share,
            "curtailed_or_withheld_mwh": bid.quantity_mwh if bid.side == "up" and not reasons else 0.0,
        },
        failed_stage=failed,
        next_state=state if not reasons else None,
        spec=spec,
    )


def _simulate_generator(
    *,
    spec: RealAssetSpec,
    state: RealAssetState,
    bid: Bid,
    forecast: ForecastMarketState,
    tau_eur: float,
) -> RealAssetBidResult:
    interval = forecast.interval_for_side(bid.side)
    expected, worst = _profit_pair(bid, forecast=forecast, interval=interval)
    current_mw = state.dispatch_mw if state.dispatch_mw is not None else spec.capacity_mw * spec.current_dispatch_share
    current_mwh = max(0.0, current_mw * 0.25)
    upward_headroom_mwh = max(0.0, spec.capacity_mw * 0.25 - current_mwh)
    downward_headroom_mwh = current_mwh
    ramp_mwh = max(0.0, spec.capacity_mw * spec.ramp_share_per_tick * 0.25)
    physical_limit = min(ramp_mwh, upward_headroom_mwh if bid.side == "up" else downward_headroom_mwh)
    net_expected = expected - bid.quantity_mwh * spec.marginal_cost_eur_mwh if bid.side == "up" else expected
    net_worst = worst - bid.quantity_mwh * spec.marginal_cost_eur_mwh if bid.side == "up" else worst
    reasons: list[str] = []
    if bid.quantity_mwh > physical_limit + 1e-9:
        reasons.append("generator_ramp_or_headroom_exceeded")
    if net_worst + 1e-9 < tau_eur:
        reasons.append("conformal_profit_below_threshold")
    return _result(
        archetype="generator",
        accepted=not reasons,
        reason_codes=reasons,
        expected_profit_eur=net_expected,
        worst_case_profit_eur=net_worst,
        forecast_interval_eur_mwh=interval,
        physical_limit_mwh=physical_limit,
        projected_state={
            "state_asset_id": state.asset_id,
            "current_dispatch_mwh": current_mwh,
            "projected_dispatch_mw": max(0.0, min(spec.capacity_mw, current_mw + (bid.quantity_mwh / 0.25 if bid.side == "up" else -bid.quantity_mwh / 0.25))),
            "ramp_limit_mwh": ramp_mwh,
            "marginal_cost_eur_mwh": spec.marginal_cost_eur_mwh,
        },
        failed_stage="physical" if "generator_ramp_or_headroom_exceeded" in reasons else "conformal" if reasons else None,
        next_state=RealAssetState(
            archetype="generator",
            asset_id=state.asset_id,
            dispatch_mw=max(0.0, min(spec.capacity_mw, current_mw + (bid.quantity_mwh / 0.25 if bid.side == "up" else -bid.quantity_mwh / 0.25))),
        ) if not reasons else None,
        spec=spec,
    )


def _simulate_retailer(
    *,
    spec: RealAssetSpec,
    state: RealAssetState,
    bid: Bid,
    forecast: ForecastMarketState,
    tau_eur: float,
) -> RealAssetBidResult:
    interval = forecast.interval_for_side(bid.side)
    expected, worst = _profit_pair(bid, forecast=forecast, interval=interval)
    flexible_mwh = max(0.0, spec.capacity_mw * spec.availability_share * 0.25)
    rebound_penalty = spec.rebound_fraction * abs(expected)
    net_expected = expected - rebound_penalty
    net_worst = worst - rebound_penalty
    reasons: list[str] = []
    if bid.quantity_mwh > flexible_mwh + 1e-9:
        reasons.append("retailer_flexible_load_exceeded")
    if state.consecutive_event_ticks >= spec.max_event_duration_ticks:
        reasons.append("retailer_event_duration_exceeded")
    if net_worst + 1e-9 < tau_eur:
        reasons.append("conformal_profit_below_threshold")
    return _result(
        archetype="retailer",
        accepted=not reasons,
        reason_codes=reasons,
        expected_profit_eur=net_expected,
        worst_case_profit_eur=net_worst,
        forecast_interval_eur_mwh=interval,
        physical_limit_mwh=flexible_mwh,
        projected_state={
            "state_asset_id": state.asset_id,
            "flexible_load_mwh": flexible_mwh,
            "rebound_penalty_eur": rebound_penalty,
            "rebound_obligation_mwh": state.rebound_obligation_mwh + (bid.quantity_mwh * spec.rebound_fraction if not reasons else 0.0),
            "event_duration_ticks": state.consecutive_event_ticks + (1 if not reasons else 0),
        },
        failed_stage="physical" if any(reason.startswith("retailer_") for reason in reasons) else "conformal" if reasons else None,
        next_state=RealAssetState(
            archetype="retailer",
            asset_id=state.asset_id,
            rebound_obligation_mwh=state.rebound_obligation_mwh + bid.quantity_mwh * spec.rebound_fraction,
            consecutive_event_ticks=state.consecutive_event_ticks + 1,
        ) if not reasons else None,
        spec=spec,
    )


def _profit_pair(
    bid: Bid,
    *,
    forecast: ForecastMarketState,
    interval: tuple[float, float],
) -> tuple[float, float]:
    lower, upper = interval
    median = (lower + upper) / 2.0
    if bid.side == "up":
        return (
            round(bid.quantity_mwh * (median - forecast.spot_price_eur_mwh), 6),
            round(bid.quantity_mwh * (lower - forecast.spot_price_eur_mwh), 6),
        )
    return (
        round(bid.quantity_mwh * (forecast.spot_price_eur_mwh - median), 6),
        round(bid.quantity_mwh * (forecast.spot_price_eur_mwh - upper), 6),
    )


def _result(
    *,
    archetype: AssetArchetype,
    accepted: bool,
    reason_codes: list[str],
    expected_profit_eur: float,
    worst_case_profit_eur: float,
    forecast_interval_eur_mwh: tuple[float, float],
    physical_limit_mwh: float,
    projected_state: dict[str, float | str],
    failed_stage: str | None,
    next_state: RealAssetState | None,
    spec: RealAssetSpec,
) -> RealAssetBidResult:
    result = RealAssetBidResult(
        accepted=accepted,
        simulator_kind=f"{archetype}_{SCENARIO_ENVELOPE_BACKEND}",
        archetype=archetype,
        backend=SCENARIO_ENVELOPE_BACKEND,
        authority="authoritative",
        reason_codes=reason_codes,
        expected_profit_eur=round(expected_profit_eur, 6),
        worst_case_profit_eur=round(worst_case_profit_eur, 6),
        forecast_interval_eur_mwh=(
            round(forecast_interval_eur_mwh[0], 6),
            round(forecast_interval_eur_mwh[1], 6),
        ),
        physical_limit_mwh=round(physical_limit_mwh, 6),
        projected_state={key: round(value, 6) if isinstance(value, float) else value for key, value in projected_state.items()},
        failed_stage=failed_stage,
        scenario_asset_id=spec.scenario_asset_id or str(projected_state.get("state_asset_id") or archetype),
        provenance=spec.provenance or {},
        next_state=next_state,
        result_hash="",
    )
    return replace(result, result_hash=_hash_payload(asdict(result)))


def _hash_payload(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
