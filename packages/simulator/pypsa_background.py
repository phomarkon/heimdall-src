from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from packages.pypsa_adapter import HeimdallScenario

from .action_assets import (
    RealAssetSpec,
    RealAssetState,
    initial_state_from_spec,
    simulate_real_asset_bid,
)
from .forecast import ForecastMarketState
from .models import Bid

ActionArchetype = Literal["p2h", "ev", "wind", "generator", "renewables", "retailer"]


@dataclass(frozen=True)
class BackgroundBidResult:
    accepted: bool
    simulator_kind: str
    archetype: ActionArchetype
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


def simulate_p2h_scenario_envelope_bid(
    *,
    scenario: HeimdallScenario,
    bid: Bid,
    forecast: ForecastMarketState,
    tau_eur: float = 0.0,
    current_power_mw: float = 8.0,
) -> BackgroundBidResult:
    p2h = scenario.p2h_assets[bid.zone]
    storage = scenario.thermal_storage[bid.zone]
    interval = forecast.interval_for_side(bid.side)
    expected, worst = _profit_pair(bid, forecast=forecast, interval=interval)
    current_mwh = max(0.0, current_power_mw * 0.25)
    ramp_mwh = max(0.0, p2h.ramp_limit_mw_per_tick * 0.25)
    if bid.side == "up":
        storage_support_mwh = max(0.0, storage.initial_soc_mwh / max(p2h.cop, 1e-9))
        physical_limit = min(current_mwh, ramp_mwh, storage_support_mwh)
        projected_power = max(0.0, current_power_mw - bid.quantity_mwh / 0.25)
        projected_soc = max(0.0, storage.initial_soc_mwh - bid.quantity_mwh * p2h.cop)
    else:
        headroom_mwh = max(0.0, p2h.p_nom_mw * 0.25 - current_mwh)
        storage_room_mwh = max(0.0, (storage.e_nom_mwh - storage.initial_soc_mwh) / max(p2h.cop, 1e-9))
        physical_limit = min(headroom_mwh, ramp_mwh, storage_room_mwh)
        projected_power = min(p2h.p_nom_mw, current_power_mw + bid.quantity_mwh / 0.25)
        projected_soc = min(storage.e_nom_mwh, storage.initial_soc_mwh + bid.quantity_mwh * p2h.cop)

    reasons: list[str] = []
    if bid.quantity_mwh > physical_limit + 1e-9:
        reasons.append("p2h_scenario_envelope_limit_exceeded")
    if worst + 1e-9 < tau_eur:
        reasons.append("conformal_profit_below_threshold")
    return _result(
        archetype="p2h",
        accepted=not reasons,
        reason_codes=reasons,
        expected_profit_eur=expected,
        worst_case_profit_eur=worst,
        forecast_interval_eur_mwh=interval,
        physical_limit_mwh=physical_limit,
        projected_state={
            "state_asset_id": p2h.asset_id,
            "current_power_mw": current_power_mw,
            "projected_power_mw": projected_power,
            "projected_thermal_soc_mwh": projected_soc,
            "ramp_limit_mwh": ramp_mwh,
            "cop": p2h.cop,
        },
        failed_stage="physical" if any(reason.startswith("p2h_") for reason in reasons) else "conformal" if reasons else None,
        scenario_asset_id=p2h.asset_id,
        provenance=_provenance(scenario, bid.zone, component_id=f"{bid.zone} P2H"),
        backend="scenario_envelope",
        next_state=None,
    )


def simulate_pypsa_background_asset_bid(
    *,
    scenario: HeimdallScenario,
    spec: RealAssetSpec,
    bid: Bid,
    forecast: ForecastMarketState,
    state: RealAssetState | None = None,
    tau_eur: float = 0.0,
) -> BackgroundBidResult:
    state = state or initial_state_from_spec(spec)
    envelope = simulate_real_asset_bid(
        spec=spec,
        bid=bid,
        forecast=forecast,
        state=state,
        tau_eur=tau_eur,
    )
    reasons = list(envelope.reason_codes)
    network_limit = _network_exchange_limit_mwh(scenario, bid.zone)
    if bid.quantity_mwh > network_limit + 1e-9:
        reasons.append("pypsa_background_network_limit_exceeded")
    accepted = not reasons
    failed_stage = (
        "network" if "pypsa_background_network_limit_exceeded" in reasons
        else envelope.failed_stage if reasons
        else None
    )
    return _result(
        archetype=spec.archetype,
        accepted=accepted,
        reason_codes=reasons,
        expected_profit_eur=envelope.expected_profit_eur,
        worst_case_profit_eur=envelope.worst_case_profit_eur,
        forecast_interval_eur_mwh=envelope.forecast_interval_eur_mwh,
        physical_limit_mwh=min(envelope.physical_limit_mwh, network_limit),
        projected_state={
            **envelope.projected_state,
            "network_exchange_limit_mwh": network_limit,
        },
        failed_stage=failed_stage,
        scenario_asset_id=envelope.scenario_asset_id,
        provenance={**envelope.provenance, **_provenance(scenario, bid.zone)},
        backend="pypsa_background",
        next_state=envelope.next_state if accepted else None,
    )


def pypsa_background_from_p2h_mfrr(
    *,
    scenario: HeimdallScenario,
    bid: Bid,
    forecast: ForecastMarketState,
    accepted: bool,
    reason_codes: list[str],
    expected_profit_eur: float,
    worst_case_profit_eur: float | None,
    forecast_interval_eur_mwh: tuple[float, float] | None,
    result_hash: str,
) -> dict[str, object]:
    envelope = simulate_p2h_scenario_envelope_bid(
        scenario=scenario,
        bid=bid,
        forecast=forecast,
        tau_eur=-10**12,
    )
    reasons = list(dict.fromkeys(list(envelope.reason_codes) + list(reason_codes)))
    controls_acceptance = accepted and envelope.accepted
    payload = {
        "ok": True,
        "simulator_kind": "p2h_pypsa_background",
        "archetype": "p2h",
        "backend": "pypsa_background",
        "authority": "authoritative",
        "accepted": controls_acceptance,
        "reason_codes": reasons,
        "expected_profit_eur": round(expected_profit_eur, 6),
        "worst_case_profit_eur": round(worst_case_profit_eur or 0.0, 6),
        "forecast_interval_eur_mwh": list(forecast_interval_eur_mwh or envelope.forecast_interval_eur_mwh),
        "physical_limit_mwh": envelope.physical_limit_mwh,
        "projected_state": envelope.projected_state,
        "failed_stage": None if controls_acceptance else envelope.failed_stage or "pypsa_background",
        "scenario_asset_id": envelope.scenario_asset_id,
        "provenance": envelope.provenance,
        "legacy_result_hash": result_hash,
    }
    payload["result_hash"] = _hash_payload(payload)
    return payload


def _result(
    *,
    archetype: ActionArchetype,
    accepted: bool,
    reason_codes: list[str],
    expected_profit_eur: float,
    worst_case_profit_eur: float,
    forecast_interval_eur_mwh: tuple[float, float],
    physical_limit_mwh: float,
    projected_state: dict[str, float | str],
    failed_stage: str | None,
    scenario_asset_id: str,
    provenance: dict[str, str | float | bool],
    backend: str,
    next_state: RealAssetState | None,
) -> BackgroundBidResult:
    result = BackgroundBidResult(
        accepted=accepted,
        simulator_kind=f"{archetype}_{backend}",
        archetype=archetype,
        backend=backend,
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
        scenario_asset_id=scenario_asset_id,
        provenance=provenance,
        next_state=next_state,
        result_hash="",
    )
    return replace(result, result_hash=_hash_payload(asdict(result)))


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


def _network_exchange_limit_mwh(scenario: HeimdallScenario, zone: str) -> float:
    capacities = [
        link.capacity_mw
        for link in scenario.interconnectors.values()
        if zone in (link.from_zone, link.to_zone)
    ]
    if not capacities:
        return 0.0
    return max(0.0, max(capacities) * 0.25)


def _provenance(
    scenario: HeimdallScenario,
    zone: str,
    *,
    component_id: str | None = None,
) -> dict[str, str | float | bool]:
    payload: dict[str, str | float | bool] = {
        "source": "HeimdallScenario",
        "scenario_id": str((scenario.provenance or {}).get("scenario_id", "unknown")),
        "physical_source": str((scenario.provenance or {}).get("physical_source", "compact PyPSA-derived DK1/DK2 scenario")),
        "zone": zone,
        "full_pypsa_eur_required": bool((scenario.provenance or {}).get("full_pypsa_eur_required", False)),
    }
    if component_id is not None:
        payload["component_id"] = component_id
    return payload


def _hash_payload(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
