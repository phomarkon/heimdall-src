from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .models import Bid

EVFailureStage = Literal["capacity", "soc", "availability"]


@dataclass(frozen=True)
class EVFleetState:
    asset_id: str
    capacity_mw: float
    energy_mwh: float
    soc_mwh: float
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    availability_share: float = 0.75


@dataclass(frozen=True)
class EVBidSimulationResult:
    accepted: bool
    simulator_kind: str
    archetype: str
    authority: str
    reason_codes: list[str]
    projected_soc_mwh: float
    remaining_charge_mwh: float
    remaining_discharge_mwh: float
    failed_stage: EVFailureStage | None
    result_hash: str


class EVVirtualBatterySimulator:
    def __init__(self, state: EVFleetState) -> None:
        self._state = state

    def simulate_bid(self, bid: Bid) -> EVBidSimulationResult:
        available_power_mw = self._state.capacity_mw * self._state.availability_share
        available_quantity_mwh = available_power_mw * 0.25
        if bid.quantity_mwh - available_quantity_mwh > 1e-9:
            return _result(
                accepted=False,
                reason_codes=["ev_capacity_exceeded"],
                projected_soc_mwh=self._state.soc_mwh,
                remaining_charge_mwh=_remaining_charge(self._state),
                remaining_discharge_mwh=_remaining_discharge(self._state),
                failed_stage="capacity",
            )

        if bid.side == "down":
            projected_soc = self._state.soc_mwh + bid.quantity_mwh * self._state.charge_efficiency
            if projected_soc - self._state.energy_mwh > 1e-9:
                return _result(
                    accepted=False,
                    reason_codes=["ev_soc_upper_exceeded"],
                    projected_soc_mwh=projected_soc,
                    remaining_charge_mwh=_remaining_charge(self._state),
                    remaining_discharge_mwh=_remaining_discharge(self._state),
                    failed_stage="soc",
                )
        else:
            projected_soc = self._state.soc_mwh - bid.quantity_mwh / max(self._state.discharge_efficiency, 1e-9)
            if projected_soc < -1e-9:
                return _result(
                    accepted=False,
                    reason_codes=["ev_soc_lower_exceeded"],
                    projected_soc_mwh=projected_soc,
                    remaining_charge_mwh=_remaining_charge(self._state),
                    remaining_discharge_mwh=_remaining_discharge(self._state),
                    failed_stage="soc",
                )

        return _result(
            accepted=True,
            reason_codes=[],
            projected_soc_mwh=projected_soc,
            remaining_charge_mwh=max(0.0, self._state.energy_mwh - projected_soc),
            remaining_discharge_mwh=max(0.0, projected_soc),
            failed_stage=None,
        )


def _remaining_charge(state: EVFleetState) -> float:
    return max(0.0, state.energy_mwh - state.soc_mwh)


def _remaining_discharge(state: EVFleetState) -> float:
    return max(0.0, state.soc_mwh)


def _result(
    *,
    accepted: bool,
    reason_codes: list[str],
    projected_soc_mwh: float,
    remaining_charge_mwh: float,
    remaining_discharge_mwh: float,
    failed_stage: EVFailureStage | None,
) -> EVBidSimulationResult:
    result = EVBidSimulationResult(
        accepted=accepted,
        simulator_kind="ev_virtual_battery",
        archetype="ev",
        authority="authoritative",
        reason_codes=reason_codes,
        projected_soc_mwh=round(projected_soc_mwh, 6),
        remaining_charge_mwh=round(remaining_charge_mwh, 6),
        remaining_discharge_mwh=round(remaining_discharge_mwh, 6),
        failed_stage=failed_stage,
        result_hash="",
    )
    return replace(result, result_hash=_hash_payload(asdict(result)))


def _hash_payload(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
