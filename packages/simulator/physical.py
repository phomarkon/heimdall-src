from __future__ import annotations

from packages.pypsa_adapter import HeimdallScenario

from .models import Bid, ConstraintDecision, SimulatorAssetState


class PhysicalConstraintProvider:
    def __init__(self, scenario: HeimdallScenario) -> None:
        self._scenario = scenario

    def validate_bid(
        self, bid: Bid, state: SimulatorAssetState
    ) -> ConstraintDecision:
        asset = self._scenario.p2h_assets[bid.asset_id]
        storage = self._scenario.thermal_storage[bid.asset_id]
        if bid.zone != asset.zone:
            return ConstraintDecision(False, "zone_mismatch", state.electric_power_mw, 0.0, 0.0)

        current_soc = (
            storage.initial_soc_mwh
            if state.thermal_soc_mwh is None
            else state.thermal_soc_mwh
        )
        power_delta_mw = bid.quantity_mwh / 0.25
        signed_delta = power_delta_mw if bid.side == "down" else -power_delta_mw
        projected_power = state.electric_power_mw + signed_delta

        if projected_power < -1e-9 or projected_power - asset.p_nom_mw > 1e-9:
            return ConstraintDecision(
                False,
                "capacity_exceeded",
                projected_power,
                current_soc,
                max(0.0, asset.p_nom_mw - state.electric_power_mw),
            )

        if abs(signed_delta) - asset.ramp_limit_mw_per_tick > 1e-9:
            return ConstraintDecision(
                False,
                "ramp_exceeded",
                projected_power,
                current_soc,
                max(0.0, asset.p_nom_mw - state.electric_power_mw),
            )

        projected_soc = current_soc * (1.0 - storage.thermal_loss_per_tick)
        if bid.side == "down":
            projected_soc += bid.quantity_mwh * asset.cop

        if projected_soc - storage.e_nom_mwh > 1e-9:
            return ConstraintDecision(
                False,
                "storage_exceeded",
                projected_power,
                projected_soc,
                max(0.0, asset.p_nom_mw - state.electric_power_mw),
            )

        return ConstraintDecision(
            True,
            None,
            projected_power,
            projected_soc,
            max(0.0, asset.p_nom_mw - projected_power),
        )
