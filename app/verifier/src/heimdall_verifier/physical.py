"""Physical-feasibility checks (Stage 1). Per docs/RESEARCH-PROPOSAL.md §4.5.

Encodes the deterministic constraints listed in §4.5 for a P2H +
thermal-storage operator (Danfoss-style focal agent):

- position envelope: |position + delta| <= Q_max
- ramp limit:        |delta - delta_prev| <= R * dt_h
- SoC bounds:        0 <= SoC' <= S_max  with  SoC' = SoC*(1-loss) + COP*power*dt_h
- gate-closure:      delivery_quarter must be after gate-closure cutoff
- bid-tick size:     price aligned to bid-tick (default 0.01 EUR/MWh)

A violation returns the first binding constraint with a structured suggestion
so the focal agent can retry. The function is pure / deterministic; no I/O,
no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from heimdall_contracts import BidAction, PhysicalViolation


@dataclass(frozen=True)
class AssetSpec:
    """Physical envelope of the focal asset. Sourced from PyPSA-Eur-Sec (§4.8)."""

    q_max_mw: float
    ramp_mw_per_min: float
    storage_mwh: float
    cop: float
    loss_per_quarter: float = 0.005  # 0.5% per 15 min thermal-storage standing loss
    bid_tick_eur: float = 0.01


@dataclass(frozen=True)
class AssetState:
    """Live state of the asset at decision time."""

    position_mw: float
    last_delta_mw: float
    soc_mwh: float
    cash_eur: float
    now_utc: datetime
    gate_closure_utc: datetime


def _quantity_signed(bid: BidAction) -> float:
    return -bid.quantity_mw if bid.direction == "buy" else bid.quantity_mw


def physical_check(
    bid: BidAction, spec: AssetSpec, state: AssetState
) -> PhysicalViolation | None:
    """Return the first binding physical constraint, or None if feasible."""

    # Gate closure first — cheapest, most categorical.
    if bid.delivery_quarter <= state.gate_closure_utc:
        return PhysicalViolation(
            constraint="gate_closure",
            current_value=(bid.delivery_quarter - state.now_utc).total_seconds(),
            bound_value=(state.gate_closure_utc - state.now_utc).total_seconds(),
            suggestion="bid for a later delivery quarter; gate-closure has passed",
        )

    # Bid-tick size.
    ticks = round(bid.price_eur_per_mwh / spec.bid_tick_eur)
    if abs(bid.price_eur_per_mwh - ticks * spec.bid_tick_eur) > 1e-9:
        return PhysicalViolation(
            constraint="bid_tick_size",
            current_value=bid.price_eur_per_mwh,
            bound_value=ticks * spec.bid_tick_eur,
            suggestion=f"round price to the nearest {spec.bid_tick_eur} EUR/MWh tick",
        )

    delta = _quantity_signed(bid)
    new_position = state.position_mw + delta
    if abs(new_position) > spec.q_max_mw:
        return PhysicalViolation(
            constraint="position_envelope",
            current_value=abs(new_position),
            bound_value=spec.q_max_mw,
            suggestion=(
                f"reduce quantity by >= {abs(new_position) - spec.q_max_mw:.3f} MW"
            ),
        )

    # Ramp: change in *delta* between consecutive quarters, bounded by ramp_rate * dt.
    dt_min = bid.duration_minutes
    max_ramp = spec.ramp_mw_per_min * dt_min
    if abs(delta - state.last_delta_mw) > max_ramp:
        return PhysicalViolation(
            constraint="ramp_limit",
            current_value=abs(delta - state.last_delta_mw),
            bound_value=max_ramp,
            suggestion=(
                f"reduce quantity by >= "
                f"{abs(delta - state.last_delta_mw) - max_ramp:.3f} MW to respect ramp"
            ),
        )

    # Storage SoC update for a P2H operator: a 'buy' action consumes
    # electricity to charge thermal storage; a 'sell' draws stored heat back.
    dt_h = dt_min / 60.0
    soc_after = state.soc_mwh * (1.0 - spec.loss_per_quarter) - delta * spec.cop * dt_h
    if soc_after < 0.0:
        return PhysicalViolation(
            constraint="soc_floor",
            current_value=soc_after,
            bound_value=0.0,
            suggestion="reduce sell volume; storage would discharge below empty",
        )
    if soc_after > spec.storage_mwh:
        return PhysicalViolation(
            constraint="soc_ceiling",
            current_value=soc_after,
            bound_value=spec.storage_mwh,
            suggestion="reduce buy volume; storage would charge above capacity",
        )

    # Cash floor: a buy at the bid price within this quarter must be coverable
    # at worst case (= the bid price itself, since we paid in).
    if bid.direction == "buy":
        max_outlay = bid.quantity_mw * bid.price_eur_per_mwh * dt_h
        if state.cash_eur < max_outlay:
            return PhysicalViolation(
                constraint="cash_floor",
                current_value=state.cash_eur,
                bound_value=max_outlay,
                suggestion=f"reduce volume; current cash ({state.cash_eur:.0f}) "
                f"cannot cover worst-case outlay {max_outlay:.0f} EUR",
            )

    return None


def default_p2h_spec() -> AssetSpec:
    """A canonical P2H focal-asset spec used by smoke tests and tutorials."""
    return AssetSpec(
        q_max_mw=50.0,
        ramp_mw_per_min=5.0,
        storage_mwh=100.0,
        cop=3.0,
    )


def default_p2h_state(now: datetime | None = None) -> AssetState:
    if now is None:
        now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    return AssetState(
        position_mw=0.0,
        last_delta_mw=0.0,
        soc_mwh=50.0,
        cash_eur=1_000_000.0,
        now_utc=now,
        gate_closure_utc=now - timedelta(minutes=5),  # gate already closed for past quarters
    )
