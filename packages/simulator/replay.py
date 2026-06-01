from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Protocol

from packages.pypsa_adapter import HeimdallScenario

from .models import (
    AcceptedBid,
    Bid,
    MarketState,
    RejectedBid,
    SimulationResult,
    SimulatorAssetState,
)
from .market import MFRRBidBook, MFRRMarketClock
from .physical import PhysicalConstraintProvider


class ReplayPolicy(Protocol):
    def bids_for_state(self, state: MarketState) -> list[Bid]: ...


class ConstantBidPolicy:
    def __init__(self, bids: list[Bid]) -> None:
        self._bids = bids
        self._used = False

    def bids_for_state(self, state: MarketState) -> list[Bid]:
        if self._used:
            return []
        self._used = True
        return self._bids

    def reset(self) -> None:
        self._used = False


class ReplaySimulator:
    def __init__(
        self,
        fixture: dict,
        scenario: HeimdallScenario,
        *,
        clock: MFRRMarketClock | None = None,
    ) -> None:
        self._fixture = fixture
        self._scenario = scenario
        self._constraint_provider = PhysicalConstraintProvider(scenario)
        self._clock = clock or MFRRMarketClock()
        self._states = {
            zone: SimulatorAssetState.for_asset(
                zone,
                thermal_soc_mwh=scenario.thermal_storage[zone].initial_soc_mwh,
            )
            for zone in scenario.zones
        }

    @classmethod
    def from_files(
        cls,
        fixture_path: Path,
        scenario: HeimdallScenario,
        *,
        clock: MFRRMarketClock | None = None,
    ) -> "ReplaySimulator":
        return cls(
            json.loads(fixture_path.read_text(encoding="utf-8")),
            scenario,
            clock=clock,
        )

    def run(self, policy: ReplayPolicy) -> SimulationResult:
        if hasattr(policy, "reset"):
            policy.reset()
        accepted: list[AcceptedBid] = []
        rejected: list[RejectedBid] = []
        for tick in self._fixture["ticks"]:
            timestamp = _parse_utc(tick["utc_timestamp"])
            markets = {
                row["zone"]: {
                    key: float(value)
                    for key, value in row.items()
                    if key != "zone" and isinstance(value, int | float)
                }
                for row in tick["markets"]
            }
            state = MarketState(
                utc_timestamp=timestamp,
                zones=list(self._fixture["zones"]),
                markets=markets,
                asset_states=self._states.copy(),
            )
            bid_book = MFRRBidBook(policy.bids_for_state(state))
            for bid in bid_book.sorted_for_clearing():
                if bid.utc_timestamp != timestamp:
                    rejected.append(RejectedBid(bid=bid, reason_code="wrong_tick"))
                    continue
                if not self._clock.is_gate_open(bid, state):
                    rejected.append(RejectedBid(bid=bid, reason_code="gate_closed"))
                    continue
                if bid.zone not in markets:
                    rejected.append(RejectedBid(bid=bid, reason_code="zone_missing"))
                    continue
                asset_state = self._states[bid.asset_id]
                decision = self._constraint_provider.validate_bid(bid, asset_state)
                if not decision.accepted:
                    rejected.append(
                        RejectedBid(
                            bid=bid,
                            reason_code=decision.reason_code or "physical_rejected",
                        )
                    )
                    continue

                settlement = _settlement_eur(bid, markets[bid.zone])
                accepted_bid = AcceptedBid(
                    bid=bid,
                    settlement_eur=round(settlement, 6),
                    projected_power_mw=round(decision.projected_power_mw, 6),
                    projected_thermal_soc_mwh=round(
                        decision.projected_thermal_soc_mwh, 6
                    ),
                    clearing_market="mFRR",
                    accepted_at_utc=self._clock.acceptance_timestamp(state),
                )
                accepted.append(accepted_bid)
                self._states[bid.asset_id] = SimulatorAssetState.for_asset(
                    bid.asset_id,
                    electric_power_mw=decision.projected_power_mw,
                    thermal_soc_mwh=decision.projected_thermal_soc_mwh,
                )

        return SimulationResult(
            tick_count=int(self._fixture["tick_count"]),
            zones=list(self._fixture["zones"]),
            accepted_bids=accepted,
            rejected_bids=rejected,
            final_asset_states=self._states.copy(),
            result_hash=_result_hash(accepted, rejected, self._states),
        )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _settlement_eur(bid: Bid, market: dict[str, float]) -> float:
    if bid.side == "down":
        price_delta = (
            market["spot_price_eur_mwh"]
            - market["mfrr_marginal_price_down_eur_mwh"]
        )
    else:
        price_delta = (
            market["mfrr_marginal_price_up_eur_mwh"]
            - market["spot_price_eur_mwh"]
        )
    return bid.quantity_mwh * price_delta


def _bid_payload(bid: Bid) -> dict:
    payload = bid.model_dump(mode="json")
    payload["utc_timestamp"] = bid.utc_iso()
    return payload


def _result_hash(
    accepted: list[AcceptedBid],
    rejected: list[RejectedBid],
    states: dict[str, SimulatorAssetState],
) -> str:
    payload = {
        "accepted": [
            {
                "bid": _bid_payload(row.bid),
                "settlement_eur": row.settlement_eur,
                "projected_power_mw": row.projected_power_mw,
                "projected_thermal_soc_mwh": row.projected_thermal_soc_mwh,
                "clearing_market": row.clearing_market,
                "accepted_at_utc": (
                    row.accepted_at_utc.isoformat().replace("+00:00", "Z")
                    if row.accepted_at_utc
                    else None
                ),
            }
            for row in accepted
        ],
        "rejected": [
            {"bid": _bid_payload(row.bid), "reason_code": row.reason_code}
            for row in rejected
        ],
        "states": {key: asdict(value) for key, value in sorted(states.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def result_to_dict(result: SimulationResult) -> dict:
    return {
        "schema_version": "1.0.0",
        "tick_count": result.tick_count,
        "zones": result.zones,
        "result_hash": result.result_hash,
        "accepted_bids": [
            {
                "bid": _bid_payload(row.bid),
                "settlement_eur": row.settlement_eur,
                "projected_power_mw": row.projected_power_mw,
                "projected_thermal_soc_mwh": row.projected_thermal_soc_mwh,
                "clearing_market": row.clearing_market,
                "accepted_at_utc": (
                    row.accepted_at_utc.isoformat().replace("+00:00", "Z")
                    if row.accepted_at_utc
                    else None
                ),
            }
            for row in result.accepted_bids
        ],
        "rejected_bids": [
            {"bid": _bid_payload(row.bid), "reason_code": row.reason_code}
            for row in result.rejected_bids
        ],
        "final_asset_states": {
            key: asdict(value) for key, value in sorted(result.final_asset_states.items())
        },
    }


def write_result(result: SimulationResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
