from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Bid(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    asset_id: str
    zone: Literal["DK1", "DK2"]
    utc_timestamp: datetime
    side: Literal["up", "down"]
    quantity_mwh: float = Field(gt=0)
    limit_price_eur_mwh: float
    submitted_at_utc: datetime | None = None

    @field_validator("utc_timestamp", "submitted_at_utc", mode="before")
    @classmethod
    def _parse_utc_timestamp(cls, value: object) -> datetime:
        if value is None:
            return value
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return parsed.astimezone(UTC)

    def utc_iso(self) -> str:
        return self.utc_timestamp.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SimulatorAssetState:
    asset_id: str
    electric_power_mw: float = 0.0
    thermal_soc_mwh: float | None = None

    @classmethod
    def for_asset(
        cls,
        asset_id: str,
        *,
        electric_power_mw: float = 0.0,
        thermal_soc_mwh: float | None = None,
    ) -> "SimulatorAssetState":
        return cls(
            asset_id=asset_id,
            electric_power_mw=electric_power_mw,
            thermal_soc_mwh=thermal_soc_mwh,
        )


@dataclass(frozen=True)
class ConstraintDecision:
    accepted: bool
    reason_code: str | None
    projected_power_mw: float
    projected_thermal_soc_mwh: float
    remaining_capacity_mw: float


@dataclass(frozen=True)
class MarketState:
    utc_timestamp: datetime
    zones: list[str]
    markets: dict[str, dict[str, float]]
    asset_states: dict[str, SimulatorAssetState]

    def utc_iso(self) -> str:
        return self.utc_timestamp.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AcceptedBid:
    bid: Bid
    settlement_eur: float
    projected_power_mw: float
    projected_thermal_soc_mwh: float
    clearing_market: str = "mFRR"
    accepted_at_utc: datetime | None = None


@dataclass(frozen=True)
class RejectedBid:
    bid: Bid
    reason_code: str


@dataclass(frozen=True)
class TickResult:
    utc_timestamp: datetime
    accepted_bids: list[AcceptedBid]
    rejected_bids: list[RejectedBid]


@dataclass(frozen=True)
class SimulationResult:
    tick_count: int
    zones: list[str]
    accepted_bids: list[AcceptedBid]
    rejected_bids: list[RejectedBid]
    final_asset_states: dict[str, SimulatorAssetState]
    result_hash: str
