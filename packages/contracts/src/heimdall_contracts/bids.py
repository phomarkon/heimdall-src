"""Bid action schema. Per docs/RESEARCH-PROPOSAL.md Appendix B and §3 (mFRR EAM rules).

The `delivery_quarter` MUST be aligned to a 15-minute boundary in UTC — this is
enforced because the Nordic mFRR EAM (post-2025-03-04) operates on a strict
15-minute resolution. Misalignment is a physical-stage verifier rejection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MarketKind = Literal["DA", "ID", "mFRR"]
BidDirection = Literal["buy", "sell"]


class BidAction(BaseModel):
    """A single bid submitted to one of the three market venues.

    Cross-references:
    - docs/RESEARCH-PROPOSAL.md §4.5 (verifier inputs)
    - docs/RESEARCH-PROPOSAL.md Appendix B (canonical schema)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: MarketKind
    direction: BidDirection
    quantity_mw: float = Field(ge=0.0, description="Volume bid; non-negative MW.")
    price_eur_per_mwh: float = Field(description="Limit price in EUR/MWh.")
    delivery_quarter: datetime = Field(
        description="UTC timestamp at the start of the 15-min delivery quarter."
    )
    duration_minutes: int = Field(
        default=15,
        ge=15,
        description="Default 15 min per mFRR EAM. Multiple of 15.",
    )

    @field_validator("delivery_quarter")
    @classmethod
    def _utc_aligned_quarter(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        elif v.utcoffset() != v.tzinfo.utcoffset(v) or v.utcoffset().total_seconds() != 0:
            # require UTC; agents must convert beforehand
            raise ValueError("delivery_quarter must be in UTC")
        if v.minute % 15 != 0 or v.second != 0 or v.microsecond != 0:
            raise ValueError("delivery_quarter must align to a 15-minute UTC boundary")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def _multiple_of_15(cls, v: int) -> int:
        if v % 15 != 0:
            raise ValueError("duration_minutes must be a multiple of 15")
        return v
