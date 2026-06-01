"""Observable market state and clearing record. Per docs/RESEARCH-PROPOSAL.md §4.7."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heimdall_contracts.bids import BidAction, MarketKind


class MarketState(BaseModel):
    """Snapshot broadcast to agents in step 1 of each tick (§4.7).

    What is included is filtered per-agent by the information-access matrix
    (§4.2.1); this schema is the upper bound (i.e. what an agent with full
    real-time access would see).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick: int = Field(ge=0)
    timestamp: datetime
    zone: Literal["DK1", "DK2"]
    last_da_price_eur_per_mwh: float | None = None
    last_id_price_eur_per_mwh: float | None = None
    last_mfrr_price_eur_per_mwh: float | None = None
    imbalance_mw: float | None = None
    gate_closure_in_seconds: int | None = Field(default=None, ge=0)


class MarketClearing(BaseModel):
    """Per-bid clearing record produced in step 6 of each tick (§4.7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick: int = Field(ge=0)
    timestamp: datetime
    market: MarketKind
    cleared_price_eur_per_mwh: float
    bid: BidAction
    filled_mw: float = Field(ge=0.0)
    pnl_eur: float
