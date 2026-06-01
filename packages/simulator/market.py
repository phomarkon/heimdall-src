from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .models import Bid, MarketState


@dataclass(frozen=True)
class MFRRMarketClock:
    submission_gate_minutes: float = 45.0
    acceptance_notice_minutes: float = 7.5

    def is_gate_open(self, bid: Bid, state: MarketState) -> bool:
        if bid.submitted_at_utc is None:
            return True
        return bid.submitted_at_utc <= self.submission_gate_timestamp(state)

    def submission_gate_timestamp(self, state: MarketState):
        return state.utc_timestamp - timedelta(minutes=self.submission_gate_minutes)

    def acceptance_timestamp(self, state: MarketState):
        return state.utc_timestamp - timedelta(minutes=self.acceptance_notice_minutes)


@dataclass
class MFRRBidBook:
    bids: list[Bid]

    def sorted_for_clearing(self) -> list[Bid]:
        return sorted(
            self.bids,
            key=lambda bid: (
                bid.utc_timestamp,
                bid.zone,
                bid.side,
                bid.limit_price_eur_mwh if bid.side == "down" else -bid.limit_price_eur_mwh,
                bid.agent_id,
                bid.asset_id,
            ),
        )
