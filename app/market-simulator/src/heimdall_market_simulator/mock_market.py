"""In-memory mFRR mock market. Deterministic given a seed.

Each `step()` advances one 15-minute tick: emits a `MarketState`, accepts a
list of `BidAction`s, runs a trivial uniform-price clearing against an AR(1)
shadow price process, and returns `MarketClearing`s. This is *not* a model
of the real Nordic clearing — it's a contract-driven test fixture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from heimdall_contracts import BidAction, MarketClearing, MarketState
from heimdall_markets import realized_profit


class MockMFRRMarket:
    def __init__(
        self,
        *,
        zone: Literal["DK1", "DK2"] = "DK1",
        seed: int = 13,
        start: datetime | None = None,
        phi: float = 0.7,
        sigma: float = 5.0,
        mean_price: float = 50.0,
    ) -> None:
        self.zone = zone
        self._rng = np.random.default_rng(seed)
        self._tick = 0
        self._now = start or datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        self._phi = phi
        self._sigma = sigma
        self._mean = mean_price
        self._last_price = mean_price

    @property
    def now(self) -> datetime:
        return self._now

    def state(self) -> MarketState:
        return MarketState(
            tick=self._tick,
            timestamp=self._now,
            zone=self.zone,
            last_mfrr_price_eur_per_mwh=self._last_price,
            gate_closure_in_seconds=900,  # 15 min ahead
        )

    def step(self, bids: list[BidAction]) -> tuple[MarketState, list[MarketClearing]]:
        # Shadow price evolves AR(1) around `mean_price`.
        eps = self._sigma * float(self._rng.standard_normal())
        new_price = self._mean + self._phi * (self._last_price - self._mean) + eps

        clearings: list[MarketClearing] = []
        for bid in bids:
            pnl = realized_profit(bid, new_price)
            filled = (
                bid.quantity_mw
                if (bid.direction == "sell" and new_price >= bid.price_eur_per_mwh)
                or (bid.direction == "buy" and new_price <= bid.price_eur_per_mwh)
                else 0.0
            )
            clearings.append(
                MarketClearing(
                    tick=self._tick,
                    timestamp=self._now,
                    market=bid.market,
                    cleared_price_eur_per_mwh=new_price,
                    bid=bid,
                    filled_mw=filled,
                    pnl_eur=pnl,
                )
            )

        self._last_price = new_price
        self._now = self._now + timedelta(minutes=15)
        self._tick += 1
        return self.state(), clearings
