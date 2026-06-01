"""Tests for the shared profit math (verifier <-> simulator identity)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from heimdall_contracts import BidAction
from heimdall_markets import realized_profit, worst_case_profit


def _bid(direction: str, price: float, qty: float = 10.0) -> BidAction:
    return BidAction(
        market="mFRR",
        direction=direction,  # type: ignore[arg-type]
        quantity_mw=qty,
        price_eur_per_mwh=price,
        delivery_quarter=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
    )


def test_sell_above_price_clears() -> None:
    b = _bid("sell", price=40.0, qty=10.0)
    # 15 min duration -> dt_h = 0.25
    assert realized_profit(b, clearing_price=50.0) == pytest.approx((50 - 40) * 10 * 0.25)


def test_sell_below_price_no_fill() -> None:
    b = _bid("sell", price=40.0)
    assert realized_profit(b, clearing_price=30.0) == 0.0


def test_buy_below_price_clears() -> None:
    b = _bid("buy", price=40.0, qty=10.0)
    assert realized_profit(b, clearing_price=30.0) == pytest.approx((40 - 30) * 10 * 0.25)


def test_worst_case_profit_sell_is_at_lower_endpoint() -> None:
    b = _bid("sell", price=40.0, qty=10.0)
    # interval [30, 50]: at 30 -> 0 fill, at 50 -> +10*10*0.25 -> min=0
    assert worst_case_profit(b, 30.0, 50.0) == 0.0


def test_worst_case_profit_buy_is_at_upper_endpoint() -> None:
    b = _bid("buy", price=40.0, qty=10.0)
    # interval [30, 50]: at 30 -> +25, at 50 -> 0 fill -> min=0
    assert worst_case_profit(b, 30.0, 50.0) == 0.0


def test_worst_case_profit_invalid_interval() -> None:
    b = _bid("sell", price=40.0)
    with pytest.raises(ValueError):
        worst_case_profit(b, lower=100.0, upper=50.0)
