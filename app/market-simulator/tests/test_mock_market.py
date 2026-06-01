from datetime import datetime, timezone

from heimdall_contracts import BidAction
from heimdall_market_simulator import MockMFRRMarket


def _b(price: float) -> BidAction:
    return BidAction(
        market="mFRR",
        direction="sell",
        quantity_mw=10.0,
        price_eur_per_mwh=price,
        delivery_quarter=datetime(2026, 5, 9, 0, 15, tzinfo=timezone.utc),
    )


def test_market_step_advances_tick_and_clears_when_in_money() -> None:
    m = MockMFRRMarket(seed=13, sigma=0.0, mean_price=50.0, phi=0.0)
    # phi=0 + sigma=0 -> shadow price exactly == 50.
    state, clearings = m.step([_b(price=40.0), _b(price=60.0)])
    assert state.tick == 1
    in_money, out_of_money = clearings
    assert in_money.filled_mw == 10.0
    assert out_of_money.filled_mw == 0.0
    assert in_money.pnl_eur > 0


def test_market_is_deterministic_given_seed() -> None:
    a = MockMFRRMarket(seed=42)
    b = MockMFRRMarket(seed=42)
    for _ in range(8):
        sa, _ = a.step([])
        sb, _ = b.step([])
        assert sa.last_mfrr_price_eur_per_mwh == sb.last_mfrr_price_eur_per_mwh
