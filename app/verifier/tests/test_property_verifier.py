"""Property-based tests for the verifier's conformal stage (Stage 2).

The contract (apps/verifier/.../conformal.py) is ``accept iff pi_min(a) >= tau``.
The thesis claim is stronger: *conditional on the realised price landing in the
conformal interval*, an accepted bid realises profit >= tau. These properties
pin both, over randomised bids / intervals / thresholds, and assert
proposer-invariance (the verdict depends only on the bid + interval + tau, never
on who proposed the bid — the property that makes the guarantee hold "regardless
of LLM hallucinations").
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from heimdall_contracts import BidAction, ConformalInterval
from heimdall_markets import realized_profit
from heimdall_verifier.conformal import conformal_check
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.property

_DELIVERY = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)

_prices = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_qty = st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _bids(draw) -> BidAction:
    return BidAction(
        market="mFRR",
        direction=draw(st.sampled_from(["buy", "sell"])),
        quantity_mw=draw(_qty),
        price_eur_per_mwh=draw(_prices),
        delivery_quarter=_DELIVERY,
        duration_minutes=draw(st.sampled_from([15, 30, 45, 60])),
    )


@st.composite
def _intervals(draw) -> ConformalInterval:
    lo = draw(_prices)
    hi = lo + draw(st.floats(0.0, 2e4, allow_nan=False, allow_infinity=False))
    return ConformalInterval(
        horizon_minutes=draw(st.sampled_from([15, 30, 60])),
        alpha=draw(st.floats(0.01, 0.5)),
        lower=lo,
        upper=hi,
        method=draw(st.sampled_from(["split_cp", "aci"])),
    )


@given(bid=_bids(), interval=_intervals(), tau=_prices, t=st.floats(0.0, 1.0))
def test_accepted_bid_realises_at_least_tau_inside_interval(
    bid: BidAction, interval: ConformalInterval, tau: float, t: float
) -> None:
    accepted, _pi_min = conformal_check(bid, interval, tau_eur=tau)
    clearing = interval.lower + t * (interval.upper - interval.lower)
    if accepted:
        # The verifier's promise: if the price lands anywhere in the band, an
        # accepted bid clears at least the threshold. This is the whole point.
        assert realized_profit(bid, clearing) >= tau - 1e-6


@given(bid=_bids(), interval=_intervals(), tau=_prices)
def test_accept_iff_worst_case_at_least_tau(
    bid: BidAction, interval: ConformalInterval, tau: float
) -> None:
    accepted, pi_min = conformal_check(bid, interval, tau_eur=tau)
    assert accepted == (pi_min >= tau)


@given(bid=_bids(), interval=_intervals(), tau=_prices)
def test_verdict_is_proposer_invariant_and_deterministic(
    bid: BidAction, interval: ConformalInterval, tau: float
) -> None:
    # Same (bid, interval, tau) -> identical verdict, no hidden state. The check
    # cannot depend on provenance, so an LLM and a script that emit the same bid
    # get the same ruling.
    a1, p1 = conformal_check(bid, interval, tau_eur=tau)
    a2, p2 = conformal_check(bid, interval, tau_eur=tau)
    assert (a1, p1) == (a2, p2)
