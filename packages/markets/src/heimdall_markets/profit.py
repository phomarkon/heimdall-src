"""Profit functions for the mFRR/DA/ID markets.

Per docs/RESEARCH-PROPOSAL.md §4.5 (verifier Stage 2) and §4.6 (Theorem 1a/b
footnote): the verifier's `pi_min(a)` and the simulator's realised profit
`pi(a, p)` MUST be the same function — coverage flows through this identity.

For the day-1 simplified P2H market-maker case we model:

    pi(bid, p) = (p - bid.price) * filled_mw * dt_h     (sell)
                 (bid.price - p) * filled_mw * dt_h     (buy)

where `filled_mw = bid.quantity_mw if bid clears, else 0`. A sell clears iff
`p >= bid.price`; a buy clears iff `p <= bid.price`. This is the standard
limit-order semantics for a single bidder against a residual auction price.

For piecewise-linear payoffs in `p` (the case here), the worst-case profit
over an interval `[lo, hi]` is realised at one of the endpoints — no
optimisation needed at runtime.
"""

from __future__ import annotations

from heimdall_contracts import BidAction


def realized_profit(bid: BidAction, clearing_price: float) -> float:
    """Realised profit (EUR) for a single 15-minute bid at the given clearing price."""
    dt_h = bid.duration_minutes / 60.0
    if bid.direction == "sell":
        filled = bid.quantity_mw if clearing_price >= bid.price_eur_per_mwh else 0.0
        return (clearing_price - bid.price_eur_per_mwh) * filled * dt_h
    # buy
    filled = bid.quantity_mw if clearing_price <= bid.price_eur_per_mwh else 0.0
    return (bid.price_eur_per_mwh - clearing_price) * filled * dt_h


def worst_case_profit(bid: BidAction, lower: float, upper: float) -> float:
    """`pi_min(a)` over the interval [lower, upper]. Closed-form per §4.5.

    For `sell` orders profit is non-decreasing in `p` -> minimum at `lower`.
    For `buy`  orders profit is non-increasing in `p` -> minimum at `upper`.
    Either way, evaluating at both endpoints and taking the min is exact and
    cheap.
    """
    if upper < lower:
        raise ValueError("upper must be >= lower")
    return min(realized_profit(bid, lower), realized_profit(bid, upper))
