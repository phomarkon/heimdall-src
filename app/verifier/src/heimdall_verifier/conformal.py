"""Conformal-stage check (Stage 2). Per docs/RESEARCH-PROPOSAL.md §4.5 + Theorem 1a/b
in §4.6.

Given a candidate bid `a`, a price interval `[lower, upper]` produced by either
split-CP (Theorem 1a) or online ACI (Theorem 1b), and a loss threshold `tau`,
this function evaluates `pi_min(a)` (worst-case profit) closed-form via the
shared `heimdall_markets.profit.worst_case_profit` (the *same* function used by
the simulator — this identity is what Theorem 1's footnote requires).

Contract:
    accept iff pi_min(a) >= tau
"""

from __future__ import annotations

from heimdall_contracts import BidAction, ConformalInterval
from heimdall_markets import worst_case_profit


def conformal_check(
    bid: BidAction,
    interval: ConformalInterval,
    tau_eur: float,
) -> tuple[bool, float]:
    """Returns (accepted, worst_case_profit_eur).

    Coverage flows through this evaluation: by Theorem 1a (or 1b), with
    probability >= 1 - alpha (resp. long-run), the realised price `p_t` lies
    in `[lower, upper]`. Conditional on that event, realised profit is at
    least `pi_min(a)`. Hence acceptance gives `Pr(profit >= tau) >= 1 - alpha`.
    """
    pi_min = worst_case_profit(bid, interval.lower, interval.upper)
    return pi_min >= tau_eur, pi_min
