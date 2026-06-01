"""Tests for the counterfactual XAI module.

Uses lightweight stand-in forecasters: linear and quadratic. Verifies:
- forecast counterfactual flips the q50 sign when sigma_budget allows,
- forecast counterfactual gives up gracefully when target is unreachable,
- bid counterfactual returns no-op when bid is already accepted,
- bid counterfactual finds a finite-σ perturbation along the steepest direction.
"""

from __future__ import annotations

import numpy as np

from heimdall_contracts import BidAction, ConformalInterval
from heimdall_ml.explain.counterfactual import (
    counterfactual_bid,
    counterfactual_forecast,
)


def test_forecast_counterfactual_reaches_target() -> None:
    # forecaster: q50 = 2 * f[0] - f[1]. Target q50 = 0 starting from f=(1, 0).
    def q50(x):
        return 2.0 * x[0] - x[1]

    base = np.array([1.0, 0.0])
    names = ("f0", "f1")
    sigma = np.array([1.0, 1.0])
    res = counterfactual_forecast(
        base_features=base,
        feature_names=names,
        feature_sigma=sigma,
        forecaster_q50=q50,
        target_q50=0.0,
        sigma_budget=4.0,
    )
    assert res.found, f"should reach q50=0 within 4σ budget; got delta={res.delta_sigma}"
    new_q = q50(res.perturbed_features)
    assert abs(new_q) < 1e-3


def test_forecast_counterfactual_respects_budget() -> None:
    def q50(x):
        return 100.0 * x[0]  # very steep but limited by 1σ budget

    res = counterfactual_forecast(
        base_features=np.array([5.0]),
        feature_names=("f0",),
        feature_sigma=np.array([1.0]),
        forecaster_q50=q50,
        target_q50=0.0,
        sigma_budget=1.0,
        max_iters=20,
    )
    # The target requires -5σ, budget is 1σ → should give up but stay within budget.
    assert res.delta_l2_sigma <= 1.0 + 1e-6


def test_bid_counterfactual_noop_when_accepted() -> None:
    # forecaster: interval centred at f[0], width 1.
    def to_interval(x):
        return ConformalInterval(
            lower=float(x[0] - 0.5),
            upper=float(x[0] + 0.5),
            horizon_minutes=15,
            alpha=0.1,
            method="aci",
        )

    from datetime import datetime, timezone
    bid = BidAction(
        market="mFRR",
        direction="sell",
        quantity_mw=1.0,
        price_eur_per_mwh=10.0,
        delivery_quarter=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    res = counterfactual_bid(
        bid=bid,
        base_features=np.array([20.0]),
        feature_names=("price",),
        feature_sigma=np.array([5.0]),
        forecaster_to_interval=to_interval,
        tau=0.0,  # very low threshold — already accepted
    )
    assert not res.found
    assert res.delta_l2_sigma == 0.0


def test_bid_counterfactual_finds_flip() -> None:
    """Construct a scenario where a moderate σ-perturbation flips the verdict."""

    def to_interval(x):
        # Interval centred at f[0], width 2 — high uncertainty.
        return ConformalInterval(
            lower=float(x[0] - 1.0),
            upper=float(x[0] + 1.0),
            horizon_minutes=15,
            alpha=0.1,
            method="aci",
        )

    from datetime import datetime, timezone
    # Buy bid at 15 EUR, base interval [10, 12] → both endpoints fill (price ≤ 15).
    # Worst-case profit = min((15-10)*1*0.25, (15-12)*1*0.25) = 0.75. tau = 1.1
    # → rejected. Move the interval *down* by negative δ to widen profit margin.
    bid = BidAction(
        market="mFRR",
        direction="buy",
        quantity_mw=1.0,
        price_eur_per_mwh=15.0,
        delivery_quarter=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    res = counterfactual_bid(
        bid=bid,
        base_features=np.array([11.0]),
        feature_names=("price",),
        feature_sigma=np.array([1.0]),
        forecaster_to_interval=to_interval,
        tau=1.1,
        sigma_budget=8.0,
    )
    assert res.found, f"expected a fix within 8σ; got {res}"
    assert res.delta_l2_sigma > 0
    assert res.delta_l2_sigma <= 8.0
