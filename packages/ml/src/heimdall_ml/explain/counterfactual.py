"""Counterfactual explanations for the verifier-guarded LLM-agent society.

Per Plan v2 Track E. Two flavours:

1. **Forecast counterfactual** (Wachter-style): minimal feature perturbation
   such that the forecaster's q50 crosses a target threshold (e.g., flips the
   sign of the price prediction). Useful for "what would have to change to make
   wind look generation-positive at hour T?".

2. **Bid counterfactual**: minimal feature perturbation such that the verifier's
   verdict flips from REJECT → ACCEPT (or vice versa). Reuses the
   ``attribute_conformal_tail`` 1σ-sensitivity machinery from
   ``bid_attribution.py`` — we line-search along the steepest-descent direction
   to find the closest accepting perturbation under an L2 budget.

Both are continuous-only and feature-agnostic — the caller wires the
``forecaster_to_interval`` / ``forecaster_q50`` callables.

Outputs include an ``ExplanationRecord``-compatible payload so traces are
auditable via ``heimdall_ml.explain.trace_log``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from heimdall_contracts import BidAction, ConformalInterval
from heimdall_markets import worst_case_profit

from heimdall_ml.explain.bid_attribution import attribute_conformal_tail


@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual search."""

    found: bool
    """True iff a perturbation flipped the predicate within the budget."""
    delta_sigma: np.ndarray
    """Per-feature perturbation in σ-units; same shape as ``base_features``."""
    delta_l2_sigma: float
    """L2 norm of ``delta_sigma`` — total perturbation size in σ-units."""
    perturbed_features: np.ndarray
    """Raw feature vector after applying the perturbation."""
    iterations: int
    """Steps consumed by the search."""
    rationale: list[tuple[str, float]] = field(default_factory=list)
    """(feature_name, signed_delta_sigma) sorted by |delta| descending — the
    paper's "auditable rationale" string."""


def counterfactual_forecast(
    *,
    base_features: np.ndarray,
    feature_names: Sequence[str],
    feature_sigma: np.ndarray,
    forecaster_q50: Callable[[np.ndarray], float],
    target_q50: float,
    max_iters: int = 50,
    learning_rate: float = 0.25,
    sigma_budget: float = 6.0,
    finite_diff_eps_sigma: float = 0.25,
) -> CounterfactualResult:
    """Find a minimal perturbation that pushes q50 to ``target_q50``.

    Optimisation: vanilla projected gradient descent under an L2 σ-ball
    constraint. The forecaster is treated as black-box — gradient is estimated
    by central finite differences (one scalar evaluation per ±ε per feature),
    cost = O(F) per iteration. Acceptable for F < 100 and max_iters < 100.

    Stops when ``|q50(f + delta) - target_q50| < 1e-6`` (in q50 native units)
    or when ``max_iters`` is reached.
    """
    base = base_features.astype(np.float64).copy()
    sigma_safe = np.where(feature_sigma > 0, feature_sigma, 1.0)
    delta = np.zeros_like(base)
    last_residual = float(forecaster_q50(base) - target_q50)
    if abs(last_residual) < 1e-6:
        return CounterfactualResult(
            found=True,
            delta_sigma=np.zeros_like(base),
            delta_l2_sigma=0.0,
            perturbed_features=base.copy(),
            iterations=0,
        )

    for it in range(1, max_iters + 1):
        # Finite-diff gradient of q50 w.r.t. each feature, in σ-units.
        grad_sigma = np.zeros_like(base)
        f_now = base + delta * sigma_safe
        for i in range(base.size):
            if feature_sigma[i] == 0:
                continue
            f_plus = f_now.copy()
            f_plus[i] += finite_diff_eps_sigma * sigma_safe[i]
            f_minus = f_now.copy()
            f_minus[i] -= finite_diff_eps_sigma * sigma_safe[i]
            grad_sigma[i] = (forecaster_q50(f_plus) - forecaster_q50(f_minus)) / (
                2 * finite_diff_eps_sigma
            )

        # Move δ_sigma in the direction that reduces residual.
        # If forecaster predicts above target, we need grad·δ < 0 (descent).
        residual = float(forecaster_q50(f_now) - target_q50)
        if abs(residual) < 1e-6:
            return _result(found=True, delta=delta, base=base, sigma=sigma_safe,
                           feature_names=feature_names, iterations=it)
        # Newton-ish step: δ ← δ - lr * residual * grad / ||grad||²
        gnorm_sq = float((grad_sigma ** 2).sum())
        if gnorm_sq < 1e-12:
            break  # flat gradient — give up
        step = learning_rate * residual * grad_sigma / gnorm_sq
        new_delta = delta - step
        # Project onto L2 σ-ball.
        nrm = float(np.linalg.norm(new_delta))
        if nrm > sigma_budget:
            new_delta = new_delta * (sigma_budget / nrm)
        # Accept the step if residual magnitude shrinks; else halve learning rate.
        candidate_residual = float(
            forecaster_q50(base + new_delta * sigma_safe) - target_q50
        )
        if abs(candidate_residual) < abs(residual):
            delta = new_delta
            last_residual = candidate_residual
        else:
            learning_rate *= 0.5
            if learning_rate < 1e-4:
                break

    return _result(
        found=abs(last_residual) < 1e-3,
        delta=delta,
        base=base,
        sigma=sigma_safe,
        feature_names=feature_names,
        iterations=max_iters,
    )


def counterfactual_bid(
    *,
    bid: BidAction,
    base_features: np.ndarray,
    feature_names: Sequence[str],
    feature_sigma: np.ndarray,
    forecaster_to_interval: Callable[[np.ndarray], ConformalInterval],
    tau: float,
    max_iters: int = 30,
    sigma_budget: float = 6.0,
    line_search_steps: int = 16,
) -> CounterfactualResult:
    """Find a minimal perturbation that flips the verifier verdict.

    Strategy: take the steepest-ascent direction of ``worst_case_profit`` w.r.t.
    feature σ-perturbations (using the sensitivities already produced by
    ``attribute_conformal_tail``), then bisect along that direction to find the
    smallest σ-scaled step that achieves ``worst_case_profit ≥ tau``.

    The current verdict is assumed to be REJECT (i.e. base pi < tau). For
    already-accepted bids, returns ``found=False, delta=0``.
    """
    base = base_features.astype(np.float64).copy()
    sigma_safe = np.where(feature_sigma > 0, feature_sigma, 1.0)
    base_iv = forecaster_to_interval(base)
    base_pi = worst_case_profit(bid, base_iv.lower, base_iv.upper)
    if base_pi >= tau:
        return _result(
            found=False, delta=np.zeros_like(base), base=base, sigma=sigma_safe,
            feature_names=feature_names, iterations=0,
        )

    # Direction: signed by which side of 1σ-perturb increases pi the most.
    sens = attribute_conformal_tail(
        bid, base, feature_names, feature_sigma, forecaster_to_interval
    )
    direction = np.zeros_like(base)
    for fs in sens:
        try:
            idx = feature_names.index(fs.feature)
        except ValueError:
            continue
        # Take the side (±) that increased pi most.
        if fs.delta_pi_plus >= fs.delta_pi_minus:
            direction[idx] = 1.0 if fs.delta_pi_plus > 0 else 0.0
        else:
            direction[idx] = -1.0 if fs.delta_pi_minus > 0 else 0.0
    nrm = float(np.linalg.norm(direction))
    if nrm == 0.0:
        return _result(found=False, delta=np.zeros_like(base), base=base, sigma=sigma_safe,
                       feature_names=feature_names, iterations=0)
    direction = direction / nrm

    # Bisection along ``direction`` for the smallest step that achieves pi >= tau.
    lo, hi = 0.0, sigma_budget
    best_delta = None
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        delta_sigma = direction * mid
        f = base + delta_sigma * sigma_safe
        iv = forecaster_to_interval(f)
        pi = worst_case_profit(bid, iv.lower, iv.upper)
        if pi >= tau:
            best_delta = delta_sigma
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-3:
            break

    if best_delta is None:
        return _result(found=False, delta=np.zeros_like(base), base=base, sigma=sigma_safe,
                       feature_names=feature_names, iterations=max_iters)
    return _result(
        found=True,
        delta=best_delta,
        base=base,
        sigma=sigma_safe,
        feature_names=feature_names,
        iterations=max_iters,
    )


def _result(
    *,
    found: bool,
    delta: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    feature_names: Sequence[str],
    iterations: int,
) -> CounterfactualResult:
    delta_sigma = delta
    rationale = sorted(
        ((feature_names[i], float(delta_sigma[i])) for i in range(len(delta_sigma))),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return CounterfactualResult(
        found=found,
        delta_sigma=delta_sigma,
        delta_l2_sigma=float(np.linalg.norm(delta_sigma)),
        perturbed_features=base + delta_sigma * sigma,
        iterations=iterations,
        rationale=rationale,
    )


__all__ = [
    "CounterfactualResult",
    "counterfactual_bid",
    "counterfactual_forecast",
]
