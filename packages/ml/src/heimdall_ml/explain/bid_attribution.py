"""Bid-level attribution: given a verifier verdict, identify which constraint
was binding *and* (for conformal rejections) which input feature most shifted
the worst-case-profit lower bound.

Per docs/RESEARCH-PROPOSAL.md §4.5 the verifier is piecewise-linear in the bid and
in the conformal interval endpoints, so the binding-constraint attribution is
exact (not an approximation). For the conformal-tail driver we fit a one-step
sensitivity by perturbing each forecaster input feature by ±1σ and reading off
the change in ``pi_min`` — a closed-form gradient-equivalent for piecewise-linear
profit. The output is a (feature, signed-sensitivity) record suitable for the
paper's auditability story (§5.3, "rationale quality" Likert).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from heimdall_contracts import BidAction, ConformalInterval, VerifierVerdict
from heimdall_markets import worst_case_profit


@dataclass
class FeatureSensitivity:
    feature: str
    delta_pi_plus: float  # change in pi_min when feature is +1 sigma
    delta_pi_minus: float  # change in pi_min when feature is -1 sigma
    abs_max_delta: float  # max(|delta_+|, |delta_-|), what we sort on


def attribute_verifier_verdict(verdict: VerifierVerdict) -> dict[str, str | float | None]:
    """Reduce a verdict to a single (binding_constraint, slack, suggestion) record.

    For physical violations the binding constraint is fully described in the
    ``physical_violation`` payload. For conformal-stage rejections the binding
    constraint is the worst-case-profit shortfall vs ``tau``.
    """
    if verdict.accepted:
        return {
            "stage_failed": None,
            "binding_constraint": None,
            "slack": None,
            "suggestion": None,
        }

    if verdict.stage_failed == "physical" and verdict.physical_violation is not None:
        pv = verdict.physical_violation
        return {
            "stage_failed": "physical",
            "binding_constraint": pv.constraint,
            "current_value": pv.current_value,
            "bound_value": pv.bound_value,
            "slack": pv.bound_value - pv.current_value,
            "suggestion": pv.suggestion,
        }

    if verdict.stage_failed == "conformal":
        wcp = verdict.worst_case_profit_eur
        tau = verdict.threshold_eur
        slack = (wcp - tau) if (wcp is not None and tau is not None) else None
        return {
            "stage_failed": "conformal",
            "binding_constraint": "worst_case_profit",
            "worst_case_profit_eur": wcp,
            "threshold_eur": tau,
            "slack": slack,
            "suggestion": verdict.retry_suggestion,
        }

    return {
        "stage_failed": verdict.stage_failed,
        "binding_constraint": "unknown",
        "slack": None,
        "suggestion": verdict.retry_suggestion,
    }


def attribute_conformal_tail(
    bid: BidAction,
    base_features: np.ndarray,
    feature_names: Sequence[str],
    feature_sigma: np.ndarray,
    forecaster_to_interval: Callable[[np.ndarray], ConformalInterval],
) -> list[FeatureSensitivity]:
    """For a conformal rejection, identify which input feature most shifts ``pi_min``.

    Parameters
    ----------
    bid:
        The candidate bid whose verdict we want to attribute.
    base_features:
        ``(F,)`` array of raw input features fed into the forecaster (already
        time-collapsed, e.g. the most recent step of each covariate).
    feature_names:
        ``(F,)`` names matching ``base_features``.
    feature_sigma:
        ``(F,)`` per-feature standard deviation from the training window.
    forecaster_to_interval:
        A closure that takes a perturbed feature vector and returns the
        resulting ``ConformalInterval`` (i.e. ``f → forecast → ACI``). The
        caller wires this; the explainer is forecaster-agnostic.

    Returns
    -------
    A list of ``FeatureSensitivity`` sorted by ``abs_max_delta`` descending —
    the top of the list is the feature that, if shifted by 1σ, would have the
    largest impact on the verifier's worst-case-profit calculation.
    """
    base_iv = forecaster_to_interval(base_features)
    base_pi = worst_case_profit(bid, base_iv.lower, base_iv.upper)

    out: list[FeatureSensitivity] = []
    for i, (name, sigma) in enumerate(zip(feature_names, feature_sigma, strict=True)):
        if sigma == 0.0:
            out.append(FeatureSensitivity(name, 0.0, 0.0, 0.0))
            continue
        f_plus = base_features.copy()
        f_plus[i] += sigma
        iv_plus = forecaster_to_interval(f_plus)
        d_plus = worst_case_profit(bid, iv_plus.lower, iv_plus.upper) - base_pi

        f_minus = base_features.copy()
        f_minus[i] -= sigma
        iv_minus = forecaster_to_interval(f_minus)
        d_minus = worst_case_profit(bid, iv_minus.lower, iv_minus.upper) - base_pi

        out.append(
            FeatureSensitivity(
                feature=name,
                delta_pi_plus=float(d_plus),
                delta_pi_minus=float(d_minus),
                abs_max_delta=float(max(abs(d_plus), abs(d_minus))),
            )
        )
    out.sort(key=lambda r: r.abs_max_delta, reverse=True)
    return out


__all__ = [
    "FeatureSensitivity",
    "attribute_conformal_tail",
    "attribute_verifier_verdict",
]
