"""Explainability hooks. Per the user's XAI emphasis + docs/RESEARCH-PROPOSAL.md §1.3
(Danfoss requirement: trustworthy / interpretable / robustness).

Day-2 scope: forecaster-level SHAP (``explain_forecast.explain``) live for
F7/F8 + the toy contract example. Verifier attribution unchanged from day 1."""

from heimdall_ml.explain.bid_attribution import attribute_verifier_verdict
from heimdall_ml.explain.explain_forecast import ForecastExplanation, explain
from heimdall_ml.explain.lime_forecast import LimeForecastExplanation, explain_lime
from heimdall_ml.explain.shap_forecast import toy_shap_example
from heimdall_ml.explain.trace_log import ExplanationRecord

__all__ = [
    "ExplanationRecord",
    "ForecastExplanation",
    "LimeForecastExplanation",
    "attribute_verifier_verdict",
    "explain",
    "explain_lime",
    "toy_shap_example",
]
