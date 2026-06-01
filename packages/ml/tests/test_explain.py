"""SHAP smoke + bid-attribution unit tests."""

from __future__ import annotations

import numpy as np

from heimdall_contracts import (
    PhysicalViolation,
    VerifierVerdict,
)
from heimdall_ml.explain import (
    ExplanationRecord,
    attribute_verifier_verdict,
    toy_shap_example,
)


def test_toy_shap_runs_and_has_correct_shape() -> None:
    res = toy_shap_example(seed=13)
    assert res.values.shape == (4,)
    # Linear SHAP: a fitted LinearRegression on noisy linear data should give
    # SHAP values whose signs match the true coefficients (1.5, -2, 0.5, 0).
    # Values are *contributions* for a single sample; sign agreement with
    # coef * (x - mean) is the right comparator. We only sanity-check that
    # at least the dominant features have non-trivial magnitudes.
    assert np.max(np.abs(res.values)) > 0.0


def test_attribute_accepted_verdict() -> None:
    v = VerifierVerdict(accepted=True, alpha=0.1)
    out = attribute_verifier_verdict(v)
    assert out["binding_constraint"] is None
    assert out["stage_failed"] is None


def test_attribute_physical_rejection_ramp() -> None:
    pv = PhysicalViolation(
        constraint="ramp_limit",
        current_value=12.0,
        bound_value=8.0,
        suggestion="reduce by >=4 MW",
    )
    v = VerifierVerdict(accepted=False, stage_failed="physical", physical_violation=pv)
    out = attribute_verifier_verdict(v)
    assert out["binding_constraint"] == "ramp_limit"
    assert out["slack"] == -4.0


def test_attribute_conformal_rejection() -> None:
    v = VerifierVerdict(
        accepted=False,
        stage_failed="conformal",
        worst_case_profit_eur=-150.0,
        threshold_eur=-100.0,
        alpha=0.1,
    )
    out = attribute_verifier_verdict(v)
    assert out["binding_constraint"] == "worst_case_profit"
    assert out["slack"] == -50.0


def test_explanation_record_round_trip() -> None:
    r = ExplanationRecord(
        run_id="run-1",
        step=0,
        agent_id="focal",
        decision_id="d-0",
        kind="verifier_attribution",
        payload={"slack": -50.0},
    )
    again = ExplanationRecord.model_validate_json(r.model_dump_json())
    assert again.payload == {"slack": -50.0}
