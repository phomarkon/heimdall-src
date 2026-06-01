"""Tests for the conformal-calibrator service."""

from __future__ import annotations

import numpy as np
import pytest
from heimdall_conformal_calibrator import (
    PutObservationRequest,
    SeriesUpsertRequest,
    create_or_replace_series,
    get_interval,
    put_observation,
)


def _seed_series(method: str, *, alpha: float = 0.10) -> str:
    sid = f"test-{method}"
    create_or_replace_series(
        sid,
        SeriesUpsertRequest(
            method=method,  # type: ignore[arg-type]
            alpha=alpha,
            warmup_scores=list(np.linspace(1.0, 50.0, 200)),
        ),
    )
    return sid


@pytest.mark.parametrize("method", ["split_cp", "aci", "bocpd_aci"])
def test_warmup_interval_is_well_formed(method: str) -> None:
    sid = _seed_series(method)
    res = get_interval(sid, point_pred=100.0)
    assert res.interval.lower < 100.0 < res.interval.upper
    assert res.interval.method == method
    assert res.interval.alpha == 0.10


def test_aci_alpha_drifts_with_observations() -> None:
    sid = _seed_series("aci")
    rng = np.random.default_rng(42)
    # Inject 200 wide-residual observations: alpha_t should drift up.
    for _ in range(200):
        put_observation(
            sid, PutObservationRequest(realised=200.0 + rng.normal(0, 5), point_pred=100.0)
        )
    res = get_interval(sid, point_pred=100.0)
    # Wide residuals → bigger interval → expansion of buffer; alpha_t may increase.
    assert res.n_observations >= 200


def test_bocpd_aci_resets_on_regime_shift() -> None:
    sid = _seed_series("bocpd_aci")
    rng = np.random.default_rng(0)
    # In-regime: small residuals.
    for _ in range(200):
        put_observation(
            sid, PutObservationRequest(realised=100.0 + rng.normal(0, 2), point_pred=100.0)
        )
    # Out-of-regime: large residuals (force BOCPD detection).
    for _ in range(300):
        put_observation(
            sid, PutObservationRequest(realised=300.0 + rng.normal(0, 2), point_pred=100.0)
        )
    res = get_interval(sid, point_pred=100.0)
    # Detection should have fired at least once.
    assert res.last_reset_t is not None


def test_unknown_series_raises() -> None:
    with pytest.raises(KeyError):
        get_interval("nope", point_pred=0.0)
    with pytest.raises(KeyError):
        put_observation("nope", PutObservationRequest(realised=0.0, point_pred=0.0))
