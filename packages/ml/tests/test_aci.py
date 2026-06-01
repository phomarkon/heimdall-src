"""Long-run coverage smoke tests for online ACI. Per docs/RESEARCH-PROPOSAL.md §4.6
Theorem 1b — empirical miscoverage converges to alpha as T -> infinity,
without exchangeability."""

from __future__ import annotations

import numpy as np
import pytest

from heimdall_ml.conformal import AdaptiveConformalInference


def test_aci_long_run_miscoverage_under_stationary_data() -> None:
    rng = np.random.default_rng(13)
    cp = AdaptiveConformalInference(alpha=0.1, gamma=0.02, window=500)
    cp.warm_start(np.abs(rng.standard_normal(500)))
    T = 5_000
    # Stationary stream of nonconformity scores.
    for _ in range(T):
        s = float(np.abs(rng.standard_normal()))
        cp.update(s)
    # Under stationary data ACI's miscoverage is within a small slack of alpha.
    assert abs(cp.empirical_miscoverage - 0.1) < 0.02


def test_aci_recovers_under_regime_shift() -> None:
    """Theorem 1b's distinguishing claim: long-run coverage holds even when
    exchangeability breaks. We simulate that with a half-stream variance jump."""
    rng = np.random.default_rng(42)
    cp = AdaptiveConformalInference(alpha=0.1, gamma=0.05, window=500)
    cp.warm_start(np.abs(rng.standard_normal(500)))
    T = 6_000
    for t in range(T):
        sigma = 1.0 if t < T // 2 else 3.0  # variance jumps mid-stream
        cp.update(float(np.abs(sigma * rng.standard_normal())))
    # Slack a bit larger than the stationary case because the run-up to the
    # post-shift quantile is finite-time; long-run target still hits.
    assert abs(cp.empirical_miscoverage - 0.1) < 0.04


def test_aci_invalid_args() -> None:
    with pytest.raises(ValueError):
        AdaptiveConformalInference(alpha=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalInference(gamma=0.0)


def test_aci_predict_in_band_consistency() -> None:
    cp = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
    cp.warm_start([0.5, 1.0, 1.5, 2.0, 2.5])
    q = cp.quantile()
    assert cp.predict_in_band(q) is True
    assert cp.predict_in_band(q + 1e-9) is False
