"""Forecaster smoke tests. CPU-only by design (peer agent owns the B200)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from heimdall_forecaster import AR1FallbackForecaster
from heimdall_forecaster.fallback import synthetic_ar1
from heimdall_forecaster.service import ForecastRequest, app, forecast


def test_ar1_fallback_returns_one_forecast_per_horizon_step() -> None:
    y = synthetic_ar1(256, phi=0.7, sigma=1.0, seed=13)
    f = AR1FallbackForecaster(seed=13)
    qfs = f.fit_predict_quantiles(y, horizon=4)
    assert len(qfs) == 4
    for h, qf in enumerate(qfs, start=1):
        assert qf.horizon_minutes == 15 * h
        assert qf.values[0] <= qf.values[1] <= qf.values[2]


def test_quantile_levels_have_approx_calibrated_coverage_under_iid() -> None:
    """A *coarse* sanity check: the 80% bootstrap interval should cover the
    realised next step ~80% of the time on synthetic AR(1). Tolerant
    threshold so CI is not flaky on rare seeds."""
    rng = np.random.default_rng(13)
    cov = []
    for _ in range(200):
        seed = int(rng.integers(0, 1_000_000))
        y = synthetic_ar1(128, phi=0.6, sigma=1.0, seed=seed)
        f = AR1FallbackForecaster(seed=seed)
        # Forecast at horizon 1 from a leave-one-out split.
        qfs = f.fit_predict_quantiles(y[:-1], horizon=1)
        lo, _, hi = qfs[0].values
        cov.append(int(lo <= y[-1] <= hi))
    # Target 0.8 under bootstrap; allow generous slack (small-T effects).
    assert 0.65 <= np.mean(cov) <= 0.95


def test_forecast_endpoint_round_trips() -> None:
    y = synthetic_ar1(64, seed=42).tolist()
    # Pin the CPU-friendly ar1 backend explicitly: the service default is now
    # "auto" (resolves to f7), so a bare request no longer round-trips ar1.
    req = ForecastRequest(history=y, horizon=2, levels=(0.1, 0.5, 0.9), backend="ar1")
    resp = forecast(req)
    assert resp.backend_used == "ar1"
    assert len(resp.quantiles) == 2


def test_healthz_route_reports_backend_registry() -> None:
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    # healthz now advertises the default backend and the registered zoo.
    assert "default_backend" in body
    assert isinstance(body["registered_backends"], list)
    assert any(b["name"] == "ar1" for b in body["registered_backends"])
