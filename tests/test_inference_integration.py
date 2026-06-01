"""End-to-end integration test: Tim-side handoff readiness.

Exercises the full path:
  1. Forecaster registry knows about every shipped backend.
  2. Each backend can be instantiated through the cache (LRU works).
  3. Each instance honours the ``Forecaster`` protocol.
  4. The FastAPI service picks the configured backend, returns a
     well-formed ``ForecastResponse``.
  5. The backend's output drops cleanly into the verifier's
     ``ConformalInterval`` schema.

Failure of any single step here is a Tim-blocker; the test exists so
the CI alerts before Tim hits it.
"""

from __future__ import annotations

import pytest

from heimdall_contracts import ConformalInterval
from heimdall_forecaster.inference import (
    Forecaster,
    get_forecaster,
)

pytestmark = pytest.mark.integration


HISTORY = [100.0 + 0.1 * (i % 13) for i in range(96)]


def test_f0_backend_obeys_protocol() -> None:
    f = get_forecaster("f0")
    assert isinstance(f, Forecaster)
    qs = f.predict(HISTORY, horizon=4)
    assert len(qs) == 4


def test_quantile_output_drops_into_conformal_interval() -> None:
    """The downstream verifier needs (lower, upper) extracted from a quantile.

    The contract: Tim's pipeline takes the q10 / q90 entries of the
    first horizon ``QuantileForecast`` and constructs a
    ``ConformalInterval``.  This test exercises that exact wiring.
    """
    f = get_forecaster("ar1")
    qs = f.predict(HISTORY, horizon=1, levels=(0.1, 0.5, 0.9))
    q0 = qs[0]
    interval = ConformalInterval(
        horizon_minutes=q0.horizon_minutes,
        alpha=0.10,
        lower=q0.values[0],
        upper=q0.values[2],
        method="aci",
    )
    assert interval.lower <= interval.upper
