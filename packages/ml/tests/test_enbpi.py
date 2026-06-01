"""Unit + property tests for EnbPI (conformal trio member, ablation A8).

EnbPI had no dedicated test (it sat at ~26% line coverage). It is the third
conformal paradigm next to split-CP (Theorem 1a) and ACI (Theorem 1b), so the
suite should exercise it to the same standard: shape/alpha guards, both
warm-start branches, the small-buffer corner, and the structural invariants
(symmetric interval, non-negative width, statistical coverage on stationary
data).
"""

from __future__ import annotations

import numpy as np
import pytest
from heimdall_ml.conformal.enbpi import EnbPIResult, enbpi_intervals


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        enbpi_intervals([1.0, 2.0, 3.0], [1.0, 2.0])


def test_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        enbpi_intervals([1.0, 2.0], [1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        enbpi_intervals([1.0, 2.0], [1.0, 2.0], alpha=1.0)


def test_result_shape_and_step_count() -> None:
    rng = np.random.default_rng(13)
    n_obs, window = 400, 100
    p = rng.normal(0, 1, n_obs)
    y = p + rng.normal(0, 1, n_obs)
    res = enbpi_intervals(p, y, alpha=0.1, window=window, warm=window)
    assert isinstance(res, EnbPIResult)
    assert res.n_steps == n_obs - window
    assert res.intervals.shape == (n_obs - window, 2)
    assert res.window == window


