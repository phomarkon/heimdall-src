"""Empirical-coverage smoke tests for split-CP. Per docs/RESEARCH-PROPOSAL.md §4.6
Theorem 1a — under exchangeability, marginal coverage is >= 1 - alpha
(finite-sample), with a small over-coverage of order O(1/n)."""

from __future__ import annotations

import numpy as np
import pytest

from heimdall_ml import FROZEN_SEEDS
from heimdall_ml.conformal import SplitConformal, fit_quantile, predict_in_band


def _empirical_coverage(alpha: float, n_cal: int, n_test: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    cal = np.abs(rng.standard_normal(n_cal))  # nonconformity scores |y - y_hat|
    test = np.abs(rng.standard_normal(n_test))
    q = fit_quantile(cal, alpha)
    return float(np.mean(test <= q))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_split_cp_meets_target_coverage_on_average(alpha: float) -> None:
    cov = np.mean(
        [_empirical_coverage(alpha, n_cal=500, n_test=2_000, seed=s) for s in FROZEN_SEEDS]
    )
    # Theorem 1a: marginal coverage >= 1 - alpha. We allow a small slack of
    # 1.5 / sqrt(n_test) to absorb one-sided sampling noise across seeds.
    target = 1.0 - alpha
    slack = 1.5 / np.sqrt(2_000)
    assert cov >= target - slack, f"cov={cov:.4f} below target {target:.4f} - {slack:.4f}"


def test_predict_in_band_matches_quantile() -> None:
    rng = np.random.default_rng(42)
    cal = np.abs(rng.standard_normal(200))
    q = fit_quantile(cal, alpha=0.1)
    assert predict_in_band(0.0, q) is True
    assert predict_in_band(q + 1e-9, q) is False


def test_class_api_round_trip() -> None:
    rng = np.random.default_rng(13)
    cal = np.abs(rng.standard_normal(300))
    cp = SplitConformal.fit(cal, alpha=0.1)
    lo, hi = cp.interval_from_point(np.array([0.0, 5.0]))
    assert (hi - lo > 0).all()
    assert cp.n_calibration == 300


def test_empty_calibration_raises() -> None:
    with pytest.raises(ValueError):
        fit_quantile([], alpha=0.1)


def test_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        fit_quantile([0.1, 0.2], alpha=0.0)
    with pytest.raises(ValueError):
        fit_quantile([0.1, 0.2], alpha=1.0)


def test_finite_sample_correction_returns_inf_for_tiny_calibration() -> None:
    # n=5, alpha=0.01 -> ceil(6 * 0.99) = 6 > n -> infinite quantile (no
    # finite bound is tight enough). Per Theorem 1a this is the correct,
    # honest answer rather than a silently-loose interval.
    q = fit_quantile([1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.01)
    assert np.isinf(q)
