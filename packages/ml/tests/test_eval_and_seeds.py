"""Tests for `heimdall_ml.eval` and `heimdall_ml.seeds`."""

from __future__ import annotations

import numpy as np

from heimdall_ml import FROZEN_SEEDS, seed_everything
from heimdall_ml.eval import (
    conditional_coverage,
    interval_width,
    marginal_coverage,
    pinball_loss,
)


def test_frozen_seeds_match_proposal() -> None:
    assert FROZEN_SEEDS == (13, 42, 137, 1729, 31415)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(13)
    a = np.random.random(8)
    seed_everything(13)
    b = np.random.random(8)
    assert np.allclose(a, b)


def test_marginal_coverage_at_endpoints() -> None:
    y = np.array([0.0, 1.0, 2.0])
    lo = np.array([-1.0, 0.0, 3.0])
    hi = np.array([1.0, 2.0, 4.0])
    assert marginal_coverage(y, lo, hi) == 2 / 3


def test_conditional_coverage_buckets() -> None:
    y = np.array([0.0, 0.0, 5.0, 5.0])
    lo = np.array([-1, -1, -1, -1])
    hi = np.array([1, 1, 1, 6])
    cov = conditional_coverage(y, lo, hi, strata=[0, 0, 1, 1])
    assert cov[0] == 1.0
    assert cov[1] == 0.5


def test_interval_width_basic() -> None:
    w = interval_width([0, 1], [2, 4])
    assert w.tolist() == [2.0, 3.0]


def test_pinball_loss_at_median_equals_half_mae() -> None:
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 2.0, 2.0])
    pinball = pinball_loss(y, q, level=0.5)
    expected = 0.5 * np.mean(np.abs(y - q))
    assert abs(pinball - expected) < 1e-12
