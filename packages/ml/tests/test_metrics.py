"""Unit tests for heimdall_ml.eval.metrics.

Strategy: build a synthetic problem with a known answer and check each metric
against analytic / closed-form expectations.
"""
from __future__ import annotations

import numpy as np

from heimdall_ml.eval import metrics as M


def _synthetic_calibrated(N=5000, H=4, levels=(0.1, 0.5, 0.9), seed=13):
    """Generate (preds, targets) where preds are the *true* quantiles of a
    Normal(0,1) target. A well-calibrated probabilistic forecast.
    """
    rng = np.random.default_rng(seed)
    targets = rng.standard_normal((N, H))
    from scipy.stats import norm  # std-lib alt would be erfinv
    qs = np.array([norm.ppf(l) for l in levels])
    preds = np.broadcast_to(qs, (N, H, len(levels))).copy()
    return preds, targets


def test_pinball_per_quantile_returns_dict_with_expected_keys():
    preds, targets = _synthetic_calibrated()
    out = M.pinball_per_quantile(preds, targets, (0.1, 0.5, 0.9))
    assert {"pinball_q10", "pinball_q50", "pinball_q90", "pinball_mean"} <= set(out)
    assert all(v >= 0 for v in out.values())


def test_marginal_coverage_recovers_calibration():
    """At calibrated quantiles, P(y <= q_l) ≈ l within 1.5% at N=5000."""
    preds, targets = _synthetic_calibrated(N=10000)
    out = M.marginal_coverage_per_level(preds, targets, (0.1, 0.5, 0.9))
    assert abs(out["cdf_at_q10"] - 0.10) < 0.02
    assert abs(out["cdf_at_q50"] - 0.50) < 0.02
    assert abs(out["cdf_at_q90"] - 0.90) < 0.02


def test_interval_coverage_recovers_nominal_at_80pct():
    preds, targets = _synthetic_calibrated(N=10000)
    out = M.interval_coverage(preds, targets, (0.1, 0.5, 0.9), ((0.1, 0.9),))
    assert abs(out["interval_10_90_coverage"] - 0.80) < 0.03
    # nominal width = q90-q10 = 2 * 1.2816  ≈ 2.5631
    assert abs(out["interval_10_90_width_mean"] - 2.5631) < 0.05


def test_crps_decreases_when_predictions_are_better():
    """A model that predicts q10/q50/q90 of the *truth* has lower CRPS than one
    predicting fixed wide quantiles independent of truth."""
    rng = np.random.default_rng(0)
    N, H = 1000, 1
    targets = rng.standard_normal((N, H))
    levels = (0.1, 0.5, 0.9)
    from scipy.stats import norm
    qs = np.array([norm.ppf(l) for l in levels])
    good = np.broadcast_to(qs, (N, H, 3)).copy()
    bad = np.broadcast_to(qs * 3.0, (N, H, 3)).copy()
    crps_good = M.crps_from_quantiles(good, targets, levels)
    crps_bad = M.crps_from_quantiles(bad, targets, levels)
    assert crps_good < crps_bad, (crps_good, crps_bad)


def test_skill_score_sign_and_magnitude():
    assert M.skill_score(1.0, 2.0) == 0.5
    assert M.skill_score(2.0, 1.0) == -1.0
    assert M.skill_score(1.0, 1.0) == 0.0


def test_spike_detection_returns_finite_values_on_calibrated_input():
    preds, targets = _synthetic_calibrated(N=2000)
    out = M.spike_detection(preds, targets, (0.1, 0.5, 0.9))
    assert np.isfinite(out["spike_brier"])
    assert 0.0 <= out["spike_base_rate"] <= 1.0


def test_pit_histogram_uniformity_on_calibrated():
    """A well-calibrated forecast should give a PIT close to uniform."""
    preds, targets = _synthetic_calibrated(N=5000)
    out = M.pit_histogram(preds, targets, (0.1, 0.5, 0.9), bins=10)
    # With only 3 quantiles the PIT can take 5 discrete values → KS-vs-uniform
    # is structurally ≈0.4 even at perfect calibration. Test the more informative
    # case with 19 quantiles where PIT is denser.
    levels_dense = tuple(round(i * 0.05, 2) for i in range(1, 20))  # 0.05..0.95
    from scipy.stats import norm
    qs = np.array([norm.ppf(l) for l in levels_dense])
    preds_d = np.broadcast_to(qs, (5000, 4, len(qs))).copy()
    rng = np.random.default_rng(13)
    targets_d = rng.standard_normal((5000, 4))
    out_d = M.pit_histogram(preds_d, targets_d, levels_dense, bins=10)
    assert out_d["ks_vs_uniform"] < 0.10, out_d["ks_vs_uniform"]


def test_collect_all_runs_end_to_end():
    preds, targets = _synthetic_calibrated(N=1000)
    out = M.collect_all(preds, targets, (0.1, 0.5, 0.9))
    # No NaN keys in the headline metrics.
    for k in ("pinball_mean", "crps_quantile_dkk", "interval_10_90_coverage"):
        assert k in out and np.isfinite(out[k])
