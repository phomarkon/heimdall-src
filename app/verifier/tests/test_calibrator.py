"""Tests for ``calibrator.py`` — wires real F7/F8 val residuals into a
``ConformalInterval`` consumed by the existing ``conformal_check``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heimdall_verifier.calibrator import CalibratedForecaster, split_cp_interval


def _write_synth_val_preds(tmp_path: Path, n: int = 500) -> Path:
    rng = np.random.default_rng(13)
    targets = rng.standard_normal((n, 16)) * 50.0  # std 50
    # q10 / q50 / q90 around the truth with a 30-DKK over-spread.
    preds = np.stack(
        [
            targets - 60.0,  # q10 too tight
            targets + rng.standard_normal((n, 16)) * 5.0,  # q50 with small noise
            targets + 60.0,
        ],
        axis=-1,
    )
    p = tmp_path / "val_preds.npz"
    np.savez(p, preds=preds, targets=targets)
    return p


def test_calibrated_forecaster_warm_start_from_synth(tmp_path: Path) -> None:
    p = _write_synth_val_preds(tmp_path)
    cal = CalibratedForecaster.from_val_preds(p, alpha=0.1)
    interval = cal.interval(point_pred=100.0)
    assert interval.method == "aci"
    assert interval.lower < 100.0 < interval.upper
    # The ACI quantile must be finite given a 500-row warm-start.
    assert np.isfinite(interval.lower)
    assert np.isfinite(interval.upper)


def test_split_cp_interval_centered_around_point(tmp_path: Path) -> None:
    p = _write_synth_val_preds(tmp_path)
    interval = split_cp_interval(p, point_pred=100.0, alpha=0.1)
    assert interval.method == "split_cp"
    assert (interval.upper - 100.0) == pytest.approx(100.0 - interval.lower)


def test_calibrator_update_changes_aci_state(tmp_path: Path) -> None:
    p = _write_synth_val_preds(tmp_path)
    cal = CalibratedForecaster.from_val_preds(p, alpha=0.1)
    before = cal.aci.alpha_t
    # Feed a sequence of misses; ACI should *increase* alpha_t (widen interval) ... wait,
    # the Gibbs-Candes update is alpha_{t+1} = alpha_t + gamma*(alpha - 1{miss}). Misses
    # decrease alpha_t (tighten the target miscoverage budget).
    for _ in range(20):
        cal.update(realised=10_000.0, point_pred=0.0)  # huge residual ⇒ miss
    assert cal.aci.alpha_t < before  # state moved
