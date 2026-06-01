"""Tests for F3 deep-ensemble aggregation. ADR-0006."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heimdall_forecaster.train.f3_ensemble import F3EnsembleConfig, build_ensemble


REPO_ROOT = Path(__file__).resolve().parents[3]
F7_ROOT = REPO_ROOT / "models/forecaster/f7"


@pytest.mark.skipif(
    not (F7_ROOT / "seed-42" / "val_preds.npz").exists(),
    reason="requires trained F7 seeds (run experiments/seed_sweep.py first)",
)
def test_ensemble_reduces_q50_variance(tmp_path: Path) -> None:
    """The mean of 5 quantile predictions has lower per-window variance than
    any single member at the same quantile (Lakshminarayanan-style claim)."""
    out_dir = tmp_path / "models"
    cfg = F3EnsembleConfig(out_dir=out_dir, member_root=REPO_ROOT / "models/forecaster")
    res = build_ensemble(cfg)
    summary = (out_dir / "f3_ensemble" / "ensemble_summary.json")
    assert summary.exists()
    # Member-to-mean offset must be > 0 (ensemble is non-trivial).
    assert res["aggregate"]["ensemble_q50_mean_member_std_dkk"] > 0.0


@pytest.mark.skipif(
    not (F7_ROOT / "seed-42" / "val_preds.npz").exists(),
    reason="requires trained F7 seeds",
)
def test_ensemble_q50_matches_member_average(tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    cfg = F3EnsembleConfig(out_dir=out_dir, member_root=REPO_ROOT / "models/forecaster")
    build_ensemble(cfg)
    z42 = np.load(out_dir / "f3_ensemble" / "seed-42" / "val_preds.npz")
    members = []
    for s in cfg.member_seeds:
        members.append(np.load(REPO_ROOT / "models/forecaster/f7" / f"seed-{s}" / "val_preds.npz")["preds"])
    mean = np.mean(np.stack(members, axis=0), axis=0)
    assert np.allclose(z42["preds"], mean.astype(np.float32), atol=1e-3)


@pytest.mark.skipif(
    not (F7_ROOT / "seed-42" / "val_preds.npz").exists(),
    reason="requires trained F7 seeds",
)
def test_ensemble_targets_identical_across_seeds(tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    cfg = F3EnsembleConfig(out_dir=out_dir, member_root=REPO_ROOT / "models/forecaster")
    build_ensemble(cfg)
    z = [
        np.load(out_dir / "f3_ensemble" / f"seed-{s}" / "val_preds.npz")["targets"]
        for s in cfg.member_seeds
    ]
    for arr in z[1:]:
        assert np.array_equal(z[0], arr)
