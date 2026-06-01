"""F7/F8 training plumbing tests.

End-to-end ``train_model`` is exercised on a tiny synthetic panel: this
verifies wiring (dataset → model → loss → checkpoint) without spending GPU
time. Real training is run from ``train/run.py`` against the persisted DK1
panel; that's covered by the manual sprint runbook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from heimdall_forecaster.train.dataset import (
    F8_FEATURES,
    QuantilePanelDataset,
    make_windows,
)
from heimdall_forecaster.train.activation_direction_classifier import (
    ActivationDirectionConfig,
    train_activation_direction,
)
from heimdall_forecaster.train.model import PatchTransformerQuantile, quantile_loss
from heimdall_forecaster.train.f5_f6_neural_process import NeuralProcessConfig, train_neural_process
from heimdall_forecaster.train.trainer import TrainConfig, train_model


def _synth_panel(n: int = 256, seed: int = 13) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100 + 30 * np.sin(2 * np.pi * t / 96)
    price = base + 10 * rng.standard_normal(n)
    load = 1500 + 100 * np.sin(2 * np.pi * t / 96) + 20 * rng.standard_normal(n)
    da = base + 5 * rng.standard_normal(n)
    return pl.DataFrame(
        {
            "timestamp_utc": [f"2025-03-04T00:{i:02d}:00" for i in range(n)],
            "imbalance_price_dkk_mwh_15min": price.astype(np.float64),
            "load_actual_mw": load.astype(np.float64),
            "da_price_dkk_mwh": da.astype(np.float64),
            "mfrr_up_volume_mw": np.where(price > base, 12.0, 0.0),
            "mfrr_down_volume_mw": np.where(price < base - 5.0, 8.0, 0.0),
        }
    )


def test_make_windows_shapes(tmp_path: Path) -> None:
    panel = _synth_panel(256)
    p = tmp_path / "panel.parquet"
    panel.write_parquet(p)
    X, Y, stats = make_windows(p, seq_len=96, horizon=16, multivariate=False)
    assert X.shape == (256 - 96 - 16 + 1, 96, 1)
    assert Y.shape == (X.shape[0], 16)
    assert stats.feature_names == ("imbalance_price_dkk_mwh_15min",)
    # Inputs are normalised; targets are normalised too.
    assert abs(X.mean()) < 0.5


def test_make_windows_multivariate(tmp_path: Path) -> None:
    panel = _synth_panel(256)
    p = tmp_path / "panel.parquet"
    panel.write_parquet(p)
    X, Y, stats = make_windows(p, multivariate=True)
    assert X.shape[-1] == len(F8_FEATURES) == 3
    assert stats.feature_names == F8_FEATURES


def test_make_windows_activation_targets(tmp_path: Path) -> None:
    panel = _synth_panel(128)
    p = tmp_path / "panel.parquet"
    panel.write_parquet(p)
    _, volume, volume_stats = make_windows(p, seq_len=16, horizon=4, target="activation_volume")
    _, direction, direction_stats = make_windows(p, seq_len=16, horizon=4, target="activation_direction")
    assert volume.shape == direction.shape
    assert volume_stats.target_name == "signed_activation_volume_mwh"
    assert direction_stats.target_name == "activation_direction_label"


def test_model_forward_shape() -> None:
    m = PatchTransformerQuantile(n_features=1)
    x = torch.randn(4, 96, 1)
    out = m(x)
    assert out.shape == (4, 16, 3)


def test_quantile_loss_positive() -> None:
    pred = torch.zeros(2, 4, 3)
    y = torch.ones(2, 4)
    qs = (0.1, 0.5, 0.9)
    assert quantile_loss(pred, y, qs).item() > 0


def test_train_model_runs_end_to_end(tmp_path: Path) -> None:
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    _synth_panel(256).write_parquet(train_p)
    _synth_panel(256, seed=14).write_parquet(val_p)
    cfg = TrainConfig(
        name="f7-test",
        train_panel=train_p,
        val_panel=val_p,
        multivariate=False,
        epochs=1,
        batch_size=16,
        seed=13,
        out_dir=tmp_path / "models",
        device="cpu",
        experiment="heimdall-test",
    )
    res = train_model(cfg)
    assert "val_pinball_mean" in res
    assert res["ckpt"].exists()
    assert (tmp_path / "models" / "f7-test" / "seed-13" / "stats.pkl").exists()
    # Sanity: predictions in original units roughly within range
    val_preds = res["val_preds"]
    assert np.isfinite(val_preds).all()


def test_train_f5_tiny_smoke(tmp_path: Path) -> None:
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    _synth_panel(96).write_parquet(train_p)
    _synth_panel(96, seed=14).write_parquet(val_p)
    cfg = NeuralProcessConfig(
        name="f5",
        train_panel=train_p,
        val_panel=val_p,
        seq_len=24,
        horizon=4,
        d_model=16,
        n_layers=1,
        epochs=1,
        batch_size=8,
        seed=13,
        out_dir=tmp_path / "models",
        device="cpu",
        experiment="heimdall-test",
    )
    res = train_neural_process(cfg)
    assert res["ckpt"].exists()
    z = np.load(tmp_path / "models" / "f5" / "seed-13" / "val_preds.npz")
    assert z["preds"].shape[-1] == 3
    assert np.all(np.diff(z["preds"], axis=-1) >= -1e-6)


def test_train_activation_direction_tiny_smoke(tmp_path: Path) -> None:
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    _synth_panel(128).write_parquet(train_p)
    _synth_panel(128, seed=14).write_parquet(val_p)
    cfg = ActivationDirectionConfig(
        name="activation-direction-test",
        train_panel=train_p,
        val_panel=val_p,
        feature_names=F8_FEATURES,
        seq_len=24,
        horizon=4,
        d_model=16,
        nhead=4,
        n_layers=1,
        epochs=1,
        batch_size=8,
        seed=13,
        out_dir=tmp_path / "models",
        device="cpu",
        experiment="heimdall-test",
    )
    res = train_activation_direction(cfg)
    assert res["ckpt"].exists()
    z = np.load(tmp_path / "models" / "activation-direction-test" / "seed-13" / "val_direction_probs.npz")
    assert z["probs"].shape[-1] == 3
    assert np.allclose(z["probs"].sum(axis=-1), 1.0, atol=1e-5)


def test_quantile_dataset_as_tensor() -> None:
    X = np.zeros((4, 96, 1), dtype=np.float32)
    Y = np.zeros((4, 16), dtype=np.float32)
    ds = QuantilePanelDataset(X, Y)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (96, 1)
    assert y.shape == (16,)
