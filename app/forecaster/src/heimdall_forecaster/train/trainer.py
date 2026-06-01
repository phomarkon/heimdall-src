"""F7/F8 training loop. Per docs/RESEARCH-PROPOSAL.md §4.4 + §5.7 (fair-comparison).

Uses ``packages/ml/seeds.py`` to fix all seeds and ``packages/ml/tracking.py``
to log every metric, hyperparam, and final pinball loss to MLflow.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from heimdall_forecaster.train.dataset import (
    QUANTILES,
    QuantilePanelDataset,
    WindowStats,
    make_windows,
)
from heimdall_forecaster.train._utils import pinball_loss, resolve_device
from heimdall_forecaster.train.model import PatchTransformerQuantile, quantile_loss
from heimdall_ml import seeds, tracking


@dataclass
class TrainConfig:
    """Hyperparameters frozen at config-time. See ``train/configs/{f7,f8,f8b,f8c}.yaml``."""

    name: str = "f7"
    train_panel: Path = Path("data/processed/dk1_panel_train.parquet")
    val_panel: Path = Path("data/processed/dk1_panel_val.parquet")
    multivariate: bool = False  # F8a → True (legacy switch; ignored when feature_names is set)
    feature_names: tuple[str, ...] | None = None  # F8b/c/d use explicit set
    target: str = "price"
    target_column: str | None = None
    anomaly_panel: Path | None = None  # F8c/d join anomaly features
    seq_len: int = 96
    horizon: int = 16
    patch_len: int = 8
    d_model: int = 128
    nhead: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = Path("models/forecaster")
    device: str = "auto"
    experiment: str = "heimdall-forecaster-train"
    use_rin: bool = False  # Reversible Instance Norm; required for F8b/c/d/e/F13.



def train_model(cfg: TrainConfig) -> dict[str, object]:
    """Train one F7/F8 model end-to-end and return metrics + checkpoint path."""
    seeds.seed_everything(cfg.seed)

    # --- data ---------------------------------------------------------
    X_tr, Y_tr, stats = make_windows(
        cfg.train_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        multivariate=cfg.multivariate,
        feature_names=cfg.feature_names,
        target=cfg.target,
        target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
    )
    X_va, Y_va, _ = make_windows(
        cfg.val_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        multivariate=cfg.multivariate,
        feature_names=cfg.feature_names,
        target=cfg.target,
        target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
        stats=stats,  # apply training stats to val (no leakage)
    )
    train_loader = DataLoader(
        QuantilePanelDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        QuantilePanelDataset(X_va, Y_va), batch_size=cfg.batch_size, shuffle=False
    )

    # --- model + opt --------------------------------------------------
    device = resolve_device(cfg.device)
    # Auto-enable RIN when input is high-dim multivariate (>5 features); without
    # RIN, patchTST collapses to constant on this task (verified 2026-05-17).
    use_rin = cfg.use_rin or (X_tr.shape[-1] > 5)
    model = PatchTransformerQuantile(
        n_features=X_tr.shape[-1],
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        n_quantiles=len(cfg.quantiles),
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        use_rin=use_rin,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- MLflow run ---------------------------------------------------
    tracking.init(experiment=cfg.experiment)
    params = {
        "name": cfg.name,
        "multivariate": cfg.multivariate,
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "patch_len": cfg.patch_len,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "target": cfg.target,
        "target_column": cfg.target_column or "",
        "target_name": stats.target_name,
        "n_features": X_tr.shape[-1],
        "n_train_windows": X_tr.shape[0],
        "n_val_windows": X_va.shape[0],
    }
    out_dir = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_preds_final: np.ndarray | None = None

    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", params=params):
        with tracking.track_experiment_compute(f"{cfg.name}.train"):
            for epoch in range(cfg.epochs):
                model.train()
                running = 0.0
                n = 0
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    optim.zero_grad()
                    pred = model(x)
                    loss = quantile_loss(pred, y, cfg.quantiles)
                    loss.backward()
                    optim.step()
                    running += loss.item() * x.shape[0]
                    n += x.shape[0]
                train_loss = running / max(n, 1)

                # --- val pass --------------------------------------------
                model.eval()
                v_running = 0.0
                v_n = 0
                preds_chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        pred = model(x)
                        v_running += quantile_loss(pred, y, cfg.quantiles).item() * x.shape[0]
                        v_n += x.shape[0]
                        preds_chunks.append(pred.cpu().numpy())
                val_loss = v_running / max(v_n, 1)
                val_preds = np.concatenate(preds_chunks, axis=0)
                tracking.log_metrics(
                    {"train_pinball_mean": train_loss, "val_pinball_mean": val_loss},
                    step=epoch,
                )
                val_preds_final = val_preds

        # De-normalise predictions and targets to original units (DKK/MWh).
        assert val_preds_final is not None
        val_preds_dn = stats.denormalise_target(val_preds_final)
        Y_va_dn = stats.denormalise_target(Y_va)

        per_q = {
            f"val_pinball_q{int(qq*100)}": pinball_loss(
                Y_va_dn, val_preds_dn[:, :, qi], qq
            )
            for qi, qq in enumerate(cfg.quantiles)
        }
        per_q["val_pinball_mean_dkk"] = float(np.mean(list(per_q.values())))
        tracking.log_metrics(per_q)

        # Empirical [q10, q90] coverage on the val set (sorted predictions).
        sorted_pred = np.sort(val_preds_dn, axis=-1)
        lo, hi = sorted_pred[:, :, 0], sorted_pred[:, :, -1]
        coverage = float(np.mean((Y_va_dn >= lo) & (Y_va_dn <= hi)))
        tracking.log_metrics({"val_q10_q90_coverage": coverage})

        # --- persistence -----------------------------------------
        ckpt_path = out_dir / "model.pt"
        torch.save(model.state_dict(), ckpt_path)
        with open(out_dir / "stats.pkl", "wb") as fh:
            pickle.dump(stats, fh)
        with open(out_dir / "config.json", "w") as fh:
            json.dump({**params, "out_dir": str(out_dir)}, fh, indent=2)
        # Save val predictions in *original units* for ACI/CP wrappers downstream.
        np.savez(out_dir / "val_preds.npz", preds=val_preds_dn, targets=Y_va_dn)

    return {
        "val_pinball_mean": per_q["val_pinball_mean_dkk"],
        "val_q10_q90_coverage": coverage,
        "per_quantile": per_q,
        "ckpt": ckpt_path,
        "stats_path": out_dir / "stats.pkl",
        "val_preds": val_preds_dn,
        "val_targets": Y_va_dn,
    }


__all__ = ["TrainConfig", "train_model"]
