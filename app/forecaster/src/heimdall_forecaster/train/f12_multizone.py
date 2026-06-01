"""F12 — multi-zone transfer-learning forecaster (Plan v2 Track C).

A PatchTST with a learned zone-embedding token. Pretrained on the multi-zone
DA-price series from Trinity (DE / SE3 / NO2 / DK1), then fine-tuned on the
DK1 imbalance rich panel.

The novelty: DE / SE3 / NO2 did not undergo the 2025-03-04 mFRR EAM regime
change. They expose the model to weather-coupled price dynamics under a
*stable* regime, providing a prior the DK1-only model can't learn.

Architecture
------------
- Same PatchEmbedding + TransformerEncoder as F7/F8.
- Prepended ``[zone] token`` learned per zone (4 → d_model).
- Quantile head identical.

Training stages
---------------
1. **Pretrain** on ``multi_zone_da_panel`` produced by
   ``tools/build_multi_zone_panel.py`` (q50 ≈ Huber loss on raw zone DA prices,
   subscale by zone-mean to keep loss balanced).
2. **Fine-tune** last 2 transformer layers + quantile head on DK1 rich panel
   with the existing pinball loss. Frozen embeddings of DK1 zone retained.

The trainer is a thin variant of ``trainer.train_model`` — kept separate so the
F8a/b/c lineage remains untouched.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from heimdall_ml import seeds, tracking
from torch import nn
from torch.utils.data import DataLoader, Dataset

from heimdall_forecaster.train._utils import pinball_loss
from heimdall_forecaster.train.dataset import (
    QUANTILES,
    QuantilePanelDataset,
    WindowStats,
    make_windows,
)
from heimdall_forecaster.train.model import PatchEmbedding, quantile_loss

ZONES = ("DK1", "DE", "SE3", "NO2")


@dataclass
class F12Config:
    name: str = "f12"
    train_panel: Path = Path("data/processed/dk1_panel_rich_train.parquet")
    val_panel: Path = Path("data/processed/dk1_panel_rich_val.parquet")
    anomaly_panel: Path | None = None
    multi_zone_panel: Path = Path("data/processed/multi_zone_da_panel.parquet")
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    seq_len: int = 96
    horizon: int = 16
    patch_len: int = 8
    d_model: int = 128
    nhead: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    pretrain_epochs: int = 5
    finetune_epochs: int = 5
    batch_size: int = 64
    lr_pre: float = 1e-3
    lr_fine: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = Path("models/forecaster/f12")
    device: str = "auto"
    experiment: str = "heimdall-forecaster-train"


class ZoneEmbeddingPatchTransformer(nn.Module):
    """PatchTST with a learnable [zone] token prepended to each sequence."""

    def __init__(self, *, n_features: int, n_zones: int = 4, seq_len: int = 96,
                 horizon: int = 16, n_quantiles: int = 3, patch_len: int = 8,
                 d_model: int = 128, nhead: int = 8, n_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.patch = PatchEmbedding(n_features, patch_len, d_model)
        n_patches = seq_len // patch_len
        self.zone_emb = nn.Embedding(n_zones, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))  # +1 for [zone]
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * (n_patches + 1), horizon * n_quantiles)

    def forward(self, x: torch.Tensor, zone_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F); zone_id: (B,)
        z = self.zone_emb(zone_id).unsqueeze(1)  # (B, 1, d_model)
        p = self.patch(x)  # (B, n_patches, d_model)
        h = torch.cat([z, p], dim=1) + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        b = h.shape[0]
        out = self.head(h.reshape(b, -1))
        return out.reshape(b, self.horizon, self.n_quantiles)


class MultiZoneDataset(Dataset):
    """Long-form (timestamp, zone, price) windows with zone_id labels."""

    def __init__(self, panel_path: Path, seq_len: int, horizon: int, stats: WindowStats):
        df = pl.read_parquet(panel_path).sort(["zone", "utc_timestamp"])
        zones = list(ZONES)
        X_list = []
        Y_list = []
        z_list = []
        for zi, zname in enumerate(zones):
            zdf = df.filter(pl.col("zone") == zname)
            if zdf.height < seq_len + horizon:
                continue
            arr = zdf.select(["price_eur_mwh"]).to_numpy().astype(np.float64)
            arr_norm = stats.normalise(arr)
            for i in range(arr_norm.shape[0] - seq_len - horizon + 1):
                X_list.append(arr_norm[i : i + seq_len].astype(np.float32))
                Y_list.append(arr_norm[i + seq_len : i + seq_len + horizon, 0].astype(np.float32))
                z_list.append(zi)
        self.X = torch.tensor(np.stack(X_list)) if X_list else torch.empty(0)
        self.Y = torch.tensor(np.stack(Y_list)) if Y_list else torch.empty(0)
        self.Z = torch.tensor(z_list, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


def train_f12(cfg: F12Config) -> dict[str, object]:
    """Two-stage train: multi-zone pretrain → DK1 fine-tune."""
    seeds.seed_everything(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else "cpu")

    # --- Pretrain on multi-zone DA panel -------------------------------------
    # Compute global stats on the multi-zone panel for normalisation.
    mzpanel = pl.read_parquet(cfg.multi_zone_panel)
    prices = mzpanel["price_eur_mwh"].to_numpy().astype(np.float64)
    pre_stats = WindowStats(
        mean=np.array([prices.mean()]),
        std=np.array([prices.std() + 1e-8]),
        feature_names=("price_eur_mwh",),
        target_mean=float(prices.mean()),
        target_std=float(prices.std() + 1e-8),
    )
    mz_train = MultiZoneDataset(cfg.multi_zone_panel, cfg.seq_len, cfg.horizon, pre_stats)
    mz_loader = DataLoader(mz_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = ZoneEmbeddingPatchTransformer(
        n_features=1, n_zones=len(ZONES), seq_len=cfg.seq_len, horizon=cfg.horizon,
        n_quantiles=len(cfg.quantiles), patch_len=cfg.patch_len,
        d_model=cfg.d_model, nhead=cfg.nhead, n_layers=cfg.n_layers, dropout=cfg.dropout,
    ).to(device)
    optim_pre = torch.optim.AdamW(model.parameters(), lr=cfg.lr_pre, weight_decay=cfg.weight_decay)
    params = {k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool))}
    with tracking.run(name=f"f12-seed{cfg.seed}", experiment=cfg.experiment, params=params):
        model.train()
        for epoch in range(cfg.pretrain_epochs):
            total = 0.0
            for X, Y, Z in mz_loader:
                X, Y, Z = X.to(device), Y.to(device), Z.to(device)
                optim_pre.zero_grad()
                pred = model(X, Z)
                loss = quantile_loss(pred, Y, cfg.quantiles)
                loss.backward()
                optim_pre.step()
                total += float(loss) * X.size(0)
            tracking.log_metrics({"pretrain_loss": total / max(len(mz_train), 1)}, step=epoch)

    # --- Fine-tune on DK1 imbalance -----------------------------------------
    X_tr, Y_tr, dk1_stats = make_windows(
        cfg.train_panel, seq_len=cfg.seq_len, horizon=cfg.horizon,
        feature_names=cfg.feature_names, anomaly_panel_path=cfg.anomaly_panel,
    )
    X_va, Y_va, _ = make_windows(
        cfg.val_panel, seq_len=cfg.seq_len, horizon=cfg.horizon,
        feature_names=cfg.feature_names, anomaly_panel_path=cfg.anomaly_panel,
        stats=dk1_stats,
    )

    # The DK1 feature vector has F > 1 — switch patch projection.
    n_features = X_tr.shape[-1]
    # Rebuild patch projection for the new feature dim, but keep encoder.
    model.patch = PatchEmbedding(n_features, cfg.patch_len, cfg.d_model).to(device)
    # Freeze encoder layers except the last two for fine-tune.
    for p in model.encoder.layers[:-2].parameters():
        p.requires_grad_(False)
    optim_fine = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_fine,
        weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(QuantilePanelDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(QuantilePanelDataset(X_va, Y_va), batch_size=cfg.batch_size, shuffle=False)
    z_dk1 = torch.full((cfg.batch_size,), ZONES.index("DK1"), dtype=torch.long, device=device)
    for epoch in range(cfg.finetune_epochs):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            z = z_dk1[: X.size(0)]
            optim_fine.zero_grad()
            pred = model(X, z)
            Y_norm = (Y - dk1_stats.target_mean) / dk1_stats.target_std
            loss = quantile_loss(pred, Y_norm, cfg.quantiles)
            loss.backward()
            optim_fine.step()

    # Eval
    model.eval()
    preds_va, ys_va = [], []
    with torch.no_grad():
        for X, Y in val_loader:
            X = X.to(device)
            z = z_dk1[: X.size(0)].to(device)
            preds_va.append(model(X, z).cpu().numpy())
            ys_va.append(Y.numpy())
    pred = np.concatenate(preds_va)
    y_val = np.concatenate(ys_va)
    pred_denorm = pred * dk1_stats.target_std + dk1_stats.target_mean
    y_val = y_val * dk1_stats.target_std + dk1_stats.target_mean

    metrics = {f"val_pinball_q{int(q * 100)}": pinball_loss(y_val, pred_denorm[..., i], q)
               for i, q in enumerate(cfg.quantiles)}
    metrics["val_pinball_mean_dkk"] = float(np.mean(list(metrics.values())))
    metrics["val_q10_q90_coverage"] = float(
        ((y_val >= pred_denorm[..., 0]) & (y_val <= pred_denorm[..., -1])).mean()
    )
    with tracking.run(name=f"f12-seed{cfg.seed}-eval", experiment=cfg.experiment, params={"seed": cfg.seed, "phase": "eval"}):
        tracking.log_metrics(metrics, step=cfg.finetune_epochs)

    out = cfg.out_dir / f"seed-{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out / "model.pt")
    with open(out / "stats.pkl", "wb") as f:
        pickle.dump(dk1_stats, f)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"ckpt": out / "model.pt", **metrics}


__all__ = ["F12Config", "ZoneEmbeddingPatchTransformer", "train_f12"]
