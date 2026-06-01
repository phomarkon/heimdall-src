"""Multi-zone ACI panel — Nature MI Phase 1.5.

Train F7 on each zone (DK1, DE, NO2, SE3) using the Trinity multi-zone
DA panel, then evaluate ACI coverage per zone. Headline claim:
**ACI converges to nominal (1 - α = 0.90) across 4 distinct Nordic /
CWE market designs**, validating Theorem 1b's market-design-agnostic
guarantee.

This is the multi-zone empirical bedrock for the Nature MI submission.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from heimdall_forecaster.train.dataset import (
    QUANTILES, HORIZON, SEQ_LEN, QuantilePanelDataset, WindowStats,
)
from heimdall_forecaster.train.model import PatchTransformerQuantile, quantile_loss
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "data/processed/multi_zone_da_panel.parquet"
OUT_ROOT = REPO / "models/forecaster"
ZONES = ("DK1", "DE", "NO2", "SE3")
SEEDS = (42,)  # Phase 1.5 = single-seed proof; scale to 5 once result is positive.
EPOCHS = 5
BATCH = 64
LR = 1e-3
WD = 1e-4
PATCH_LEN = 8
D_MODEL = 128
NHEAD = 8
N_LAYERS = 6


def _windows_for_zone(df: pl.DataFrame, zone: str, seq_len: int, horizon: int,
                     val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, WindowStats]:
    zdf = df.filter(pl.col("zone") == zone).sort("utc_timestamp").drop_nulls()
    series = zdf["price_eur_mwh"].to_numpy().astype(np.float64)
    n = len(series) - seq_len - horizon
    if n <= 0:
        raise ValueError(f"not enough data for {zone}")
    # Sliding windows; chronological train/val split.
    X = np.empty((n, seq_len, 1), dtype=np.float32)
    Y = np.empty((n, horizon), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = series[i:i + seq_len]
        Y[i] = series[i + seq_len:i + seq_len + horizon]
    val_split = int(n * (1 - val_frac))
    Xtr, Ytr = X[:val_split], Y[:val_split]
    Xva, Yva = X[val_split:], Y[val_split:]
    # Per-feature normalization using TRAIN stats (no leakage).
    mean = Xtr.mean(axis=(0, 1)); std = Xtr.std(axis=(0, 1))
    std = np.where(std == 0.0, 1.0, std)
    tgt_mean = float(Ytr.mean()); tgt_std = float(Ytr.std() or 1.0)
    stats = WindowStats(mean=mean, std=std, feature_names=("price",),
                        target_mean=tgt_mean, target_std=tgt_std)
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)
    Ytr = ((Ytr - tgt_mean) / tgt_std).astype(np.float32)
    Yva_norm = ((Yva - tgt_mean) / tgt_std).astype(np.float32)
    Yva_dn = Yva.astype(np.float32)  # keep denormalized for reporting
    return Xtr, Ytr, Xva, Yva_dn, stats


def train_zone(zone: str, seed: int) -> dict:
    seeds.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pl.read_parquet(PANEL)
    Xtr, Ytr, Xva, Yva_dn, stats = _windows_for_zone(df, zone, SEQ_LEN, HORIZON)
    print(f"[multizone {zone} seed={seed}] train={Xtr.shape[0]} val={Xva.shape[0]}")

    train_ds = QuantilePanelDataset(Xtr, Ytr)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    model = PatchTransformerQuantile(
        n_features=1, seq_len=SEQ_LEN, horizon=HORIZON, n_quantiles=len(QUANTILES),
        patch_len=PATCH_LEN, d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); ep = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            loss = quantile_loss(pred, y, QUANTILES)
            opt.zero_grad(); loss.backward(); opt.step()
            ep += loss.item() * y.shape[0]
        print(f"  [{zone} seed={seed}] epoch {epoch+1}/{EPOCHS} loss={ep/len(train_ds):.4f}")

    model.eval()
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_norm = model.predict_quantiles(Xva_t).cpu().numpy()  # (N, H, Q) normalized
    preds_dn = preds_norm * stats.target_std + stats.target_mean
    targets_dn = Yva_dn

    out_dir = OUT_ROOT / f"f7_zone_{zone.lower()}" / f"seed-{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "val_preds.npz", preds=preds_dn, targets=targets_dn)
    aci = aci_coverage_from_val(out_dir / "val_preds.npz", alpha=0.1, gamma=0.05)

    per_q = {}
    for qi, q in enumerate(QUANTILES):
        err = targets_dn[..., None] - preds_dn[..., qi:qi+1]
        per_q[f"val_pinball_q{int(q*100)}"] = float(np.mean(np.maximum(q * err, (q - 1.0) * err)))
    pinball_mean = float(np.mean(list(per_q.values())))
    sorted_p = np.sort(preds_dn, axis=-1)
    raw_cov = float(np.mean((targets_dn >= sorted_p[..., 0]) & (targets_dn <= sorted_p[..., -1])))

    metrics = {
        "zone": zone, "seed": seed, **per_q,
        "val_pinball_mean": pinball_mean,
        "val_q10_q90_coverage": raw_cov,
        "aci_alpha_target": aci.alpha_target,
        "aci_empirical_coverage": aci.empirical_coverage,
        "aci_mean_width": aci.mean_width,
        "n_val_windows": int(Xva.shape[0]),
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  [{zone} seed={seed}] pinball={pinball_mean:.1f}  raw_cov={raw_cov:.3f}  "
          f"ACI={aci.empirical_coverage:.3f}")
    return metrics


def main() -> int:
    panel_rows = []
    for zone in ZONES:
        for seed in SEEDS:
            m = train_zone(zone, seed)
            panel_rows.append(m)
    out = REPO / "experiments/outputs/multizone_aci_panel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": panel_rows}, indent=2))
    print("\n===== Multi-Zone ACI Panel =====")
    print(f"{'zone':6s} pinball  raw_cov  ACI_cov")
    for r in panel_rows:
        print(f"{r['zone']:6s} {r['val_pinball_mean']:7.2f}  {r['val_q10_q90_coverage']:.3f}    {r['aci_empirical_coverage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
