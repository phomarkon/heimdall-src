"""LGBM-gate × PatchTST-magnitude ensemble for activation hurdle.

Stage A (event probability) = LGBM classifier (better AUC + Brier).
Stage B (conditional magnitude q10/q50/q90) = PatchTST quantile head.

Joint pinball = P_lgbm · pinball(q50_patchtst, y) + (1 - P_lgbm) · pinball(0, y)

Re-uses already-trained models from models/forecaster/hurdle/{dir}_seed-*/
and models/forecaster/hurdle_patchtst/{dir}_seed-*/. No retraining.

Writes outputs/hurdle_ensemble/leaderboard.json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/hurdle_ensemble"
LGBM_DIR = REPO / "models/forecaster/hurdle"
PTST_DIR = REPO / "models/forecaster/hurdle_patchtst"
SEEDS = (13, 42, 137, 1729, 31415)
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 96
HORIZON = 16
PATCH_LEN = 8
D_MODEL = 128
NHEAD = 8
N_LAYERS = 6
DROPOUT = 0.1
BATCH = 256
THRESH = 0.0


class PatchTSTHurdle(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.rin_gamma = nn.Parameter(torch.ones(n_features))
        self.rin_beta = nn.Parameter(torch.zeros(n_features))
        self.patch = nn.Conv1d(n_features, D_MODEL, kernel_size=PATCH_LEN, stride=PATCH_LEN)
        n_patches = SEQ_LEN // PATCH_LEN
        self.pos = nn.Parameter(torch.zeros(1, n_patches, D_MODEL))
        enc = nn.TransformerEncoderLayer(D_MODEL, NHEAD, 4 * D_MODEL, DROPOUT,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        flat = D_MODEL * n_patches
        self.head_cls = nn.Linear(flat, HORIZON)
        self.head_reg = nn.Linear(flat, HORIZON * len(QUANTILES))

    def forward(self, x):
        m = x.mean(1, keepdim=True); s = x.std(1, keepdim=True).clamp(min=1e-5)
        x = (x - m) / s * self.rin_gamma + self.rin_beta
        h = self.patch(x.transpose(1, 2)).transpose(1, 2) + self.pos
        h = self.norm(self.encoder(h))
        flat = h.reshape(h.shape[0], -1)
        logits = self.head_cls(flat)
        qout = self.head_reg(flat).reshape(-1, HORIZON, len(QUANTILES))
        return logits, qout


def _windows(panel_path, anom_path, *, feature_names, target_col):
    df = pl.read_parquet(panel_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    if anom_path.exists():
        anom = pl.read_parquet(anom_path).with_columns(
            pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
        )
        df = df.join(anom, on="timestamp_utc", how="left").fill_null(0.0)
    have = [c for c in feature_names if c in df.columns]
    X = df.select(have).to_numpy().astype(np.float32)
    y = df.select(target_col).to_numpy().astype(np.float32).ravel()
    n = len(y) - SEQ_LEN - HORIZON + 1
    Xw = np.zeros((n, SEQ_LEN, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, HORIZON), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i + SEQ_LEN]
        Yw[i] = y[i + SEQ_LEN:i + SEQ_LEN + HORIZON]
    return Xw, Yw


def _lgbm_p(direction: str, seed: int, X) -> np.ndarray:
    sd = LGBM_DIR / f"{direction}_seed-{seed}"
    a = pickle.load(open(sd / "stage_a.pkl", "rb"))
    n, sl, f = X.shape
    Xf = X.reshape(n, sl * f)
    P = np.zeros((n, HORIZON), dtype=np.float32)
    for h in range(HORIZON):
        k = f"h{h}"
        if k not in a:
            continue
        clf = lgb.Booster(model_str=a[k])
        P[:, h] = clf.predict(Xf).astype(np.float32)
    return P


def _ptst_q50(direction: str, seed: int, X, train_X) -> np.ndarray:
    sd = PTST_DIR / f"{direction}_seed-{seed}"
    meta = json.loads((sd / "metrics.json").read_text())
    ymu, ysd = meta["ymu"], meta["ysd"]
    mu = train_X.reshape(-1, train_X.shape[-1]).mean(0)
    s = train_X.reshape(-1, train_X.shape[-1]).std(0).clip(min=1e-5)
    Xn = (X - mu) / s
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PatchTSTHurdle(n_features=X.shape[-1]).to(device).eval()
    model.load_state_dict(torch.load(sd / "model.pt", map_location=device, weights_only=True))
    q50 = np.zeros((X.shape[0], HORIZON), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(Xn[i:i + BATCH]).to(device)
            _, qout = model(xb)
            q50[i:i + BATCH] = (qout[..., 1].cpu().numpy() * ysd + ymu).astype(np.float32)
    return q50


def _joint(p_event, q50, y):
    err = y - q50
    pin_pos = np.maximum(0.5 * err, -0.5 * err)
    pin_zero = np.maximum(0.5 * y, -0.5 * y)
    return float(np.mean(p_event * pin_pos + (1 - p_event) * pin_zero))


def main() -> int:
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES
    OUT.mkdir(parents=True, exist_ok=True)
    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    test_panel = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
    anom_train = REPO / "data/processed/anomaly_features_train.parquet"
    anom_test = REPO / "data/processed/anomaly_features_test.parquet"

    rows = []
    for direction in ("up", "down"):
        target_col = f"mfrr_{direction}_volume_mw"
        print(f"[ensemble] windows dir={direction}", flush=True)
        X_tr, _ = _windows(train_panel, anom_train, feature_names=F_CANONICAL_FEATURES, target_col=target_col)
        X_te, Y_te = _windows(test_panel, anom_test, feature_names=F_CANONICAL_FEATURES, target_col=target_col)
        for seed in SEEDS:
            P = _lgbm_p(direction, seed, X_te)
            Q = _ptst_q50(direction, seed, X_te, X_tr)
            joint = _joint(P, Q, Y_te)
            rows.append({"direction": direction, "seed": seed, "test_joint_pinball": joint})
            print(f"  seed={seed} dir={direction} joint={joint:.3f}", flush=True)

    lb = []
    for d in ("up", "down"):
        sub = [r["test_joint_pinball"] for r in rows if r["direction"] == d]
        lb.append({
            "direction": d, "n_seeds": len(sub),
            "test_joint_pinball_mean": float(np.mean(sub)),
            "test_joint_pinball_std": float(np.std(sub)),
        })
    (OUT / "leaderboard.json").write_text(json.dumps(lb, indent=2))
    (OUT / "per_seed.json").write_text(json.dumps(rows, indent=2))
    print(f"[ensemble] wrote {OUT}/leaderboard.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
