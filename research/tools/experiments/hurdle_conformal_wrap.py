"""Split-CP + online ACI wrap on the hurdle stage-B magnitude head.

DEPRECATED ACI PATH. The ``_aci_walkforward`` helper below is a one-off,
non-canonical width recursion that drives the interval to zero on every hit,
so its ``coverage_summary.json`` reports badly under-covered ACI (~0.51-0.65).
It is NOT the production calibrator. The reported online-ACI coverage uses
``heimdall_ml.conformal.aci.AdaptiveConformalInference`` (Gibbs-Candes), which
on the cached test predictions holds ~0.90 at the 0.90 target; reproduce with
``research/ml/experiments/test_set_aci_coverage.py``. Kept only for the
split-CP hurdle numbers; do not cite the ACI column from this file.

Stage B predicts a conditional quantile of the volume given an event. We
calibrate a finite-sample conformal interval on the val event-only subset
(split-CP, Theorem 1a) and an online ACI interval on the rolling val/test
stream (Theorem 1b). Coverage reported for alpha in {0.05, 0.10, 0.20}.

Operates on already-trained models in models/forecaster/hurdle_patchtst/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/hurdle_conformal"
MODEL_DIR = REPO / "models/forecaster/hurdle_patchtst"
SEEDS = (13, 42, 137, 1729, 31415)
QUANTILES = (0.1, 0.5, 0.9)
ALPHAS = (0.05, 0.10, 0.20)
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


def _predict(seed: int, direction: str, X, train_X) -> tuple[np.ndarray, np.ndarray]:
    sd = MODEL_DIR / f"{direction}_seed-{seed}"
    meta = json.loads((sd / "metrics.json").read_text())
    ymu, ysd = meta["ymu"], meta["ysd"]
    mu = train_X.reshape(-1, train_X.shape[-1]).mean(0)
    s = train_X.reshape(-1, train_X.shape[-1]).std(0).clip(min=1e-5)
    Xn = (X - mu) / s
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PatchTSTHurdle(n_features=X.shape[-1]).to(device).eval()
    model.load_state_dict(torch.load(sd / "model.pt", map_location=device, weights_only=True))
    P = np.zeros((X.shape[0], HORIZON), dtype=np.float32)
    Q = np.zeros((X.shape[0], HORIZON, len(QUANTILES)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(Xn[i:i + BATCH]).to(device)
            logits, qout = model(xb)
            P[i:i + BATCH] = torch.sigmoid(logits).cpu().numpy()
            qout_real = qout * ysd + ymu
            Q[i:i + BATCH] = qout_real.cpu().numpy()
    return P, Q


def _split_cp_coverage(q_va: np.ndarray, y_va: np.ndarray, mask_va: np.ndarray,
                       q_te: np.ndarray, y_te: np.ndarray, mask_te: np.ndarray,
                       alpha: float) -> dict:
    """CQR-style nonconformity on the event subset of val; evaluate coverage on test event subset.

    Returns dict with empirical coverage and mean interval width.
    """
    # CQR score: max(q_lo - y, y - q_hi)
    q_lo_va, q_hi_va = q_va[..., 0], q_va[..., -1]
    s_va = np.maximum(q_lo_va - y_va, y_va - q_hi_va)
    s_va = s_va[mask_va].ravel()
    if len(s_va) == 0:
        return {"alpha": alpha, "coverage": float("nan"), "width": float("nan"), "n_cal": 0}
    n = len(s_va)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.sort(s_va)[k - 1])
    # Apply to test: interval = [q_lo - qhat, q_hi + qhat]
    q_lo_te, q_hi_te = q_te[..., 0], q_te[..., -1]
    lo = q_lo_te - qhat; hi = q_hi_te + qhat
    cov = ((y_te >= lo) & (y_te <= hi))
    cov_event = float(cov[mask_te].mean()) if mask_te.any() else float("nan")
    width_event = float((hi - lo)[mask_te].mean()) if mask_te.any() else float("nan")
    return {"alpha": alpha, "coverage": cov_event, "width": width_event, "n_cal": n}


def _aci_walkforward(q_te: np.ndarray, y_te: np.ndarray, mask_te: np.ndarray,
                     alpha: float, gamma: float = 0.05) -> dict:
    """Online ACI on the test stream, event-only positions."""
    q_lo, q_hi = q_te[..., 0], q_te[..., -1]
    # Flatten over (N, H)
    y_f = y_te.reshape(-1); lo_f = q_lo.reshape(-1); hi_f = q_hi.reshape(-1); m_f = mask_te.reshape(-1)
    # Only update on event slots
    a_t = alpha
    covers = []
    widths = []
    qhat = 0.0
    for t in range(len(y_f)):
        if not m_f[t]:
            continue
        lo_t = lo_f[t] - qhat; hi_t = hi_f[t] + qhat
        ok = (y_f[t] >= lo_t) and (y_f[t] <= hi_t)
        covers.append(int(ok))
        widths.append(float(hi_t - lo_t))
        # Update qhat via ACI (use score magnitude proxy)
        err_t = 0.0 if ok else 1.0
        a_t = float(np.clip(a_t + gamma * (alpha - err_t), 1e-3, 1 - 1e-3))
        # Adjust qhat to track current alpha (simple linear bridge over score range)
        s_t = max(lo_t - y_f[t], y_f[t] - hi_t, 0.0)
        qhat = (1 - gamma) * qhat + gamma * (s_t if not ok else 0.0)
    return {"alpha": alpha, "gamma": gamma,
            "coverage": float(np.mean(covers)) if covers else float("nan"),
            "width": float(np.mean(widths)) if widths else float("nan"),
            "n_events_seen": int(sum(covers and [1]*len(covers) or []))}


def main() -> int:
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES
    OUT.mkdir(parents=True, exist_ok=True)
    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    test_panel = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
    anom_train = REPO / "data/processed/anomaly_features_train.parquet"
    anom_val = REPO / "data/processed/anomaly_features_val.parquet"
    anom_test = REPO / "data/processed/anomaly_features_test.parquet"

    rows = []
    for direction in ("up", "down"):
        target_col = f"mfrr_{direction}_volume_mw"
        print(f"[hurdle-conformal] dir={direction} windows…", flush=True)
        X_tr, _ = _windows(train_panel, anom_train, feature_names=F_CANONICAL_FEATURES, target_col=target_col)
        X_va, Y_va = _windows(val_panel, anom_val, feature_names=F_CANONICAL_FEATURES, target_col=target_col)
        X_te, Y_te = _windows(test_panel, anom_test, feature_names=F_CANONICAL_FEATURES, target_col=target_col)
        for seed in SEEDS:
            _, Q_va = _predict(seed, direction, X_va, X_tr)
            _, Q_te = _predict(seed, direction, X_te, X_tr)
            mask_va = Y_va > THRESH
            mask_te = Y_te > THRESH
            row = {"direction": direction, "seed": seed,
                   "n_event_val": int(mask_va.sum()), "n_event_test": int(mask_te.sum())}
            for a in ALPHAS:
                row[f"split_cp_alpha_{a:.2f}"] = _split_cp_coverage(
                    Q_va, Y_va, mask_va, Q_te, Y_te, mask_te, a)
                row[f"aci_alpha_{a:.2f}"] = _aci_walkforward(Q_te, Y_te, mask_te, a)
            rows.append(row)
            print(f"  seed={seed} dir={direction} done", flush=True)

    # Aggregate
    summary = []
    for direction in ("up", "down"):
        for a in ALPHAS:
            sub = [r for r in rows if r["direction"] == direction]
            cps = [r[f"split_cp_alpha_{a:.2f}"]["coverage"] for r in sub]
            cpw = [r[f"split_cp_alpha_{a:.2f}"]["width"] for r in sub]
            acic = [r[f"aci_alpha_{a:.2f}"]["coverage"] for r in sub]
            aciw = [r[f"aci_alpha_{a:.2f}"]["width"] for r in sub]
            summary.append({
                "direction": direction, "alpha": a,
                "target_coverage": 1 - a,
                "split_cp_coverage_mean": float(np.nanmean(cps)),
                "split_cp_width_mean": float(np.nanmean(cpw)),
                "aci_coverage_mean": float(np.nanmean(acic)),
                "aci_width_mean": float(np.nanmean(aciw)),
            })
    (OUT / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "per_seed.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"[hurdle-conformal] wrote {OUT}/coverage_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
