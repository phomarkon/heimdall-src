"""Baseline forecasters B1-B8. Per docs/RESEARCH-PROPOSAL.md §5.2.

We implement the non-LLM baselines:

- **B1 — RandomWalk**: y_hat[t+1..t+H] = y[t] (the last observed price). Quantile
  bands from the in-train residual distribution.
- **B2 — EWMA**: exponentially-weighted mean with halflife=16, Gaussian quantiles
  (the same baseline KE1 uses; reused here under the unified protocol).
- **B3 — SeasonalNaive**: y_hat[t+h] = y[t+h-96] (24-hour-ago lag). Quantile
  bands from the seasonal-residual distribution on train.
- **B4 — LightGBM-Quantile**: per-quantile LightGBM regressor with the seasonal
  feature stack (lag-1, lag-4, lag-96, hour-of-day, day-of-week).
- **B7 — N-BEATS-Lite**: 2-stack pure-MLP residual block; ~30 k params; quantile
  head q∈{0.1, 0.5, 0.9}. Trained on the same windows as F7.
- **B8 — PinballMean**: literal mean of the three quantile predictions across
  F7/F8/F0/F3 — a "free" ensemble baseline.

We *skip* B5 / B6, which the proposal frames as LLM-driven baselines (per the
sprint-day-3 owner split, the LLM agent stack is Tim's; not in this session).

Each baseline writes ``models/forecaster/<name>/seed-<n>/{val_preds.npz,metrics.json}``
and the leaderboard script picks them up.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm

from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    TARGET_COL,
    make_windows,
)
from heimdall_ml import seeds, tracking

REPO_ROOT = Path(__file__).resolve().parents[2]


def _series(p: Path) -> np.ndarray:
    df = pl.read_parquet(p).drop_nulls(TARGET_COL).sort("timestamp_utc")
    return df[TARGET_COL].to_numpy().astype(np.float64)


def _pinball_per_q(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _record(name: str, seed: int, preds: np.ndarray, targets: np.ndarray) -> dict:
    out = REPO_ROOT / "models/forecaster" / name / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "val_preds.npz", preds=preds, targets=targets)
    per_q = {}
    for qi, q in enumerate(QUANTILES):
        per_q[f"val_pinball_q{int(q*100)}"] = _pinball_per_q(targets, preds[..., qi], q)
    per_q["val_pinball_mean_dkk"] = float(np.mean(list(per_q.values())))
    sorted_p = np.sort(preds, axis=-1)
    coverage = float(
        np.mean((targets >= sorted_p[..., 0]) & (targets <= sorted_p[..., -1]))
    )
    metrics = {**per_q, "val_q10_q90_coverage": coverage}
    with open(out / "metrics.json", "w") as fh:
        json.dump({"seed": seed, **metrics}, fh, indent=2)
    return metrics


def _val_targets() -> tuple[np.ndarray, np.ndarray]:
    """Return (Y_va in DKK/MWh, len-N anchor index in val raw series)."""
    _, Y_va, stats = make_windows(
        REPO_ROOT / "data/processed/dk1_panel_val.parquet",
        seq_len=SEQ_LEN, horizon=HORIZON, multivariate=False,
    )
    Y_va_dn = stats.denormalise_target(Y_va)
    return Y_va_dn, stats


# ---- B1 RandomWalk ---------------------------------------------------------


def b1_random_walk(seed: int = 42) -> dict:
    seeds.seed_everything(seed)
    y_tr = _series(REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    y_va = _series(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    full = np.concatenate([y_tr, y_va])
    Y_va_dn, _ = _val_targets()
    n_windows = Y_va_dn.shape[0]
    # Anchor: last observed value at index `len(y_tr) + i - 1`.
    last = full[len(y_tr) - 1 + np.arange(n_windows)]
    point = np.broadcast_to(last[:, None], (n_windows, HORIZON)).copy()

    # Residual distribution = train one-step-ahead residuals.
    res = np.diff(y_tr)
    qs = np.quantile(res, QUANTILES)
    preds = np.empty((*point.shape, len(QUANTILES)), dtype=np.float64)
    for qi, off in enumerate(qs):
        preds[..., qi] = point + off

    return _record("b1_random_walk", seed, preds, Y_va_dn)


# ---- B2 EWMA ---------------------------------------------------------------


def b2_ewma(seed: int = 42, halflife: int = 16) -> dict:
    seeds.seed_everything(seed)
    y_tr = _series(REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    y_va = _series(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    Y_va_dn, _ = _val_targets()
    n_windows = Y_va_dn.shape[0]

    full = np.concatenate([y_tr, y_va])
    alpha = 1 - 0.5 ** (1 / halflife)
    mu = np.zeros_like(full)
    var = np.zeros_like(full)
    mu[0] = full[0]
    for t in range(1, full.size):
        mu[t] = alpha * full[t - 1] + (1 - alpha) * mu[t - 1]
        var[t] = alpha * (full[t - 1] - mu[t - 1]) ** 2 + (1 - alpha) * var[t - 1]
    sigma = np.sqrt(np.maximum(var, 1e-8))
    z = np.array([norm.ppf(q) for q in QUANTILES])

    preds = np.empty((n_windows, HORIZON, len(QUANTILES)), dtype=np.float64)
    for i in range(n_windows):
        anchor = len(y_tr) + i  # first horizon target
        # Persistence forecast: mean stays flat at mu[anchor-1] over the horizon.
        preds[i, :, :] = mu[anchor - 1] + z[None, :] * sigma[anchor - 1]

    return _record("b2_ewma", seed, preds, Y_va_dn)


# ---- B3 SeasonalNaive (96-step) -------------------------------------------


def b3_seasonal_naive(seed: int = 42) -> dict:
    seeds.seed_everything(seed)
    y_tr = _series(REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    y_va = _series(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    full = np.concatenate([y_tr, y_va])
    Y_va_dn, _ = _val_targets()
    n_windows = Y_va_dn.shape[0]

    point = np.empty((n_windows, HORIZON), dtype=np.float64)
    for i in range(n_windows):
        anchor = len(y_tr) + i  # first horizon target index in full
        # Seasonal-naive: y_hat[anchor+h] = full[anchor+h - 96]
        for h in range(HORIZON):
            point[i, h] = full[anchor + h - 96]

    # Residual distribution = train seasonal residuals.
    season_res = y_tr[96:] - y_tr[:-96]
    qs = np.quantile(season_res, QUANTILES)
    preds = np.empty((*point.shape, len(QUANTILES)), dtype=np.float64)
    for qi, off in enumerate(qs):
        preds[..., qi] = point + off

    return _record("b3_seasonal_naive", seed, preds, Y_va_dn)


# ---- B4 LightGBM-Quantile -------------------------------------------------


def _lgbm_features(y: np.ndarray) -> np.ndarray:
    """Build a (T, F) feature stack: lag-1, lag-4, lag-96, lag-672, sin/cos hour."""
    n = y.size
    feats = np.full((n, 6), np.nan, dtype=np.float64)
    feats[1:, 0] = y[:-1]      # lag 1 (15 min)
    feats[4:, 1] = y[:-4]      # lag 4 (1 h)
    feats[96:, 2] = y[:-96]    # lag 96 (24 h)
    feats[672:, 3] = y[:-672]  # lag 672 (7 d)
    hours = (np.arange(n) % 96) / 96.0  # quarter-hour-of-day
    feats[:, 4] = np.sin(2 * np.pi * hours)
    feats[:, 5] = np.cos(2 * np.pi * hours)
    return feats


def b4_lightgbm_quantile(seed: int = 42) -> dict:
    seeds.seed_everything(seed)
    y_tr = _series(REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    y_va = _series(REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    full = np.concatenate([y_tr, y_va])

    feats_full = _lgbm_features(full)
    target_full = full

    # Train on indices 672..len(y_tr) (drop NaN-padded leading rows).
    train_lo = 672
    train_hi = len(y_tr)
    Xtr = feats_full[train_lo:train_hi]
    ytr = target_full[train_lo:train_hi]

    # Per-quantile LightGBM model.
    models = {}
    for q in QUANTILES:
        mdl = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            random_state=seed,
            verbose=-1,
        )
        mdl.fit(Xtr, ytr)
        models[q] = mdl

    Y_va_dn, _ = _val_targets()
    n_windows = Y_va_dn.shape[0]

    # For multi-step we predict iteratively: substitute the previous prediction
    # into the next step's lag-1, but use ground-truth seasonal lags from
    # `full` (a fair proxy: at inference we *do* know the 24h-ago and 7d-ago
    # values — they're in the past).
    preds = np.empty((n_windows, HORIZON, len(QUANTILES)), dtype=np.float64)

    for i in range(n_windows):
        anchor = len(y_tr) + i  # first horizon target index in full
        last = full[anchor - 1]
        last_4 = full[anchor - 4] if anchor >= 4 else 0.0
        # Iterative loop per quantile.
        for qi, q in enumerate(QUANTILES):
            l1, l4 = last, last_4
            for h in range(HORIZON):
                idx = anchor + h
                feat = np.array(
                    [
                        l1,
                        full[idx - 4] if idx >= 4 else 0.0,
                        full[idx - 96] if idx >= 96 else 0.0,
                        full[idx - 672] if idx >= 672 else 0.0,
                        np.sin(2 * np.pi * (idx % 96) / 96.0),
                        np.cos(2 * np.pi * (idx % 96) / 96.0),
                    ],
                    dtype=np.float64,
                )
                yhat = models[q].predict(feat[None, :])[0]
                preds[i, h, qi] = yhat
                # roll forward
                l4 = full[idx - 3] if idx >= 3 else l1
                l1 = yhat

    return _record("b4_lightgbm_quantile", seed, preds, Y_va_dn)


# ---- B7 N-BEATS-Lite ------------------------------------------------------


def b7_nbeats_lite(seed: int = 42) -> dict:
    """Two-stack pure-MLP residual model with quantile heads."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from heimdall_forecaster.train.dataset import QuantilePanelDataset
    from heimdall_forecaster.train.model import quantile_loss

    seeds.seed_everything(seed)
    X_tr, Y_tr, stats = make_windows(
        REPO_ROOT / "data/processed/dk1_panel_train.parquet",
        seq_len=SEQ_LEN, horizon=HORIZON, multivariate=False,
    )
    X_va, Y_va, _ = make_windows(
        REPO_ROOT / "data/processed/dk1_panel_val.parquet",
        seq_len=SEQ_LEN, horizon=HORIZON, multivariate=False, stats=stats,
    )

    class NBeatsLite(nn.Module):
        def __init__(self):
            super().__init__()
            d = 128
            self.b1 = nn.Sequential(
                nn.Linear(SEQ_LEN, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()
            )
            self.b1_back = nn.Linear(d, SEQ_LEN)
            self.b1_fore = nn.Linear(d, HORIZON * 3)
            self.b2 = nn.Sequential(
                nn.Linear(SEQ_LEN, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()
            )
            self.b2_back = nn.Linear(d, SEQ_LEN)
            self.b2_fore = nn.Linear(d, HORIZON * 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xs = x.squeeze(-1)
            h = self.b1(xs)
            backc = self.b1_back(h)
            forec = self.b1_fore(h).reshape(-1, HORIZON, 3)
            r = xs - backc
            h2 = self.b2(r)
            forec2 = self.b2_fore(h2).reshape(-1, HORIZON, 3)
            return forec + forec2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NBeatsLite().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(QuantilePanelDataset(X_tr, Y_tr), batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(QuantilePanelDataset(X_va, Y_va), batch_size=64, shuffle=False)

    for _ in range(5):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = quantile_loss(pred, y, QUANTILES)
            loss.backward()
            opt.step()

    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _y in val_loader:
            chunks.append(model(x.to(device)).cpu().numpy())
    val_preds = np.concatenate(chunks, axis=0)
    val_preds_dn = stats.denormalise_target(val_preds)
    Y_va_dn = stats.denormalise_target(Y_va)
    return _record("b7_nbeats_lite", seed, val_preds_dn, Y_va_dn)


# ---- B8 PinballMean -------------------------------------------------------


def b8_pinball_mean(member_dirs: list[Path], seed: int = 42) -> dict:
    """Average the per-quantile predictions of the listed checkpoints."""
    seeds.seed_everything(seed)
    arrs = []
    targets = None
    for d in member_dirs:
        npz = np.load(d / "val_preds.npz")
        arrs.append(npz["preds"])
        if targets is None:
            targets = npz["targets"]
    stacked = np.stack(arrs, axis=0)  # (M, N, H, Q)
    mean = stacked.mean(axis=0)
    return _record("b8_pinball_mean", seed, mean, targets)


# ---- top-level driver -----------------------------------------------------


def run_all(seed: int = 42) -> dict[str, dict]:
    out: dict[str, dict] = {}
    tracking.init(experiment="heimdall-baselines")
    for name, fn in [
        ("b1_random_walk", b1_random_walk),
        ("b2_ewma", b2_ewma),
        ("b3_seasonal_naive", b3_seasonal_naive),
        ("b4_lightgbm_quantile", b4_lightgbm_quantile),
        ("b7_nbeats_lite", b7_nbeats_lite),
    ]:
        with tracking.run(name=f"{name}-seed{seed}", params={"seed": seed}):
            m = fn(seed=seed)
            tracking.log_metrics(m)
        out[name] = m
        print(name, json.dumps(m, indent=2))

    # B8 needs other baselines + forecasters — defer to leaderboard step.
    return out


if __name__ == "__main__":
    run_all(seed=42)
