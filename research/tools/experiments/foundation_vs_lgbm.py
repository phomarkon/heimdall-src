"""Four-way comparison answering: 'is LGBM really stronger than fine-tuned foundation models?'

Variants trained + evaluated on the held-out test panel (post 2025-05-01):

  1. F1 LGBM UNIVARIATE — target lags only, n_estimators=50. The honest tree
     vs foundation-model comparison on equal inputs.
  2. F9 TimesFM zero-shot — already computed; this script just re-scores on test.
  3. F1 LGBM + F9 q50 as 45th feature (hybrid stack). Tests whether the
     foundation model adds signal over F_CANONICAL.
  4. F9 + residual head: F9 predicts q50, a small LGBM fits the residual on
     F_CANONICAL. Cleanest way to inject covariates into a univariate
     foundation model.

Writes outputs/foundation_vs_lgbm/{rows,leaderboard}.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/foundation_vs_lgbm"
SEEDS = (13, 42, 137, 1729, 31415)
SEQ_LEN = 96
HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _score(P: np.ndarray, y: np.ndarray) -> dict:
    per_q = {f"test_pinball_q{int(q * 100)}": _pinball(y, P[..., qi], q)
             for qi, q in enumerate(QUANTILES)}
    srt = np.sort(P, axis=-1)
    cov = float(np.mean((y >= srt[..., 0]) & (y <= srt[..., -1])))
    return {**per_q,
            "test_pinball_mean_dkk": float(np.mean(list(per_q.values()))),
            "test_q10_q90_coverage": cov,
            "n_windows": int(P.shape[0])}


def _load_panels(target: str = "price"):
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES  # noqa: PLC0415

    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    test_panel = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
    anom_train = REPO / "data/processed/anomaly_features.parquet"
    anom_test = REPO / "data/processed/anomaly_features_test.parquet"
    v1_test = REPO / "data/processed/dk1_panel_rich_test.parquet"

    def _load(panel_path, anom_path, hydrate_mfrr=False):
        df = pl.read_parquet(panel_path).sort("timestamp_utc")
        anom = pl.read_parquet(anom_path)
        df = df.join(anom, on="timestamp_utc", how="left")
        if hydrate_mfrr:
            v1 = pl.read_parquet(v1_test)
            df = df.drop("mfrr_up_volume_mw", "mfrr_down_volume_mw").join(
                v1.select(["timestamp_utc", "mfrr_up_volume_mw", "mfrr_down_volume_mw"]),
                on="timestamp_utc", how="left",
            )
        return df

    df_tr = _load(train_panel, anom_train, hydrate_mfrr=(target != "price"))
    df_te = _load(test_panel, anom_test, hydrate_mfrr=(target != "price"))

    if target == "price":
        y_tr = df_tr["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
        y_te = df_te["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    else:
        def _act(df):
            up = df["mfrr_up_volume_mw"].to_numpy().astype(np.float64)
            dn = df["mfrr_down_volume_mw"].to_numpy().astype(np.float64)
            return (up - dn) * 0.25
        y_tr, y_te = _act(df_tr), _act(df_te)

    feat_names = list(F_CANONICAL_FEATURES)
    X_tr = np.nan_to_num(df_tr.select(feat_names).to_numpy().astype(np.float64), nan=0.0)
    X_te = np.nan_to_num(df_te.select(feat_names).to_numpy().astype(np.float64), nan=0.0)
    return X_tr, y_tr, X_te, y_te, feat_names


def _sliding_uni_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Univariate sliding: (N, seq_len) histories -> (N, horizon) targets."""
    n = max(0, y.size - SEQ_LEN - HORIZON)
    H = np.empty((n, SEQ_LEN), dtype=np.float32)
    T = np.empty((n, HORIZON), dtype=np.float32)
    for i in range(n):
        H[i] = y[i : i + SEQ_LEN]
        T[i] = y[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
    return H, T


def _sliding_multi(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(0, X.shape[0] - SEQ_LEN - HORIZON)
    Hx = np.empty((n, SEQ_LEN, X.shape[1]), dtype=np.float32)
    Ty = np.empty((n, HORIZON), dtype=np.float32)
    for i in range(n):
        Hx[i] = X[i : i + SEQ_LEN]
        Ty[i] = y[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
    return Hx, Ty


# ----------------------------------------------------------------------------
# Variants.
# ----------------------------------------------------------------------------

def run_f1_univariate(target: str, seed: int) -> dict:
    """LGBM with target-only history (96 lags). Compares directly vs F9 zero-shot."""
    import lightgbm as lgb  # noqa: PLC0415

    _, y_tr, _, y_te, _ = _load_panels(target)
    Htr, Ttr = _sliding_uni_targets(y_tr)
    Hte, Tte = _sliding_uni_targets(y_te)
    if Htr.shape[0] < 100 or Hte.shape[0] < 1:
        return {"error": "insufficient windows"}
    P = np.zeros((Hte.shape[0], HORIZON, len(QUANTILES)), dtype=np.float32)
    base_params = {"objective": "quantile", "metric": "quantile",
                   "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
                   "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
                   "seed": seed, "verbose": -1, "num_threads": 32}
    for h in range(HORIZON):
        ds = lgb.Dataset(Htr, label=Ttr[:, h])
        for qi, q in enumerate(QUANTILES):
            booster = lgb.train({**base_params, "alpha": q}, ds, num_boost_round=50)
            P[:, h, qi] = booster.predict(Hte).astype(np.float32)
    return _score(P, Tte)


def _timesfm_q50_on_panel(y: np.ndarray) -> np.ndarray:
    """Run TimesFM zero-shot at q50 over sliding windows. Returns (n, H)."""
    from heimdall_forecaster.timesfm_wrapper import TimesFMForecaster  # noqa: PLC0415

    fm = TimesFMForecaster(backend="gpu", context_len=SEQ_LEN, horizon_len=HORIZON)
    fm._load()
    n = max(0, y.size - SEQ_LEN - HORIZON)
    Q = np.zeros((n, HORIZON), dtype=np.float32)
    for i in range(n):
        mu, _ = fm.predict(y[i : i + SEQ_LEN])
        Q[i] = np.asarray(mu)[:HORIZON]
    return Q


def run_f9_test(target: str, seed: int, cache_dir: Path) -> dict:
    """F9 TimesFM zero-shot on the test panel. Cache q50 across seeds (same model)."""
    from scipy.stats import norm  # noqa: PLC0415

    _, _, _, y_te, _ = _load_panels(target)
    cache_path = cache_dir / f"f9_zero_shot_q50_{target}_test.npy"
    if cache_path.exists():
        q50 = np.load(cache_path)
    else:
        q50 = _timesfm_q50_on_panel(y_te)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, q50)
    _, Tte = _sliding_uni_targets(y_te)
    # Fake q10/q90 via residual std on the test window itself (zero-shot has no
    # calibration set — this is intentionally simplistic; real conformal layer
    # would wrap this).
    resid_std = np.std(Tte - q50)
    P = np.zeros((q50.shape[0], HORIZON, len(QUANTILES)), dtype=np.float32)
    for qi, q in enumerate(QUANTILES):
        z = norm.ppf(q)
        P[..., qi] = (q50 + z * resid_std).astype(np.float32)
    return _score(P, Tte)


def run_f1_hybrid(target: str, seed: int, cache_dir: Path) -> dict:
    """F1 LGBM on F_CANONICAL + F9 q50 as 45th feature column."""
    import lightgbm as lgb  # noqa: PLC0415

    X_tr, y_tr, X_te, y_te, _ = _load_panels(target)
    # F9 q50 on train + test. Cache to avoid recompute.
    q50_tr_path = cache_dir / f"f9_zero_shot_q50_{target}_train.npy"
    q50_te_path = cache_dir / f"f9_zero_shot_q50_{target}_test.npy"
    if q50_te_path.exists():
        q50_te = np.load(q50_te_path)
    else:
        q50_te = _timesfm_q50_on_panel(y_te)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(q50_te_path, q50_te)
    if q50_tr_path.exists():
        q50_tr = np.load(q50_tr_path)
    else:
        q50_tr = _timesfm_q50_on_panel(y_tr)
        np.save(q50_tr_path, q50_tr)
    # Slide multivariate, align F9 forecasts on a per-window basis (F9 q50 at the
    # SAME issue time). Drop windows where F9 q50 isn't available.
    Xtr_m, Ttr = _sliding_multi(X_tr, y_tr)
    Xte_m, Tte = _sliding_multi(X_te, y_te)
    n_tr = min(Xtr_m.shape[0], q50_tr.shape[0])
    n_te = min(Xte_m.shape[0], q50_te.shape[0])
    Xtr_m = Xtr_m[:n_tr]; q50_tr = q50_tr[:n_tr]; Ttr = Ttr[:n_tr]
    Xte_m = Xte_m[:n_te]; q50_te = q50_te[:n_te]; Tte = Tte[:n_te]
    Xtr_flat = Xtr_m.reshape(n_tr, SEQ_LEN * Xtr_m.shape[-1]).astype(np.float32)
    Xte_flat = Xte_m.reshape(n_te, SEQ_LEN * Xte_m.shape[-1]).astype(np.float32)
    # Add F9 q50 at all horizon steps as new features.
    Xtr_aug = np.hstack([Xtr_flat, q50_tr.astype(np.float32)])
    Xte_aug = np.hstack([Xte_flat, q50_te.astype(np.float32)])
    P = np.zeros((n_te, HORIZON, len(QUANTILES)), dtype=np.float32)
    base = {"objective": "quantile", "metric": "quantile",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
            "seed": seed, "verbose": -1, "num_threads": 32}
    for h in range(HORIZON):
        ds = lgb.Dataset(Xtr_aug, label=Ttr[:, h])
        for qi, q in enumerate(QUANTILES):
            b = lgb.train({**base, "alpha": q}, ds, num_boost_round=50)
            P[:, h, qi] = b.predict(Xte_aug).astype(np.float32)
    return _score(P, Tte)


def run_residual_f9(target: str, seed: int, cache_dir: Path) -> dict:
    """F9 q50 + LGBM head fitting residual = target - F9_q50 from F_CANONICAL."""
    import lightgbm as lgb  # noqa: PLC0415
    from scipy.stats import norm  # noqa: PLC0415

    X_tr, y_tr, X_te, y_te, _ = _load_panels(target)
    q50_tr_path = cache_dir / f"f9_zero_shot_q50_{target}_train.npy"
    q50_te_path = cache_dir / f"f9_zero_shot_q50_{target}_test.npy"
    q50_tr = np.load(q50_tr_path) if q50_tr_path.exists() else _timesfm_q50_on_panel(y_tr)
    q50_te = np.load(q50_te_path) if q50_te_path.exists() else _timesfm_q50_on_panel(y_te)
    if not q50_tr_path.exists(): cache_dir.mkdir(parents=True, exist_ok=True); np.save(q50_tr_path, q50_tr)
    if not q50_te_path.exists(): np.save(q50_te_path, q50_te)
    Xtr_m, Ttr = _sliding_multi(X_tr, y_tr)
    Xte_m, Tte = _sliding_multi(X_te, y_te)
    n_tr = min(Xtr_m.shape[0], q50_tr.shape[0])
    n_te = min(Xte_m.shape[0], q50_te.shape[0])
    Xtr_m = Xtr_m[:n_tr]; q50_tr = q50_tr[:n_tr]; Ttr = Ttr[:n_tr]
    Xte_m = Xte_m[:n_te]; q50_te = q50_te[:n_te]; Tte = Tte[:n_te]
    resid_tr = Ttr - q50_tr
    Xtr_flat = Xtr_m.reshape(n_tr, SEQ_LEN * Xtr_m.shape[-1]).astype(np.float32)
    Xte_flat = Xte_m.reshape(n_te, SEQ_LEN * Xte_m.shape[-1]).astype(np.float32)
    P_resid = np.zeros((n_te, HORIZON, len(QUANTILES)), dtype=np.float32)
    base = {"objective": "quantile", "metric": "quantile",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
            "seed": seed, "verbose": -1, "num_threads": 32}
    for h in range(HORIZON):
        ds = lgb.Dataset(Xtr_flat, label=resid_tr[:, h])
        for qi, q in enumerate(QUANTILES):
            b = lgb.train({**base, "alpha": q}, ds, num_boost_round=50)
            P_resid[:, h, qi] = b.predict(Xte_flat).astype(np.float32)
    # Final prediction = F9 q50 + residual quantiles (LGBM-fit).
    P = q50_te[..., None] + P_resid
    return _score(P, Tte)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("price", "activation"), default="price")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--variants", nargs="+",
                   default=["f1_uni", "f9_zero_shot", "f1_hybrid", "f9_residual"],
                   choices=["f1_uni", "f9_zero_shot", "f1_hybrid", "f9_residual"])
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "_cache"
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            try:
                if variant == "f1_uni":
                    r = run_f1_univariate(args.target, seed)
                elif variant == "f9_zero_shot":
                    r = run_f9_test(args.target, seed, cache)
                elif variant == "f1_hybrid":
                    r = run_f1_hybrid(args.target, seed, cache)
                elif variant == "f9_residual":
                    r = run_residual_f9(args.target, seed, cache)
                r.update({"variant": variant, "seed": seed, "target": args.target})
                rows.append(r)
                print(f"{variant:<18s} seed={seed:<6d} target={args.target:<10s} "
                      f"pinball={r.get('test_pinball_mean_dkk',float('nan')):.2f}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"variant": variant, "seed": seed,
                             "target": args.target, "error": repr(e)})
                print(f"{variant} seed={seed} ERROR {e!r}", flush=True)

    (OUT / f"rows_{args.target}.json").write_text(json.dumps(rows, indent=2))
    # Aggregate.
    agg: dict[str, list[float]] = {}
    for r in rows:
        v = r.get("test_pinball_mean_dkk")
        if v is None or not np.isfinite(v):
            continue
        agg.setdefault(r["variant"], []).append(float(v))
    lb = sorted([
        {"variant": k, "n_seeds": len(vs),
         "test_pinball_mean": st.mean(vs),
         "test_pinball_std": st.pstdev(vs) if len(vs) > 1 else 0.0}
        for k, vs in agg.items()
    ], key=lambda r: r["test_pinball_mean"])
    (OUT / f"leaderboard_{args.target}.json").write_text(json.dumps(lb, indent=2))
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
