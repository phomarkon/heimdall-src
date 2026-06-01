"""Hurdle model for activation forecasting (proposal §5.x).

Two stages per direction (up, down):
  - Stage A: P(activation_event)  — LGBM binary classifier on F_CANONICAL,
    target = 1[mfrr_*_volume_mw > THRESH] for each of H=16 horizons.
  - Stage B: E[volume | event]   — LGBM quantile regressor on the event subset.

Aggregate score combines:
  - Brier / AUC / F1 for stage A (per direction × horizon).
  - Conditional pinball + CRPS-like for stage B (on event-only val rows).
  - Joint expected pinball: P(event)·pinball_cond + (1-P)·|0 - q50|.

Frozen seeds [13,42,137,1729,31415]. Test set untouched; trained/eval on
train/val splits with full GPU/CPU saturation via threaded LGBM.

Writes outputs/hurdle/{leaderboard.json, per_direction.json, stage_a_calib.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/hurdle"
MODEL_DIR = REPO / "models/forecaster/hurdle"
SEEDS = (13, 42, 137, 1729, 31415)
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 96
HORIZON = 16
THRESH = 0.0  # event = strictly positive activation


def _windows(panel_path: Path, anom_path: Path, *, feature_names: tuple[str, ...],
             target_col: str) -> tuple[np.ndarray, np.ndarray]:
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
    if n <= 0:
        raise ValueError(f"too few rows in {panel_path}: {len(y)}")
    Xw = np.zeros((n, SEQ_LEN, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, HORIZON), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i + SEQ_LEN]
        Yw[i] = y[i + SEQ_LEN:i + SEQ_LEN + HORIZON]
    return Xw, Yw


def _train_one(seed: int, direction: str, X_tr: np.ndarray, Y_tr: np.ndarray,
               X_va: np.ndarray, Y_va: np.ndarray) -> dict:
    np.random.seed(seed)
    n_tr, sl, f = X_tr.shape
    Xtr_flat = X_tr.reshape(n_tr, sl * f)
    Xva_flat = X_va.reshape(X_va.shape[0], sl * f)

    boosters_a: dict[str, lgb.Booster] = {}
    boosters_b: dict[str, lgb.Booster] = {}
    stage_a_metrics: list[dict] = []
    stage_b_metrics: list[dict] = []
    joint_pinball: list[float] = []

    for h in range(HORIZON):
        # Stage A — event classifier.
        yh_tr = (Y_tr[:, h] > THRESH).astype(np.int32)
        yh_va = (Y_va[:, h] > THRESH).astype(np.int32)
        if yh_tr.sum() < 10 or yh_tr.sum() == len(yh_tr):
            continue  # degenerate horizon
        dtrain = lgb.Dataset(Xtr_flat, label=yh_tr)
        clf = lgb.train({
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1, "seed": seed, "num_threads": 32,
        }, dtrain, num_boost_round=100)
        p_va = clf.predict(Xva_flat)
        brier = float(np.mean((p_va - yh_va) ** 2))
        # AUC
        order = np.argsort(p_va)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1)
        pos = yh_va.sum(); neg = len(yh_va) - pos
        auc = float((ranks[yh_va == 1].sum() - pos * (pos + 1) / 2) / max(1, pos * neg))
        # F1 at 0.5
        pred = (p_va >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (yh_va == 1)).sum())
        fp = int(((pred == 1) & (yh_va == 0)).sum())
        fn = int(((pred == 0) & (yh_va == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        stage_a_metrics.append({"h": h, "brier": brier, "auc": auc, "f1": f1,
                                "pos_rate_train": float(yh_tr.mean()),
                                "pos_rate_val": float(yh_va.mean())})
        boosters_a[f"h{h}"] = clf

        # Stage B — magnitude regressor on positive subset.
        mask = yh_tr == 1
        if mask.sum() < 50:
            continue
        Xtr_pos = Xtr_flat[mask]
        ytr_pos = Y_tr[mask, h]
        preds_b: dict[float, np.ndarray] = {}
        for q in QUANTILES:
            d = lgb.Dataset(Xtr_pos, label=ytr_pos)
            reg = lgb.train({
                "objective": "quantile", "alpha": q, "metric": "quantile",
                "num_leaves": 63, "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "verbose": -1, "seed": seed, "num_threads": 32,
            }, d, num_boost_round=100)
            boosters_b[f"h{h}_q{int(q * 100)}"] = reg
            preds_b[q] = reg.predict(Xva_flat).astype(np.float32)
        # Conditional pinball on val-event subset.
        mask_va = yh_va == 1
        if mask_va.any():
            yt = Y_va[mask_va, h]
            cond_pinballs = []
            for q in QUANTILES:
                err = yt - preds_b[q][mask_va]
                cond_pinballs.append(float(np.mean(np.maximum(q * err, (q - 1.0) * err))))
            stage_b_metrics.append({
                "h": h, "n_event_val": int(mask_va.sum()),
                "cond_pinball_mean": float(np.mean(cond_pinballs)),
                "cond_per_q": dict(zip(("q10", "q50", "q90"), cond_pinballs)),
            })
        # Joint expected pinball: P·pinball_cond + (1-P)·pinball_zero.
        p_event = p_va
        q50 = preds_b[0.5]
        # cond pinball at q50 on full val (positives only counted in expectation)
        err_pos = Y_va[:, h] - q50
        pinball_pos = np.maximum(0.5 * err_pos, -0.5 * err_pos)
        pinball_zero = np.maximum(0.5 * Y_va[:, h], -0.5 * Y_va[:, h])
        joint = p_event * pinball_pos + (1 - p_event) * pinball_zero
        joint_pinball.append(float(np.mean(joint)))

    return {
        "seed": seed, "direction": direction,
        "stage_a": stage_a_metrics, "stage_b": stage_b_metrics,
        "joint_pinball_mean": float(np.mean(joint_pinball)) if joint_pinball else float("nan"),
        "_boosters_a": boosters_a, "_boosters_b": boosters_b,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--directions", nargs="+", default=["up", "down"])
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel (seed,direction) trainers — each uses 32 LGBM threads")
    args = p.parse_args(argv)

    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom_train = REPO / "data/processed/anomaly_features_train.parquet"
    anom_val = REPO / "data/processed/anomaly_features_val.parquet"
    # We don't have a train/val split file for anomaly v2; reuse the combined one.
    if not anom_train.exists():
        anom_train = REPO / "data/processed/anomaly_features.parquet"
    if not anom_val.exists():
        anom_val = REPO / "data/processed/anomaly_features.parquet"

    # Build windows once per direction (reusable across seeds).
    direction_data: dict[str, tuple] = {}
    for direction in args.directions:
        target_col = f"mfrr_{direction}_volume_mw"
        print(f"[hurdle] building windows for direction={direction} target={target_col}", flush=True)
        X_tr, Y_tr = _windows(train_panel, anom_train, feature_names=F_CANONICAL_FEATURES,
                              target_col=target_col)
        X_va, Y_va = _windows(val_panel, anom_val, feature_names=F_CANONICAL_FEATURES,
                              target_col=target_col)
        ev_rate_tr = float((Y_tr > THRESH).any(axis=1).mean())
        ev_rate_va = float((Y_va > THRESH).any(axis=1).mean())
        print(f"  X_tr={X_tr.shape} X_va={X_va.shape} event_rate train={ev_rate_tr:.3f} val={ev_rate_va:.3f}", flush=True)
        direction_data[direction] = (X_tr, Y_tr, X_va, Y_va)

    jobs = [(seed, d) for d in args.directions for seed in args.seeds]
    print(f"[hurdle] launching {len(jobs)} (seed,dir) jobs with {args.max_workers} workers", flush=True)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        futs = {exe.submit(_train_one, seed, d, *direction_data[d]): (seed, d)
                for seed, d in jobs}
        for f in as_completed(futs):
            r = f.result()
            seed, d = futs[f]
            # Persist boosters
            md = MODEL_DIR / f"{d}_seed-{seed}"
            md.mkdir(parents=True, exist_ok=True)
            ba = r.pop("_boosters_a"); bb = r.pop("_boosters_b")
            with open(md / "stage_a.pkl", "wb") as fh: pickle.dump(
                {k: v.model_to_string() for k, v in ba.items()}, fh)
            with open(md / "stage_b.pkl", "wb") as fh: pickle.dump(
                {k: v.model_to_string() for k, v in bb.items()}, fh)
            (md / "metrics.json").write_text(json.dumps(r, indent=2))
            results.append(r)
            print(f"[hurdle] DONE seed={seed} dir={d} joint_pinball={r['joint_pinball_mean']:.3f}", flush=True)

    # Aggregate leaderboard.
    by_dir: dict[str, list[float]] = {d: [] for d in args.directions}
    by_dir_aucs: dict[str, list[float]] = {d: [] for d in args.directions}
    for r in results:
        by_dir[r["direction"]].append(r["joint_pinball_mean"])
        aucs = [a["auc"] for a in r["stage_a"]]
        if aucs: by_dir_aucs[r["direction"]].append(float(np.mean(aucs)))
    leaderboard = []
    for d in args.directions:
        leaderboard.append({
            "direction": d,
            "n_seeds": len(by_dir[d]),
            "joint_pinball_mean": float(np.mean(by_dir[d])) if by_dir[d] else None,
            "joint_pinball_std": float(np.std(by_dir[d])) if len(by_dir[d]) > 1 else 0.0,
            "stage_a_auc_mean": float(np.mean(by_dir_aucs[d])) if by_dir_aucs[d] else None,
        })
    (OUT / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    (OUT / "per_direction.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[hurdle] wrote {OUT}/leaderboard.json + per_direction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
