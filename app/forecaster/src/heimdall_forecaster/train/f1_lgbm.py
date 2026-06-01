"""F1 — Quantile LightGBM. docs/RESEARCH-PROPOSAL.md §4.2.2 row F1.

Per-quantile gradient-boosted-tree forecaster. Inputs are flattened windowed
features (the same SEQ_LEN-length lag stack used by F7), one LightGBM model
per (horizon-step, quantile) cell; predictions are stacked into the canonical
F-zoo (N, H, Q) shape.

Distinct from `experiments/baselines.py`'s `b4_lightgbm_quantile`, which uses
a *seasonal* feature stack (lag-1, lag-4, lag-96, hour-of-day, dow). F1 here
is the proposal's full-window quantile-LGBM — fair-comparison protocol with
F7 (same windows, same horizon).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import os

import lightgbm as lgb
import numpy as np

from heimdall_forecaster.train._utils import pinball_loss
from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    make_windows,
)
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds, tracking


REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class F1Config:
    name: str = "f1_lgbm"
    train_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    val_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    n_estimators: int = 250
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    experiment: str = "heimdall-forecaster-f1"
    multivariate: bool = False
    feature_names: tuple[str, ...] | None = None
    target: str = "price"
    target_column: str | None = None
    anomaly_panel: Path | None = None


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, T*F). LGBM doesn't see structure, just features."""
    n, t, f = X.shape
    return X.reshape(n, t * f).astype(np.float32)



def train_f1(cfg: F1Config) -> dict:
    seeds.seed_everything(cfg.seed)
    X_tr, Y_tr_norm, stats = make_windows(
        cfg.train_panel, seq_len=cfg.seq_len, horizon=cfg.horizon,
        multivariate=cfg.multivariate, feature_names=cfg.feature_names,
        target=cfg.target, target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
    )
    X_va, Y_va_norm, _ = make_windows(
        cfg.val_panel, seq_len=cfg.seq_len, horizon=cfg.horizon,
        multivariate=cfg.multivariate, feature_names=cfg.feature_names,
        target=cfg.target, target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
        stats=stats,
    )
    Y_tr = stats.denormalise_target(Y_tr_norm)
    Y_va = stats.denormalise_target(Y_va_norm)

    Xf_tr = _flatten_windows(X_tr)
    Xf_va = _flatten_windows(X_va)

    n_va = Xf_va.shape[0]
    preds = np.zeros((n_va, cfg.horizon, len(cfg.quantiles)), dtype=np.float32)

    out = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    boosters: dict[str, lgb.Booster] = {}

    base_params = {
        "objective": "quantile",
        "metric": "quantile",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "seed": cfg.seed,
        "verbose": -1,
    }
    n_cores = os.cpu_count() or 8
    run_params = {**base_params, "n_estimators": cfg.n_estimators, "horizon": cfg.horizon, "seed": cfg.seed}
    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", experiment=cfg.experiment, params=run_params):
        for h in range(cfg.horizon):
            ds = lgb.Dataset(Xf_tr, label=Y_tr[:, h])
            for qi, q in enumerate(cfg.quantiles):
                params = {**base_params, "alpha": q, "num_threads": min(n_cores, 64)}
                booster = lgb.train(
                    params,
                    ds,
                    num_boost_round=cfg.n_estimators,
                )
                yhat = booster.predict(Xf_va).astype(np.float32)
                preds[:, h, qi] = yhat
                boosters[f"h{h}_q{int(q*100)}"] = booster
        # Save val_preds.npz in standard F-zoo format.
        np.savez(out / "val_preds.npz", preds=preds, targets=Y_va.astype(np.float32))
        with open(out / "boosters.pkl", "wb") as fh:
            # LightGBM boosters round-trip via pickle for a small zoo entry.
            pickle.dump({k: v.model_to_string() for k, v in boosters.items()}, fh)
        # Save train-stat normaliser so the inference backend can apply the
        # SAME normalisation at test time. Without this the booster receives
        # raw DKK history when it was trained on z-scored input -> garbage.
        with open(out / "stats.pkl", "wb") as fh:
            pickle.dump(stats, fh)

        per_q = {}
        for qi, q in enumerate(cfg.quantiles):
            per_q[f"val_pinball_q{int(q*100)}"] = pinball_loss(Y_va, preds[..., qi], q)
        per_q["val_pinball_mean"] = float(np.mean(list(per_q.values())))
        sorted_p = np.sort(preds, axis=-1)
        per_q["val_q10_q90_coverage"] = float(
            np.mean((Y_va >= sorted_p[..., 0]) & (Y_va <= sorted_p[..., -1]))
        )

        aci = aci_coverage_from_val(out / "val_preds.npz", alpha=0.1, gamma=0.05, horizon_step=0)
        per_q["aci_alpha_target"] = aci.alpha_target
        per_q["aci_empirical_coverage"] = aci.empirical_coverage
        per_q["aci_mean_width"] = aci.mean_width
        per_q["seed"] = cfg.seed
        per_q["description"] = "f1_quantile_lightgbm_full_window"

        tracking.log_metrics({k: v for k, v in per_q.items() if isinstance(v, (int, float))})
        with open(out / "metrics.json", "w") as fh:
            json.dump(per_q, fh, indent=2)
    return per_q


def main() -> int:
    for s in (13, 42, 137, 1729, 31415):
        cfg = F1Config(seed=s)
        m = train_f1(cfg)
        print(f"seed={s} mean_pinball={m['val_pinball_mean']:.1f}  ACI={m['aci_empirical_coverage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
