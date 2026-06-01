"""F2 — Bayesian Linear Regression. docs/RESEARCH-PROPOSAL.md §4.2.2 row F2.

scikit-learn's ``BayesianRidge`` gives a Gaussian posterior over the prediction
mean. We fit one regressor per horizon step on the flattened SEQ_LEN-window
input (matches F1's feature stack), then derive quantile predictions from the
posterior mean + std via the Gaussian inverse CDF.

The seed only affects scikit's bagging RNG and the ACI buffer ordering; the
BayesianRidge ML solution itself is deterministic given the data, so per-seed
val pinball is identical across the frozen seed list (mirrors F0). We still
emit per-seed dirs so the leaderboard treats it uniformly.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

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
class F2Config:
    name: str = "f2_blr"
    train_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    val_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    experiment: str = "heimdall-forecaster-f2"
    multivariate: bool = False
    feature_names: tuple[str, ...] | None = None
    target: str = "price"
    target_column: str | None = None
    anomaly_panel: Path | None = None


def _flatten(X: np.ndarray) -> np.ndarray:
    n, t, f = X.shape
    return X.reshape(n, t * f).astype(np.float64)



def train_f2(cfg: F2Config) -> dict:
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

    Xf_tr = _flatten(X_tr)
    Xf_va = _flatten(X_va)

    n_va = Xf_va.shape[0]
    preds = np.zeros((n_va, cfg.horizon, len(cfg.quantiles)), dtype=np.float32)

    out = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    regressors: dict[str, BayesianRidge] = {}

    with tracking.run(
        name=f"{cfg.name}-seed{cfg.seed}",
        experiment=cfg.experiment,
        params={"seed": cfg.seed, "horizon": cfg.horizon, "model": "BayesianRidge"},
    ):
        for h in range(cfg.horizon):
            reg = BayesianRidge()
            reg.fit(Xf_tr, Y_tr[:, h])
            mu, sigma = reg.predict(Xf_va, return_std=True)
            for qi, q in enumerate(cfg.quantiles):
                z = norm.ppf(q)
                preds[:, h, qi] = (mu + z * sigma).astype(np.float32)
            regressors[f"h{h}"] = reg
        np.savez(out / "val_preds.npz", preds=preds, targets=Y_va.astype(np.float32))
        # Save train-stat normaliser for the inference backend (was missing
        # pre-2026-05-17; root cause of F2 test pinball = 357k).
        with open(out / "stats.pkl", "wb") as fh:
            pickle.dump(stats, fh)
        with open(out / "regressors.pkl", "wb") as fh:
            pickle.dump(regressors, fh)

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
        per_q["description"] = "f2_bayesian_linear_regression_full_window"

        tracking.log_metrics({k: v for k, v in per_q.items() if isinstance(v, (int, float))})
        with open(out / "metrics.json", "w") as fh:
            json.dump(per_q, fh, indent=2)
    return per_q


def main() -> int:
    for s in (13, 42, 137, 1729, 31415):
        cfg = F2Config(seed=s)
        m = train_f2(cfg)
        print(f"seed={s} mean_pinball={m['val_pinball_mean']:.1f}  ACI={m['aci_empirical_coverage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
