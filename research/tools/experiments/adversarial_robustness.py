"""Adversarial robustness sweep for the canonical winners (A8, proposal §5.4).

Injects three perturbation families into the forecaster val/test inputs and
measures Δval_pinball at several magnitudes:

  - gaussian_noise(σ): N(0, σ × per-feature train_std) added independently.
  - feature_spike(σ): a single column (the target lag) gets a spike worth σ
    standard deviations every 96 steps.
  - regime_shift(σ): a global shift = σ × train_std applied to every feature.

For each (model, perturbation, σ), we re-run inference on the val window and
log pinball degradation. Output:
  outputs/adversarial/curves.json  ({model: [{perturb, sigma, pinball, delta}]})
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/adversarial"
SIGMAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
QUANTILES = (0.1, 0.5, 0.9)
SEED = 42


def _pinball(y, q, level):
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _pinball_mean(y, P):
    return float(np.mean([_pinball(y, P[..., qi], q) for qi, q in enumerate(QUANTILES)]))


def _load_val_canonical():
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, SEQ_LEN, HORIZON, make_windows,
    )
    val = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    Xva, Yva_norm, stats = make_windows(
        val, seq_len=SEQ_LEN, horizon=HORIZON, multivariate=True,
        feature_names=F_CANONICAL_FEATURES, anomaly_panel_path=anom,
    )
    Y = stats.denormalise_target(Yva_norm)
    return Xva, Y, stats, F_CANONICAL_FEATURES


def _perturb(X: np.ndarray, kind: str, sigma: float, train_std: np.ndarray,
             target_idx: int, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0.0:
        return X.copy()
    if kind == "gaussian":
        eps = rng.standard_normal(X.shape) * (sigma * train_std)
        return X + eps
    if kind == "spike":
        Xp = X.copy()
        # Spike at every 96th sample on the target lag column.
        idx = np.arange(0, X.shape[0], 96)
        Xp[idx, :, target_idx] += sigma * train_std[target_idx]
        return Xp
    if kind == "regime_shift":
        return X + sigma * train_std[None, None, :]
    raise ValueError(kind)


def _eval_patchtst(model_dir_name: str, X: np.ndarray, Y: np.ndarray, stats) -> float:
    from heimdall_forecaster.train.model import PatchTransformerQuantile  # noqa: PLC0415
    seed_dir = REPO / f"models/forecaster/{model_dir_name}/seed-{SEED}"
    cfg = json.loads((seed_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerQuantile(
        n_features=X.shape[-1], seq_len=cfg["seq_len"], horizon=cfg["horizon"],
        n_quantiles=3, patch_len=cfg["patch_len"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        use_rin=True,
    ).to(device).eval()
    sd = torch.load(seed_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    Xn = (X - stats.mean) / np.maximum(stats.std, 1e-6)
    preds = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256):
            preds.append(model(torch.from_numpy(Xn[i:i + 256]).float().to(device)).cpu().numpy())
    return _pinball_mean(Y, stats.denormalise_target(np.concatenate(preds, axis=0)))


def _eval_lgbm(model_dir_name: str, X: np.ndarray, Y: np.ndarray, stats) -> float:
    import lightgbm as lgb  # noqa: PLC0415
    seed_dir = REPO / f"models/forecaster/{model_dir_name}/seed-{SEED}"
    with open(seed_dir / "boosters.pkl", "rb") as fh:
        boosters = pickle.load(fh)
    Xn = (X - stats.mean) / np.maximum(stats.std, 1e-6)
    n, sl, f = Xn.shape
    Xf = Xn.reshape(n, sl * f).astype(np.float32)
    P = np.zeros((n, 16, len(QUANTILES)), dtype=np.float32)
    for h in range(16):
        for qi, q in enumerate(QUANTILES):
            b = lgb.Booster(model_str=boosters[f"h{h}_q{int(q * 100)}"])
            P[:, h, qi] = b.predict(Xf).astype(np.float32)
    return _pinball_mean(Y, P)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+",
                   default=["canonical_f1_lgbm_price", "canonical_f8_price"])
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, stats, feat_names = _load_val_canonical()
    rng = np.random.default_rng(SEED)
    target_idx = feat_names.index("imbalance_price_dkk_mwh_15min")
    train_std = stats.std

    results: dict[str, list[dict]] = {}
    for m in args.models:
        kind = "lgbm" if (REPO / f"models/forecaster/{m}/seed-{SEED}/boosters.pkl").exists() else "patchtst"
        baseline = (_eval_lgbm if kind == "lgbm" else _eval_patchtst)(m, X, Y, stats)
        rows = [{"perturb": "baseline", "sigma": 0.0,
                 "pinball": baseline, "delta": 0.0}]
        for perturb in ("gaussian", "spike", "regime_shift"):
            for s in SIGMAS:
                if s == 0.0:
                    continue
                Xp = _perturb(X, perturb, s, train_std, target_idx, rng)
                pin = (_eval_lgbm if kind == "lgbm" else _eval_patchtst)(m, Xp, Y, stats)
                rows.append({"perturb": perturb, "sigma": s,
                             "pinball": pin, "delta": pin - baseline})
                print(f"{m:<35s} {perturb:<14s} σ={s:.2f} pinball={pin:.2f} Δ={pin-baseline:+.2f}",
                      flush=True)
        results[m] = rows
    (OUT / "curves.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}/curves.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
