"""F5/F6 neural-process forecaster training.

F5 is a compact ConvCNP-style model. F6 uses attention over context points.
Both emit Gaussian predictive distributions and persist standard F-zoo
``val_preds.npz`` quantiles for calibration/evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from heimdall_forecaster.train._utils import pinball_loss, resolve_device
from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    QuantilePanelDataset,
    make_windows,
)
from heimdall_forecaster.train.model import (
    AttentiveNPForecaster,
    ConvCNPForecaster,
    gaussian_nll,
)
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds, tracking


REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class NeuralProcessConfig:
    name: str = "f5"
    train_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    val_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    feature_names: tuple[str, ...] | None = None
    target: str = "price"
    target_column: str | None = None
    anomaly_panel: Path | None = None
    d_model: int = 96
    n_layers: int = 4
    nhead: int = 4
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    experiment: str = "heimdall-forecaster-neural-process"
    device: str = "auto"



def _model(cfg: NeuralProcessConfig, n_features: int) -> torch.nn.Module:
    if cfg.name.startswith("f6"):
        return AttentiveNPForecaster(
            n_features=n_features,
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
    return ConvCNPForecaster(
        n_features=n_features,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )


def _quantiles(mu: np.ndarray, sigma: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    z = np.asarray([NormalDist().inv_cdf(q) for q in levels], dtype=np.float32)
    return mu[..., None] + sigma[..., None] * z



def train_neural_process(cfg: NeuralProcessConfig) -> dict[str, object]:
    seeds.seed_everything(cfg.seed)
    X_tr, Y_tr, stats = make_windows(
        cfg.train_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        feature_names=cfg.feature_names,
        target=cfg.target,
        target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
    )
    X_va, Y_va, _ = make_windows(
        cfg.val_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        feature_names=cfg.feature_names,
        target=cfg.target,
        target_column=cfg.target_column,
        anomaly_panel_path=cfg.anomaly_panel,
        stats=stats,
    )
    train_loader = DataLoader(
        QuantilePanelDataset(X_tr, Y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(QuantilePanelDataset(X_va, Y_va), batch_size=cfg.batch_size)
    device = resolve_device(cfg.device)
    model = _model(cfg, X_tr.shape[-1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out_dir = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "name": cfg.name,
        "seed": cfg.seed,
        "target": cfg.target,
        "target_column": cfg.target_column or "",
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "nhead": cfg.nhead,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "n_features": X_tr.shape[-1],
        "target_name": stats.target_name,
    }
    val_mu_final: np.ndarray | None = None
    val_sigma_final: np.ndarray | None = None
    tracking.init(experiment=cfg.experiment)
    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", params=params):
        with tracking.track_experiment_compute(f"{cfg.name}.train"):
            for epoch in range(cfg.epochs):
                model.train()
                running = 0.0
                n = 0
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    optim.zero_grad()
                    mu, sigma = model(x)
                    loss = gaussian_nll(mu, sigma, y)
                    loss.backward()
                    optim.step()
                    running += loss.item() * x.shape[0]
                    n += x.shape[0]
                model.eval()
                v_running = 0.0
                v_n = 0
                mu_chunks: list[np.ndarray] = []
                sigma_chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        mu, sigma = model(x)
                        v_running += gaussian_nll(mu, sigma, y).item() * x.shape[0]
                        v_n += x.shape[0]
                        mu_chunks.append(mu.cpu().numpy())
                        sigma_chunks.append(sigma.cpu().numpy())
                tracking.log_metrics(
                    {
                        "train_gaussian_nll": running / max(n, 1),
                        "val_gaussian_nll": v_running / max(v_n, 1),
                    },
                    step=epoch,
                )
                val_mu_final = np.concatenate(mu_chunks, axis=0)
                val_sigma_final = np.concatenate(sigma_chunks, axis=0)

        assert val_mu_final is not None and val_sigma_final is not None
        mu_dn = stats.denormalise_target(val_mu_final)
        sigma_dn = val_sigma_final * stats.target_std
        preds = np.sort(_quantiles(mu_dn, sigma_dn, cfg.quantiles), axis=-1).astype(np.float32)
        targets = stats.denormalise_target(Y_va).astype(np.float32)
        per_q = {
            f"val_pinball_q{int(q*100)}": pinball_loss(targets, preds[..., qi], q)
            for qi, q in enumerate(cfg.quantiles)
        }
        per_q["val_pinball_mean"] = float(np.mean(list(per_q.values())))
        per_q["val_q10_q90_coverage"] = float(
            np.mean((targets >= preds[..., 0]) & (targets <= preds[..., -1]))
        )
        np.savez(out_dir / "val_preds.npz", preds=preds, targets=targets)
        aci = aci_coverage_from_val(out_dir / "val_preds.npz", alpha=0.1, gamma=0.05, horizon_step=0)
        per_q["aci_alpha_target"] = aci.alpha_target
        per_q["aci_empirical_coverage"] = aci.empirical_coverage
        per_q["aci_mean_width"] = aci.mean_width
        tracking.log_metrics(per_q)
        torch.save(model.state_dict(), out_dir / "model.pt")
        with open(out_dir / "stats.pkl", "wb") as fh:
            pickle.dump(stats, fh)
        with open(out_dir / "config.json", "w") as fh:
            json.dump(params, fh, indent=2)
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(per_q, fh, indent=2)
    return {"ckpt": out_dir / "model.pt", "stats_path": out_dir / "stats.pkl", **per_q}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name", choices=("f5", "f6"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target", choices=("price", "activation_volume", "activation_direction"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    raw: dict[str, object] = {}
    if args.config is not None:
        raw = yaml.safe_load(args.config.read_text()) or {}
        for key in ("train_panel", "val_panel", "out_dir"):
            if key in raw and raw[key]:
                raw[key] = REPO_ROOT / str(raw[key])
    if args.name is not None:
        raw["name"] = args.name
    if "name" not in raw:
        parser.error("--name or --config with name is required")
    if args.target is not None:
        raw["target"] = args.target
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    seeds_to_run = [args.seed] if args.seed is not None else [13, 42, 137, 1729, 31415]
    for seed in seeds_to_run:
        cfg = NeuralProcessConfig(**{**raw, "seed": seed})
        result = train_neural_process(cfg)
        print(json.dumps({"name": cfg.name, "seed": seed, "target": cfg.target, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
