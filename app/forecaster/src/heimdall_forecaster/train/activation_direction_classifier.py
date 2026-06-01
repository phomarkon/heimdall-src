"""Probabilistic activation-direction forecaster.

Predicts categorical probabilities for each horizon step:
``down``, ``neutral``, ``up``. This complements activation-volume quantile
models; it should be used as advisory society context, not verifier evidence.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from heimdall_forecaster.train._utils import resolve_device
from heimdall_forecaster.train.dataset import F8_FEATURES, HORIZON, SEQ_LEN, WindowStats, make_windows
from heimdall_forecaster.train.model import PatchEmbedding
from heimdall_ml import seeds, tracking

REPO_ROOT = Path(__file__).resolve().parents[5]
FROZEN_SEEDS = (13, 42, 137, 1729, 31415)
CLASS_NAMES = ("down", "neutral", "up")


@dataclass
class ActivationDirectionConfig:
    name: str = "activation_direction_f8"
    train_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    val_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    feature_names: tuple[str, ...] = field(default_factory=lambda: F8_FEATURES)
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    patch_len: int = 8
    d_model: int = 128
    nhead: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    experiment: str = "heimdall-forecaster-activation-direction"
    device: str = "auto"


class DirectionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PatchTransformerDirection(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int,
        horizon: int,
        patch_len: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.patch = PatchEmbedding(n_features, patch_len, d_model)
        n_patches = seq_len // patch_len
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * n_patches, horizon * len(CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x) + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        logits = self.head(h.reshape(h.shape[0], -1))
        return logits.reshape(h.shape[0], self.horizon, len(CLASS_NAMES))



def _labels_from_normalised(y_norm: np.ndarray, stats: WindowStats) -> np.ndarray:
    labels = np.rint(stats.denormalise_target(y_norm)).astype(np.int64)
    return np.where(labels < 0, 0, np.where(labels > 0, 2, 1)).astype(np.int64)


def _metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=-1)
    onehot = np.eye(len(CLASS_NAMES), dtype=np.float32)[labels]
    p_true = np.take_along_axis(probs, labels[..., None], axis=-1).squeeze(-1)
    eps = 1e-8
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=-1)
    metrics = {
        "activation_direction_accuracy": float(np.mean(pred == labels)),
        "activation_direction_nll": float(-np.mean(np.log(np.clip(p_true, eps, 1.0)))),
        "activation_direction_brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=-1))),
        "activation_direction_mean_entropy": float(np.mean(entropy)),
        "activation_direction_mean_confidence": float(np.mean(probs.max(axis=-1))),
    }
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        if bool(mask.any()):
            metrics[f"activation_direction_recall_{name}"] = float(np.mean(pred[mask] == idx))
    return metrics


def _hard_gate(metrics: dict[str, float], labels: np.ndarray, probs: np.ndarray) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    pred = probs.argmax(axis=-1)
    pred_dom = float(np.mean(pred == np.bincount(pred.ravel(), minlength=len(CLASS_NAMES)).argmax()))
    if pred_dom > 0.97:
        reasons.append(f"prediction collapse: dominant class share={pred_dom:.3f} > 0.97")
    if metrics["activation_direction_nll"] > 1.20:
        reasons.append(f"high NLL: {metrics['activation_direction_nll']:.3f} > 1.20")
    if metrics["activation_direction_brier"] > 0.80:
        reasons.append(f"high Brier: {metrics['activation_direction_brier']:.3f} > 0.80")
    if metrics["activation_direction_mean_confidence"] < 0.34:
        reasons.append(
            f"underconfident near-uniform output: confidence={metrics['activation_direction_mean_confidence']:.3f} < 0.34"
        )
    # If a class appears at >=5% frequency, force minimum recall floor.
    for idx, cname in enumerate(CLASS_NAMES):
        class_share = float(np.mean(labels == idx))
        if class_share >= 0.05:
            recall_key = f"activation_direction_recall_{cname}"
            if metrics.get(recall_key, 0.0) < 0.05:
                reasons.append(
                    f"class recall floor failed for {cname}: share={class_share:.3f}, recall={metrics.get(recall_key, 0.0):.3f}"
                )
    return (len(reasons) == 0), reasons


def train_activation_direction(cfg: ActivationDirectionConfig) -> dict[str, object]:
    seeds.seed_everything(cfg.seed)
    x_tr, y_tr_norm, stats = make_windows(
        cfg.train_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        feature_names=cfg.feature_names,
        target="activation_direction",
    )
    x_va, y_va_norm, _ = make_windows(
        cfg.val_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        feature_names=cfg.feature_names,
        target="activation_direction",
        stats=stats,
    )
    y_tr = _labels_from_normalised(y_tr_norm, stats)
    y_va = _labels_from_normalised(y_va_norm, stats)

    train_loader = DataLoader(DirectionDataset(x_tr, y_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(DirectionDataset(x_va, y_va), batch_size=cfg.batch_size, shuffle=False)
    device = resolve_device(cfg.device)
    model = PatchTransformerDirection(
        n_features=x_tr.shape[-1],
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    out = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "name": cfg.name,
        "seed": cfg.seed,
        "target": "activation_direction",
        "class_names": list(CLASS_NAMES),
        "feature_names": list(cfg.feature_names),
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "patch_len": cfg.patch_len,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "n_train_windows": int(x_tr.shape[0]),
        "n_val_windows": int(x_va.shape[0]),
    }
    tracking.init(experiment=cfg.experiment)
    probs_final: np.ndarray | None = None
    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", params=params):
        with tracking.track_experiment_compute(f"{cfg.name}.train"):
            for epoch in range(cfg.epochs):
                model.train()
                train_loss = 0.0
                n_train = 0
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    optim.zero_grad()
                    logits = model(x)
                    loss = loss_fn(logits.reshape(-1, len(CLASS_NAMES)), y.reshape(-1))
                    loss.backward()
                    optim.step()
                    train_loss += float(loss.item()) * x.shape[0]
                    n_train += x.shape[0]
                model.eval()
                val_loss = 0.0
                n_val = 0
                chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        logits = model(x)
                        loss = loss_fn(logits.reshape(-1, len(CLASS_NAMES)), y.reshape(-1))
                        val_loss += float(loss.item()) * x.shape[0]
                        n_val += x.shape[0]
                        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
                tracking.log_metrics(
                    {
                        "activation_direction_train_ce": train_loss / max(n_train, 1),
                        "activation_direction_val_ce": val_loss / max(n_val, 1),
                    },
                    step=epoch,
                )
                probs_final = np.concatenate(chunks, axis=0)
        assert probs_final is not None
        metrics = _metrics(probs_final, y_va)
        tracking.log_metrics(metrics)
        passed, reasons = _hard_gate(metrics, y_va, probs_final)
        tracking.log_metrics(
            {
                "activation_direction_gate_passed": 1.0 if passed else 0.0,
                "activation_direction_gate_reasons_count": float(len(reasons)),
            }
        )

    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "stats.pkl", "wb") as fh:
        pickle.dump(stats, fh)
    (out / "config.json").write_text(json.dumps(params, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(out / "val_direction_probs.npz", probs=probs_final.astype(np.float32), labels=y_va.astype(np.int64), class_names=np.array(CLASS_NAMES))
    passed, reasons = _hard_gate(metrics, y_va, probs_final)
    (out / "gate_report.json").write_text(
        json.dumps({"passed": passed, "reasons": reasons, "metrics": metrics}, indent=2)
    )
    if not passed:
        raise RuntimeError(f"activation direction hard-gate failed for seed {cfg.seed}: {reasons}")
    return {"ckpt": out / "model.pt", **metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    for seed in ([args.seed] if args.seed is not None else list(FROZEN_SEEDS)):
        cfg = ActivationDirectionConfig(seed=seed)
        if args.epochs is not None:
            cfg.epochs = args.epochs
        result = train_activation_direction(cfg)
        print(json.dumps({"seed": seed, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
