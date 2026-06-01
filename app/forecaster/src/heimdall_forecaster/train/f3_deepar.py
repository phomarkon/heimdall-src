"""F3 — DeepAR-Lite. Per docs/RESEARCH-PROPOSAL.md §4.2.2 (forecaster zoo row F3).

A small 1-layer LSTM with hidden=64 and a Student-t output head (Salinas et al.
2020). Negative log-likelihood loss; quantiles are derived analytically from the
predicted (mu, sigma, nu) at inference time so they remain well-calibrated.

~50k params; trained on the same windows as F7 (univariate target only).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    QuantilePanelDataset,
    make_windows,
)
from heimdall_forecaster.train._utils import resolve_device
from heimdall_ml import seeds, tracking


# -- Student-t parametrisation ------------------------------------------------


def _student_t_quantile(mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile of a location-scale Student-t (nu > 2). Approximation via
    inverse CDF of the standard t evaluated at q, then shifted/scaled.

    For training we don't need exact inverse CDF — at inference we use
    scipy.stats.t.ppf, but this helper supports tests."""
    # Approximate using a normal correction; for nu >> 2 this is fine.
    from scipy.stats import t as student_t
    z = student_t.ppf(q, df=float(nu.detach().mean().clamp_min(2.5).item()))
    return mu + sigma * z


class DeepARLite(nn.Module):
    """1-layer LSTM + Student-t head.

    Output per (B, T) step is (mu, log_sigma, log_nu_minus_2). For multi-step
    we unroll auto-regressively: the median (mu) is fed back as the next input.
    """

    def __init__(self, hidden: int = 64, horizon: int = HORIZON) -> None:
        super().__init__()
        self.hidden = hidden
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 3)  # mu, log_sigma, log_nu_minus_2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, 1). Returns (mu, sigma, nu) at each horizon step.
        b = x.shape[0]
        # Encode the full history.
        _, (h, c) = self.lstm(x)
        last = x[:, -1:, :]  # last value as decoder input

        mus, sigmas, nus = [], [], []
        for _ in range(self.horizon):
            out, (h, c) = self.lstm(last, (h, c))
            params = self.head(out[:, -1, :])  # (B, 3)
            mu = params[:, 0]
            sigma = torch.nn.functional.softplus(params[:, 1]) + 1e-3
            nu = torch.nn.functional.softplus(params[:, 2]) + 2.5  # > 2 for finite var
            mus.append(mu)
            sigmas.append(sigma)
            nus.append(nu)
            last = mu.detach().unsqueeze(-1).unsqueeze(-1)
        mu = torch.stack(mus, dim=1)  # (B, H)
        sigma = torch.stack(sigmas, dim=1)
        nu = torch.stack(nus, dim=1)
        return mu, sigma, nu


def _student_t_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a location-scale Student-t."""
    z = (y - mu) / sigma
    # log f(y) = const(nu) - log sigma - (nu+1)/2 log(1 + z^2/nu)
    log_const = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(torch.pi * nu)
    )
    log_term = -((nu + 1.0) / 2.0) * torch.log1p(z**2 / nu)
    log_pdf = log_const - torch.log(sigma) + log_term
    return -log_pdf.mean()


@dataclass
class F3Config:
    name: str = "f3_lite"  # ADR-0006: F3 := 5-seed F7 deep ensemble (f3_ensemble/); the LSTM-DeepAR here is appendix-only.
    train_panel: Path = Path("data/processed/dk1_panel_train.parquet")
    val_panel: Path = Path("data/processed/dk1_panel_val.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    hidden: int = 64
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = Path("models/forecaster")
    device: str = "auto"
    experiment: str = "heimdall-forecaster-f3"



def train_f3(cfg: F3Config) -> dict[str, object]:
    seeds.seed_everything(cfg.seed)
    X_tr, Y_tr, stats = make_windows(
        cfg.train_panel, seq_len=cfg.seq_len, horizon=cfg.horizon, multivariate=False
    )
    X_va, Y_va, _ = make_windows(
        cfg.val_panel, seq_len=cfg.seq_len, horizon=cfg.horizon,
        multivariate=False, stats=stats,
    )
    train_loader = DataLoader(QuantilePanelDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(QuantilePanelDataset(X_va, Y_va), batch_size=cfg.batch_size, shuffle=False)

    device = resolve_device(cfg.device)
    model = DeepARLite(hidden=cfg.hidden, horizon=cfg.horizon).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tracking.init(experiment=cfg.experiment)
    n_params = sum(p.numel() for p in model.parameters())
    params_cfg = {
        "name": cfg.name,
        "seed": cfg.seed,
        "hidden": cfg.hidden,
        "n_params": int(n_params),
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "n_train_windows": int(X_tr.shape[0]),
        "n_val_windows": int(X_va.shape[0]),
    }
    out_dir = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_mu_final: np.ndarray | None = None
    val_sigma_final: np.ndarray | None = None
    val_nu_final: np.ndarray | None = None

    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", params=params_cfg):
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            n = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optim.zero_grad()
                mu, sigma, nu = model(x)
                loss = _student_t_nll(y, mu, sigma, nu)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                running += loss.item() * x.shape[0]
                n += x.shape[0]
            train_loss = running / max(n, 1)

            model.eval()
            v_running = 0.0
            v_n = 0
            mus, sigmas, nus = [], [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    mu, sigma, nu = model(x)
                    v_running += _student_t_nll(y, mu, sigma, nu).item() * x.shape[0]
                    v_n += x.shape[0]
                    mus.append(mu.cpu().numpy())
                    sigmas.append(sigma.cpu().numpy())
                    nus.append(nu.cpu().numpy())
            val_loss = v_running / max(v_n, 1)
            val_mu_final = np.concatenate(mus, axis=0)
            val_sigma_final = np.concatenate(sigmas, axis=0)
            val_nu_final = np.concatenate(nus, axis=0)
            tracking.log_metrics({"train_nll": train_loss, "val_nll": val_loss}, step=epoch)

        assert val_mu_final is not None
        # Compute predictive quantiles: shift and scale a Student-t critical value.
        from scipy.stats import t as student_t

        Y_va_dn = stats.denormalise_target(Y_va)
        val_preds = np.empty((*val_mu_final.shape, len(cfg.quantiles)), dtype=np.float64)
        for qi, q in enumerate(cfg.quantiles):
            # Use mean nu per (window, horizon) — already (N, H).
            crit = student_t.ppf(q, df=np.clip(val_nu_final, 2.5, 1e3))
            mu_dn = stats.denormalise_target(val_mu_final)
            sig_dn = val_sigma_final * stats.target_std
            val_preds[..., qi] = mu_dn + sig_dn * crit

        per_q = {}
        for qi, q in enumerate(cfg.quantiles):
            err = Y_va_dn - val_preds[..., qi]
            per_q[f"val_pinball_q{int(q*100)}"] = float(
                np.mean(np.maximum(q * err, (q - 1.0) * err))
            )
        per_q["val_pinball_mean_dkk"] = float(np.mean(list(per_q.values())))

        sorted_pred = np.sort(val_preds, axis=-1)
        coverage = float(
            np.mean(
                (Y_va_dn >= sorted_pred[:, :, 0]) & (Y_va_dn <= sorted_pred[:, :, -1])
            )
        )
        tracking.log_metrics({**per_q, "val_q10_q90_coverage": coverage})

        torch.save(model.state_dict(), out_dir / "model.pt")
        with open(out_dir / "stats.pkl", "wb") as fh:
            pickle.dump(stats, fh)
        with open(out_dir / "config.json", "w") as fh:
            json.dump({**params_cfg, "out_dir": str(out_dir)}, fh, indent=2)
        np.savez(out_dir / "val_preds.npz", preds=val_preds, targets=Y_va_dn)
        metrics = {**per_q, "val_q10_q90_coverage": coverage}
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

    return {
        "val_pinball_mean": per_q["val_pinball_mean_dkk"],
        "val_q10_q90_coverage": coverage,
        "per_quantile": per_q,
        "ckpt": out_dir / "model.pt",
        "val_preds": val_preds,
        "val_targets": Y_va_dn,
    }


__all__ = ["DeepARLite", "F3Config", "train_f3"]
