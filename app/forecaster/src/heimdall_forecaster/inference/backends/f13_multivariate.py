"""F13 — RIN-augmented multivariate patch-TST backend.

Accepts a (seq_len, n_features) array as ``history`` and runs the
PatchTransformerQuantile with use_rin=True. The trainer's stats normalisation
is applied on the target dimension only; RIN handles per-window centering.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register

F13_CHECKPOINT_FILES = ("config.json", "model.pt", "stats.pkl")


@dataclass
class _F13Backend:
    name: str
    seed: int
    model: object
    stats: object
    seq_len: int
    n_features: int

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        arr = np.asarray(list(history) if not isinstance(history, np.ndarray) else history,
                         dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < self.seq_len:
            pad = np.tile(arr[:1], (self.seq_len - arr.shape[0], 1))
            arr = np.concatenate([pad, arr], axis=0)
        arr = arr[-self.seq_len:, : self.n_features]
        if arr.shape[1] < self.n_features:
            pad_cols = np.zeros((self.seq_len, self.n_features - arr.shape[1]))
            arr = np.concatenate([arr, pad_cols], axis=1)
        # Normalise target column (index 0) using train stats; other features
        # pass through (RIN per-window centers them internally).
        arr[:, 0] = (arr[:, 0] - self.stats.target_mean) / self.stats.target_std
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_t = torch.from_numpy(arr).float().reshape(1, self.seq_len, self.n_features).to(device)
            yhat = self.model(x_t)
        yhat = yhat[0].cpu().numpy()
        yhat_dn = self.stats.denormalise_target(yhat)
        out: list[QuantileForecast] = []
        for h in range(min(horizon, yhat_dn.shape[0])):
            n_lv = min(len(levels), yhat_dn.shape[1])
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels[:n_lv]),
                values=tuple(float(v) for v in yhat_dn[h, :n_lv]),
            ))
        return out


@register("f13", description="F13 — RIN-augmented multivariate patch-TST (15 base + 8 forecast-error features)")
def _load_f13(seed: int) -> _F13Backend:
    from heimdall_forecaster.train.model import PatchTransformerQuantile

    d = checkpoint_dir("f13", seed, required_files=F13_CHECKPOINT_FILES)
    cfg = json.loads((d / "config.json").read_text())
    n_features = int(cfg.get("n_features", 23))
    model = PatchTransformerQuantile(
        n_features=n_features,
        seq_len=int(cfg["seq_len"]), horizon=int(cfg["horizon"]),
        n_quantiles=3, patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]), dropout=0.0,
        use_rin=True,
    )
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(d / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return _F13Backend(
        name="f13", seed=seed, model=model, stats=stats,
        seq_len=int(cfg["seq_len"]), n_features=n_features,
    )
