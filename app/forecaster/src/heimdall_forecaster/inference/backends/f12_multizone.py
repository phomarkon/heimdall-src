"""F12 — multi-zone transfer-learning inference backend.

Loads the fine-tuned ZoneEmbeddingPatchTransformer (DK1-tuned, F8B_FEATURES
input) and exposes the standard Forecaster.predict contract. Multivariate
input is expected: (T, len(F8B_FEATURES)).
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register


@dataclass
class F12MultiZone:
    name: str
    seed: int
    model: object
    stats: object
    n_features: int
    zone_idx: int

    def predict(
        self,
        history,
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        x = np.asarray(history if isinstance(history, np.ndarray) else list(history),
                       dtype=np.float64)
        seq_len = 96
        F = self.n_features
        if x.ndim == 1:
            warnings.warn(
                f"F12 was trained multivariate (T,{F}); broadcasting univariate "
                "history to all channels (degraded accuracy).",
                UserWarning, stacklevel=2,
            )
            if x.size < seq_len:
                x = np.concatenate([np.full(seq_len - x.size, x[0] if x.size else 0.0), x])
            x = x[-seq_len:]
            x_input = np.broadcast_to(x[:, None], (seq_len, F)).copy()
        else:
            assert x.shape[1] == F, f"F12 expects (T,{F}); got {x.shape}"
            if x.shape[0] < seq_len:
                pad = np.broadcast_to(x[0:1, :], (seq_len - x.shape[0], F))
                x = np.concatenate([pad, x], axis=0)
            x_input = x[-seq_len:]
        z = (x_input - self.stats.mean) / np.where(self.stats.std == 0.0, 1.0, self.stats.std)
        with torch.no_grad():
            zone = torch.full((1,), self.zone_idx, dtype=torch.long)
            yhat = self.model(torch.from_numpy(z).float().reshape(1, seq_len, F), zone)
        yhat = yhat[0].cpu().numpy()
        yhat_dn = yhat * self.stats.target_std + self.stats.target_mean
        out: list[QuantileForecast] = []
        n_q = yhat_dn.shape[1]
        for h in range(min(horizon, yhat_dn.shape[0])):
            n_levels = min(len(levels), n_q)
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels[:n_levels]),
                values=tuple(float(v) for v in yhat_dn[h, :n_levels]),
            ))
        return out


@register("f12", description="F12 — multi-zone pretrained transformer (Plan v2 Track C)")
def _load_f12(seed: int) -> F12MultiZone:
    from heimdall_forecaster.train.f12_multizone import (
        ZoneEmbeddingPatchTransformer, ZONES,
    )
    from heimdall_forecaster.train.model import PatchEmbedding
    d = checkpoint_dir("f12", seed)
    with open(d / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    n_features = len(stats.feature_names)
    model = ZoneEmbeddingPatchTransformer(
        n_features=1, n_zones=len(ZONES), seq_len=96, horizon=16,
        n_quantiles=3, patch_len=8, d_model=128, nhead=8, n_layers=6, dropout=0.0,
    )
    # Rebuild patch projection for the DK1 multivariate feature count
    # (mirrors the fine-tune step in train_f12).
    model.patch = PatchEmbedding(n_features, 8, 128)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return F12MultiZone(
        name="f12", seed=seed, model=model, stats=stats,
        n_features=n_features, zone_idx=ZONES.index("DK1"),
    )
