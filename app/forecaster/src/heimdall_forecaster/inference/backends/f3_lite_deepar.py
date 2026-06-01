"""F3-Lite — DeepAR LSTM Student-t backend."""
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

F3_LITE_CHECKPOINT_FILES = ("config.json", "model.pt", "stats.pkl")


@dataclass
class _F3LiteBackend:
    name: str
    seed: int
    model: object
    stats: object

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        from scipy.stats import t as student_t

        x = np.asarray(list(history), dtype=np.float64)
        seq_len = 96
        if x.size < seq_len:
            x = np.concatenate([np.full(seq_len - x.size, x[0] if x.size else 0.0), x])
        x = x[-seq_len:]
        z = (x - self.stats.target_mean) / self.stats.target_std
        with torch.no_grad():
            mu, sigma, nu = self.model(torch.from_numpy(z).float().reshape(1, seq_len, 1))
        mu = mu[0].cpu().numpy(); sigma = sigma[0].cpu().numpy(); nu = nu[0].cpu().numpy()
        out: list[QuantileForecast] = []
        for h in range(min(horizon, mu.shape[0])):
            qs = []
            for lv in levels:
                z_q = student_t.ppf(lv, df=float(nu[h]))
                q_norm = float(mu[h]) + float(sigma[h]) * z_q
                qs.append(self.stats.denormalise_target(np.array([q_norm]))[0])
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=tuple(float(v) for v in qs),
            ))
        return out


@register("f3_lite", description="F3-Lite DeepAR LSTM Student-t (appendix)")
def _load_f3_lite(seed: int) -> _F3LiteBackend:
    from heimdall_forecaster.train.f3_deepar import DeepARLite

    d = checkpoint_dir("f3_lite", seed, required_files=F3_LITE_CHECKPOINT_FILES)
    cfg = json.loads((d / "config.json").read_text())
    model = DeepARLite(hidden=int(cfg.get("hidden", 64)), horizon=int(cfg.get("horizon", 16)))
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state); model.eval()
    with open(d / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return _F3LiteBackend(name="f3_lite", seed=seed, model=model, stats=stats)
