"""F5/F6 neural-process inference backends."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast
from heimdall_forecaster.train.model import AttentiveNPForecaster, ConvCNPForecaster

from ..hf_hydrator import checkpoint_dir
from ..registry import register

REQUIRED_FILES = ("config.json", "model.pt", "stats.pkl")


@dataclass
class NeuralProcessBackend:
    name: str
    seed: int
    model: torch.nn.Module
    stats: object

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        x = np.asarray(list(history), dtype=np.float64)
        seq_len = self.model.seq_len
        if x.size < seq_len:
            x = np.concatenate([np.full(seq_len - x.size, x[0] if x.size else 0.0), x])
        x = x[-seq_len:]
        z = (x - self.stats.target_mean) / self.stats.target_std
        with torch.no_grad():
            mu, sigma = self.model(torch.from_numpy(z).float().reshape(1, seq_len, 1))
        mu_np = self.stats.denormalise_target(mu[0].cpu().numpy())
        sigma_np = sigma[0].cpu().numpy() * self.stats.target_std
        zscores = np.asarray([NormalDist().inv_cdf(q) for q in levels], dtype=np.float64)
        out: list[QuantileForecast] = []
        for h in range(min(horizon, mu_np.shape[0])):
            vals = np.sort(mu_np[h] + sigma_np[h] * zscores)
            out.append(
                QuantileForecast(
                    horizon_minutes=15 * (h + 1),
                    levels=tuple(levels),
                    values=tuple(float(v) for v in vals),
                )
            )
        return out


def _load_np(name: str, seed: int) -> NeuralProcessBackend:
    d = checkpoint_dir(name, seed, required_files=REQUIRED_FILES)
    cfg = json.loads((d / "config.json").read_text())
    model_cls = AttentiveNPForecaster if name == "f6" else ConvCNPForecaster
    kwargs = {
        "n_features": int(cfg["n_features"]),
        "seq_len": int(cfg["seq_len"]),
        "horizon": int(cfg["horizon"]),
        "d_model": int(cfg["d_model"]),
        "n_layers": int(cfg["n_layers"]),
        "dropout": 0.0,
    }
    if name == "f6":
        kwargs["nhead"] = int(cfg["nhead"])
    model = model_cls(**kwargs)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(d / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return NeuralProcessBackend(name=name, seed=seed, model=model, stats=stats)


@register("f5", description="ConvCNP neural-process forecaster with Gaussian UQ")
def _load_f5(seed: int) -> NeuralProcessBackend:
    return _load_np("f5", seed)


@register("f6", description="Attentive Neural Process forecaster with Gaussian UQ")
def _load_f6(seed: int) -> NeuralProcessBackend:
    return _load_np("f6", seed)
