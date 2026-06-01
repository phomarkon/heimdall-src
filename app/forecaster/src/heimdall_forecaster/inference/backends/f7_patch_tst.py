"""F7 — patch-transformer + per-quantile heads + split-CP.

Loads a trained checkpoint from
``models/forecaster/f7/seed-<seed>/{config.json,model.pt,stats.pkl}``.
Hydrates from HuggingFace if the local directory is empty.
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

PATCH_TST_CHECKPOINT_FILES = ("config.json", "model.pt", "stats.pkl")


@dataclass
class _PatchTSTBase:
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
        x = np.asarray(list(history), dtype=np.float64)
        seq_len = self.model.seq_len
        if x.size < seq_len:
            x = np.concatenate([np.full(seq_len - x.size, x[0] if x.size else 0.0), x])
        x = x[-seq_len:]
        z = (x - self.stats.target_mean) / self.stats.target_std
        with torch.no_grad():
            yhat = self.model(torch.from_numpy(z).float().reshape(1, seq_len, 1))
        yhat = yhat[0].cpu().numpy()
        yhat_dn = self.stats.denormalise_target(yhat)  # (H, 3) — q10/q50/q90
        n_q = yhat_dn.shape[1]
        out: list[QuantileForecast] = []
        for h in range(min(horizon, yhat_dn.shape[0])):
            n_levels = min(len(levels), n_q)
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels[:n_levels]),
                values=tuple(float(v) for v in yhat_dn[h, :n_levels]),
            ))
        return out


def _load_patch_tst(name: str, seed: int) -> _PatchTSTBase:
    from heimdall_forecaster.train.model import PatchTransformerQuantile

    d = checkpoint_dir(name, seed, required_files=PATCH_TST_CHECKPOINT_FILES)
    cfg = json.loads((d / "config.json").read_text())
    model = PatchTransformerQuantile(
        n_features=int(cfg["n_features"]),
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["horizon"]),
        n_quantiles=3,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=0.0,
    )
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(d / "stats.pkl", "rb") as fh:
        stats_obj = pickle.load(fh)
    return _PatchTSTBase(name=name, seed=seed, model=model, stats=stats_obj)


@register("f7", description="patch-TST + quantile heads + split-CP — focal verifier default")
def _load_f7(seed: int) -> _PatchTSTBase:
    return _load_patch_tst("f7", seed)


# Architecture sweeps (kept for the AF1/AF2 ablations).
@register("f7_patch4", description="F7 with patch_len=4 (AF1)")
def _load_f7_p4(seed: int) -> _PatchTSTBase:
    return _load_patch_tst("f7_patch4", seed)


@register("f7_patch8", description="F7 with patch_len=8 (AF1)")
def _load_f7_p8(seed: int) -> _PatchTSTBase:
    return _load_patch_tst("f7_patch8", seed)


@register("f7_patch16", description="F7 with patch_len=16 (AF1)")
def _load_f7_p16(seed: int) -> _PatchTSTBase:
    return _load_patch_tst("f7_patch16", seed)


@register("f7_optuna", description="F7 with Optuna-tuned HPs (50-trial TPE study)")
def _load_f7_optuna(seed: int) -> _PatchTSTBase:
    return _load_patch_tst("f7_optuna", seed)
