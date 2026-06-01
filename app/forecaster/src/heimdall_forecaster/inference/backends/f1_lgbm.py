"""F1 — Quantile LightGBM inference backend.

Loads per-(horizon,quantile) LightGBM boosters from the pickled string cache
produced by ``f1_lgbm.train_f1()``.  Flattens input windows identically to the
training pipeline.
"""

from __future__ import annotations

import pickle
from typing import Iterable

import lightgbm as lgb
import numpy as np

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register

HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
F1_CHECKPOINT_FILES = ("boosters.pkl", "stats.pkl")


class F1LightGBM:
    name: str = "f1_lgbm"
    seed: int

    def __init__(self, seed: int):
        self.seed = seed
        d = checkpoint_dir("f1_lgbm", seed, required_files=F1_CHECKPOINT_FILES)
        with open(d / "boosters.pkl", "rb") as fh:
            str_cache = pickle.load(fh)
        # str_cache maps "hH_qQQ" → model string; reconstruct boosters.
        self._boosters: dict[tuple[int, float], lgb.Booster] = {}
        for key, model_str in str_cache.items():
            # key = "h0_q10", "h15_q90", etc.
            h_str = key.split("_")[0][1:]
            q_str = key.split("_")[1][1:]
            h_idx = int(h_str)
            q_level = int(q_str) / 100.0
            self._boosters[(h_idx, q_level)] = lgb.Booster(model_str=model_str)
        # Load train-stat normalisation (training uses normalised X, denormalised Y).
        stats_path = d / "stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as fh:
                self._stats = pickle.load(fh)
        else:
            self._stats = None

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = HORIZON,
        levels: tuple[float, ...] = QUANTILES,
    ) -> list[QuantileForecast]:
        x = np.asarray(list(history), dtype=np.float32)[-96:]  # ensure seq_len
        if x.size < 96:
            raise ValueError("history too short (need ≥96)")
        # Normalise the input window with the SAME stats used during training.
        # Trainer feeds make_windows(...) → normalised X. Booster targets are
        # already denormalised in real units, so no post-norm of the output.
        if self._stats is not None:
            x_3d = x.reshape(1, -1, 1)  # (1, T, 1) — single feature column
            x = self._stats.normalise(x_3d).reshape(-1).astype(np.float32)
        flat = x.reshape(1, -1).astype(np.float32)  # (1, 96)
        out: list[QuantileForecast] = []
        for h in range(min(horizon, HORIZON)):
            qs = []
            for level in levels:
                booster = self._boosters.get((h, level))
                if booster is None:
                    qs.append(float(np.mean(history[-24:])) if history else 0.0)
                else:
                    qs.append(float(booster.predict(flat).item()))
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=tuple(qs),
            ))
        return out


@register("f1_lgbm", description="Quantile LightGBM — per-(h,q) boosters on 96-lag window")
def _load_f1_lgbm(seed: int) -> F1LightGBM:
    return F1LightGBM(seed)
