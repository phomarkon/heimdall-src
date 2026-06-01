"""F2 — Bayesian Linear Regression inference backend.

Loads per-horizon-step BayesianRidge regressors from the pickled dict
produced by ``f2_blr.train_f2()``.  Flattens input windows identically to
the training pipeline.
"""

from __future__ import annotations

import pickle
from typing import Iterable

import numpy as np
from scipy.stats import norm

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register

HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
F2_CHECKPOINT_FILES = ("regressors.pkl", "stats.pkl")


class F2BLR:
    name: str = "f2_blr"
    seed: int

    def __init__(self, seed: int):
        self.seed = seed
        d = checkpoint_dir("f2_blr", seed, required_files=F2_CHECKPOINT_FILES)
        with open(d / "regressors.pkl", "rb") as fh:
            self._regs = pickle.load(fh)  # {"h0": BayesianRidge, ...}
        # F2 trains on normalised X with denormalised Y targets (see
        # train/f2_blr.py:69-70). Posterior std comes back in real units;
        # mu comes back in real units. Apply the same input normalisation.
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
        x = np.asarray(list(history), dtype=np.float64)[-96:]
        if x.size < 96:
            raise ValueError("history too short (need ≥96)")
        if self._stats is not None:
            x_3d = x.reshape(1, -1, 1).astype(np.float32)
            x = self._stats.normalise(x_3d).reshape(-1).astype(np.float64)
        flat = x.reshape(1, -1).astype(np.float64)
        out: list[QuantileForecast] = []
        for h in range(min(horizon, HORIZON)):
            reg = self._regs.get(f"h{h}")
            if reg is None:
                fallback = float(np.mean(x[-24:]))
                qs = tuple(fallback for _ in levels)
            else:
                mu, sigma = reg.predict(flat, return_std=True)
                qs = tuple(float(mu[0] + norm.ppf(q) * sigma[0]) for q in levels)
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=qs,
            ))
        return out


@register("f2_blr", description="Bayesian Linear Regression — Gaussian posterior per horizon step")
def _load_f2_blr(seed: int) -> F2BLR:
    return F2BLR(seed)
