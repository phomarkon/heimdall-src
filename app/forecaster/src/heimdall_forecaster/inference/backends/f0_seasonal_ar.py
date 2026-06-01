"""F0 — naive seasonal AR(24).  No checkpoint required."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from heimdall_contracts import QuantileForecast

from ..registry import register

# Z-scores for 10/50/90 — Gaussian residual band (a coarse but documented
# approximation; the verifier wraps F0 in split-CP for the real coverage).
_Z = {0.1: -1.28155, 0.5: 0.0, 0.9: 1.28155}


@dataclass
class F0SeasonalAR:
    name: str = "f0"
    seed: int = 42

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        x = np.asarray(list(history), dtype=np.float64)
        out: list[QuantileForecast] = []
        scale = float(np.std(x[-96:])) if x.size >= 96 else 50.0
        for h in range(horizon):
            point = float(x[-96 + (h % 96)]) if x.size >= 96 else (float(x[-1]) if x.size else 0.0)
            qs = tuple(point + scale * _Z.get(level, 0.0) for level in levels)
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=qs,
            ))
        return out


@register("f0", description="Naive seasonal AR(24) — proposal F0 baseline; cheap, no checkpoint")
def _load_f0(seed: int) -> F0SeasonalAR:
    return F0SeasonalAR(seed=seed)
