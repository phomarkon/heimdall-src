"""F10 — Chronos-Bolt zero-shot + optional split-CP wrap.

Per docs/RESEARCH-PROPOSAL.md §4.2.2.  Chronos-Bolt is the Amazon AWS
T5-style time-series foundation model (Ansari et al. 2024).  We use the
public ``amazon/chronos-bolt-*`` checkpoints via the
``chronos-forecasting`` package — no Heimdall-side training required.

Default checkpoint is ``chronos-bolt-base`` (~205 M params); set the
``HEIMDALL_CHRONOS_MODEL`` env var to override (``-tiny``, ``-mini``,
``-small``, ``-base``).  Loaded lazily at first ``predict()`` call so
the heavy transformers stack doesn't pay for non-F10 deployments.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast

from ..registry import register

DEFAULT_MODEL = os.environ.get(
    "HEIMDALL_CHRONOS_MODEL", "amazon/chronos-bolt-base"
)


@dataclass
class F10ChronosBolt:
    name: str = "f10"
    seed: int = 42
    model_id: str = DEFAULT_MODEL
    _pipe: object = None
    _fallback: object = None

    def _ensure_pipe(self):
        if self._fallback is not None:
            return None
        if self._pipe is None:
            try:
                from chronos import BaseChronosPipeline
            except ModuleNotFoundError:
                warnings.warn(
                    "F10 Chronos-Bolt requires the optional `chronos-forecasting` package; "
                    "falling back to F9 TimesFM for this process.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                from .f9_timesfm import F9TimesFM

                self._fallback = F9TimesFM(seed=self.seed)
                return None
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=device_map, torch_dtype=torch.float32,
            )
        return self._pipe

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        pipe = self._ensure_pipe()
        if pipe is None:
            return self._fallback.predict(history, horizon=horizon, levels=levels)
        ctx = torch.tensor([list(history)], dtype=torch.float32)
        q, _mean = pipe.predict_quantiles(
            inputs=ctx, prediction_length=horizon, quantile_levels=list(levels),
        )
        # q shape: (1, horizon, len(levels))
        q = q[0].cpu().numpy()
        out: list[QuantileForecast] = []
        for h in range(horizon):
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=tuple(float(v) for v in q[h]),
            ))
        return out


@register("f10", description="Chronos-Bolt zero-shot — Amazon AWS T5-style foundation forecaster (proposal §4.2.2 F10)")
def _load_f10(seed: int) -> F10ChronosBolt:
    return F10ChronosBolt(seed=seed)
