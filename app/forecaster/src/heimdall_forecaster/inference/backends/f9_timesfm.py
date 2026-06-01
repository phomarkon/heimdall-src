"""F9 — TimesFM-2.0 zero-shot.

Thin Forecaster-Protocol wrapper around the existing
``heimdall_forecaster.timesfm_wrapper.TimesFMForecaster``.  No
checkpoint hydration is needed: TimesFM weights load directly from
the public ``google/timesfm-2.0-200m-pytorch`` repo at first call.

If the ``timesfm`` Python package is not installed, the loader raises a
clear ImportError.  Operators who want F9 should run
``uv pip install timesfm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from heimdall_contracts import QuantileForecast

from ..registry import register


@dataclass
class F9TimesFM:
    name: str = "f9"
    seed: int = 42
    _wrapper: object = None

    def _ensure(self):
        if self._wrapper is None:
            from heimdall_forecaster.timesfm_wrapper import (
                TimesFMForecaster,
                available,
            )

            if not available():
                raise ImportError(
                    "F9 requires the `timesfm` package; run "
                    "`uv pip install timesfm` and retry."
                )
            self._wrapper = TimesFMForecaster()
        return self._wrapper

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        w = self._ensure()
        # The TimesFMForecaster wrapper has a forecast(history, horizon)
        # method that already returns QuantileForecast at the wrapper's
        # configured ``levels``.  Map levels at construction time if the
        # caller asks for something other than the default (0.1/0.5/0.9).
        if tuple(levels) != tuple(w.levels):
            from heimdall_forecaster.timesfm_wrapper import TimesFMForecaster

            w = TimesFMForecaster(levels=tuple(levels))
        return w.forecast(list(history), horizon)


@register(
    "f9",
    description=(
        "TimesFM-2.0 zero-shot — Google foundation forecaster (proposal §4.2.2 F9). "
        "Loads google/timesfm-2.0-200m-pytorch on first call."
    ),
)
def _load_f9(seed: int) -> F9TimesFM:
    return F9TimesFM(seed=seed)
