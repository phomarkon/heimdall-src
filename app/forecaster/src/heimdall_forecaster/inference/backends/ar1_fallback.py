"""AR(1) fallback — wraps the existing `AR1FallbackForecaster` in the
unified `Forecaster` protocol.  Used when no F-zoo checkpoint is
available (CI, tests, fresh-clone smoke runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from heimdall_contracts import QuantileForecast
from heimdall_forecaster.fallback import AR1FallbackForecaster

from ..registry import register


@dataclass
class AR1Wrapper:
    name: str = "ar1"
    seed: int = 42

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        f = AR1FallbackForecaster(levels=tuple(levels))
        return f.fit_predict_quantiles(list(history), horizon=horizon)


@register("ar1", description="AR(1) Gaussian fallback — CI-friendly, no checkpoint needed")
def _load_ar1(seed: int) -> AR1Wrapper:
    return AR1Wrapper(seed=seed)
