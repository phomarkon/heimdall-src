"""AR(1) residual-bootstrap forecaster.

Plays two roles:
  1. F0 in the proposal's zoo (naive sanity baseline, docs/RESEARCH-PROPOSAL.md
     §4.2.2 row F0 — adapted to AR(1) here for simplicity; AR(24) is the
     literal F0 spec and will be added in Week 1 Day 4).
  2. CPU-only smoke fallback when TimesFM is not installed; CI must remain
     green without GPU/torch (peer agent owns the B200 today).

The forecaster fits an AR(1) by OLS on a univariate series, then bootstraps
residuals to draw quantile forecasts. Lightweight, deterministic with a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from heimdall_contracts import QuantileForecast


@dataclass
class AR1FallbackForecaster:
    """Tiny, deterministic, CPU-only forecaster used for smoke tests."""

    levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 13

    def fit_predict_quantiles(
        self,
        history: ArrayLike,
        horizon: int,
        *,
        n_bootstrap: int = 500,
    ) -> list[QuantileForecast]:
        y = np.asarray(history, dtype=np.float64).ravel()
        if y.size < 8:
            raise ValueError("AR(1) needs at least 8 observations")

        # OLS estimate of phi for y_t = phi * y_{t-1} + eps_t.
        x = y[:-1]
        z = y[1:]
        phi = float(np.dot(x, z) / np.dot(x, x))
        residuals = z - phi * x

        rng = np.random.default_rng(self.seed)
        out: list[QuantileForecast] = []
        last = y[-1]
        # Vectorised paths: shape (n_bootstrap, horizon).
        paths = np.empty((n_bootstrap, horizon), dtype=np.float64)
        for h in range(horizon):
            eps = rng.choice(residuals, size=n_bootstrap, replace=True)
            last_path = paths[:, h - 1] if h > 0 else np.full(n_bootstrap, last)
            paths[:, h] = phi * last_path + eps

        for h in range(horizon):
            qvals = np.quantile(paths[:, h], self.levels)
            out.append(
                QuantileForecast(
                    horizon_minutes=15 * (h + 1),
                    levels=self.levels,
                    values=tuple(float(v) for v in qvals),
                )
            )
        return out


def synthetic_ar1(n: int, phi: float = 0.7, sigma: float = 1.0, seed: int = 13) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + sigma * rng.standard_normal()
    return y
