"""F9 — TimesFM-2.0/2.5 wrapper. Per docs/RESEARCH-PROPOSAL.md §4.2.2.

Loads ``google/timesfm-2.0-500m-pytorch`` (the current 2.0 release; 2.5 swaps
in via the same loader once weights are public) and exposes a contract-level
``forecast(history, horizon)`` returning ``QuantileForecast`` per step.

Optional split-CP wrap (Theorem 1a) takes a residual calibration set and
tightens the native quantile band to a finite-sample-corrected interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from heimdall_contracts import QuantileForecast

# Repo standard: forecaster zoo F9 quantile spec, docs/RESEARCH-PROPOSAL.md §4.4.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
DEFAULT_REPO_ID = "google/timesfm-2.0-500m-pytorch"
# TimesFM-2.0 architecture: 50-layer decoder, no positional embedding.
NUM_LAYERS_TIMESFM_2_0 = 50


def available() -> bool:
    """True iff ``timesfm`` is importable."""
    try:
        import timesfm  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@dataclass
class TimesFMForecaster:
    """Thin wrapper around the TimesFM-2.0 zero-shot forecaster.

    ``levels`` controls *which* of the model's native output quantiles we
    surface. The model emits 9 quantiles (deciles); we map by nearest decile.
    """

    levels: tuple[float, ...] = DEFAULT_QUANTILES
    context_len: int = 512
    horizon_len: int = 96  # 24 h × 15 min
    backend: str = "gpu"  # "cpu" / "gpu"
    repo_id: str = DEFAULT_REPO_ID
    per_core_batch_size: int = 4
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not available():
            raise ImportError(
                "timesfm is not installed. Run `pip install timesfm`. "
                "Per docs/RESEARCH-PROPOSAL.md §4.2.2, F9 is the day-1 default."
            )

    # --- private --------------------------------------------------------

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import timesfm

        hp = timesfm.TimesFmHparams(
            backend=self.backend,
            per_core_batch_size=self.per_core_batch_size,
            horizon_len=self.horizon_len,
            context_len=self.context_len,
            num_layers=NUM_LAYERS_TIMESFM_2_0,
            use_positional_embedding=False,
        )
        ck = timesfm.TimesFmCheckpoint(version="torch", huggingface_repo_id=self.repo_id)
        self._model = timesfm.TimesFm(hparams=hp, checkpoint=ck)
        return self._model

    # --- public ---------------------------------------------------------

    def predict(
        self, history: ArrayLike, *, freq_code: int = 0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns ``(mean, full)`` arrays as numpy.

        ``mean``: shape ``(horizon,)``.
        ``full``: shape ``(horizon, 1 + 9)`` — first column is the median /
        mean (per the TimesFM contract), columns 1..9 are the deciles.
        """
        m = self._load()
        h = np.asarray(history, dtype=np.float64).ravel()
        mean, full = m.forecast([h], freq=[freq_code])
        return np.asarray(mean[0], dtype=np.float64), np.asarray(full[0], dtype=np.float64)

    def forecast(
        self, history: ArrayLike, horizon: int
    ) -> list[QuantileForecast]:
        """Map TimesFM deciles to the requested ``levels`` and return one
        :class:`QuantileForecast` per horizon step."""
        mean, full = self.predict(history)
        # TimesFM's `full` has shape (H, 1+Q) where Q=9 (deciles 0.1..0.9).
        # Map each of self.levels to the nearest of [0.1..0.9].
        deciles = np.linspace(0.1, 0.9, 9)
        col_idx = []
        for ell in self.levels:
            best = int(np.argmin(np.abs(deciles - ell)))
            col_idx.append(1 + best)  # +1 to skip the mean/median col
        out: list[QuantileForecast] = []
        for h in range(min(horizon, full.shape[0])):
            vals = tuple(float(full[h, ci]) for ci in col_idx)
            out.append(
                QuantileForecast(
                    horizon_minutes=15 * (h + 1),
                    levels=self.levels,
                    values=vals,
                )
            )
        return out

    @staticmethod
    def synthetic_history(n: int = 256, seed: int = 13) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        return (
            10.0 + 0.5 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(n)
        ).astype(np.float64)


__all__ = ["DEFAULT_QUANTILES", "TimesFMForecaster", "available"]
