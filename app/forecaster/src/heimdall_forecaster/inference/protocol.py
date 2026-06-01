"""Forecaster Protocol — the single interface every backend honours.

Per SOLID:
- *Liskov*: every concrete backend (`F0SeasonalAR`, `F7PatchTST`, ...) is
  a drop-in for a `Forecaster`-typed parameter.
- *Interface segregation*: the protocol exposes one capability — predict
  a multi-quantile forecast over a horizon — and nothing else. Training,
  fitting, calibration are strictly *not* in this interface.

The protocol is consumed by:
- the FastAPI service (`apps/forecaster/.../service.py`),
- the focal-orchestrator (`apps/focal-orchestrator/.../pipeline.py`),
- the agent-runner (`apps/agent-runner/.../reflex.py`),
- and any ablation cell that needs a swappable forecaster.

Keep this file small.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from heimdall_contracts import QuantileForecast


@runtime_checkable
class Forecaster(Protocol):
    """Stable cross-app surface for any zoo member."""

    name: str
    seed: int

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        """Return one ``QuantileForecast`` per horizon step.

        ``levels`` are quantile probabilities in (0, 1) ascending. Backends
        that natively emit only three quantiles (e.g. patch-TST + split-CP)
        should clip / pad to the requested levels rather than raise.
        """
        ...


__all__ = ["Forecaster"]
