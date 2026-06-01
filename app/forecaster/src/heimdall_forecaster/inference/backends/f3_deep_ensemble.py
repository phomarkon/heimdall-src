"""F3 — patch-TST deep ensemble.

Loads all 5 F7 patch-TST seed checkpoints from
``models/forecaster/f7/seed-{13,42,137,1729,31415}/`` (auto-pulled
from HF if missing) and averages their per-quantile predictions at
inference.  This is the redefinition recorded in ADR-0006 and is the
strongest trained F-zoo entry on val pinball (259.3 DKK).

Note: ``seed`` parameter is informational only here --- F3 always
averages the same 5 frozen seeds.  We accept any ``seed`` value so the
registry's ``LoaderFn`` shape stays uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from heimdall_contracts import QuantileForecast

from ..registry import register
from .f7_patch_tst import _load_patch_tst, _PatchTSTBase


_F3_SEEDS = (13, 42, 137, 1729, 31415)


@dataclass
class F3DeepEnsemble:
    name: str = "f3_ensemble"
    seed: int = 42
    members: list[_PatchTSTBase] = field(default_factory=list)

    def _ensure(self):
        if not self.members:
            self.members = [_load_patch_tst("f7", s) for s in _F3_SEEDS]
        return self.members

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        members = self._ensure()
        # Each member returns list[QuantileForecast]; stack their values
        # by horizon × quantile and average.
        per_member = [m.predict(history, horizon=horizon, levels=levels) for m in members]
        H = len(per_member[0])
        n_q = len(per_member[0][0].values)
        agg = np.zeros((H, n_q), dtype=np.float64)
        for outs in per_member:
            for h in range(H):
                for qi in range(n_q):
                    agg[h, qi] += outs[h].values[qi]
        agg /= len(members)
        out: list[QuantileForecast] = []
        for h in range(H):
            out.append(QuantileForecast(
                horizon_minutes=per_member[0][h].horizon_minutes,
                levels=per_member[0][h].levels,
                values=tuple(float(v) for v in agg[h]),
            ))
        return out


@register(
    "f3_ensemble",
    description=(
        "patch-TST deep ensemble (5-seed F7 average) — best trained zoo entry "
        "(val pinball 259.3, ADR-0006)"
    ),
)
def _load_f3_ensemble(seed: int) -> F3DeepEnsemble:
    return F3DeepEnsemble(seed=seed)


# Alias: "f3" maps to the deep ensemble per ADR-0006.
@register(
    "f3",
    description="alias of f3_ensemble (proposal §4.2.2 F3 redefinition; ADR-0006)",
)
def _load_f3_alias(seed: int) -> F3DeepEnsemble:
    return F3DeepEnsemble(seed=seed)
