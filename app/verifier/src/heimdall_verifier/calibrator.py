"""Build a ``ConformalInterval`` from a trained forecaster + ACI calibrator.

Per docs/RESEARCH-PROPOSAL.md §4.4 + §4.5: the focal verifier consumes a
``ConformalInterval`` produced by either split-CP (Theorem 1a) or online ACI
(Theorem 1b). This module wires the two together so the verifier never sees
synthetic residuals — it always reads from the trained forecaster's val
residuals (persisted in MLflow checkpoints under ``models/forecaster/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heimdall_contracts import ConformalInterval
from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.split_cp import SplitConformal


@dataclass
class CalibratedForecaster:
    """Pairs a forecaster's q50 point predictions with an ACI/split-CP calibrator.

    The calibrator is *seeded* from the forecaster's val residuals — that is the
    canonical Theorem 1b warm-start. ACI then updates online as new residuals
    arrive in the live trading loop.
    """

    aci: AdaptiveConformalInference
    horizon_minutes: int = 15

    @classmethod
    def from_val_preds(
        cls,
        val_preds_path: Path,
        *,
        alpha: float = 0.1,
        gamma: float = 0.05,
        horizon_step: int = 0,
        horizon_minutes: int = 15,
    ) -> "CalibratedForecaster":
        """Load val_preds.npz produced by ``train_model`` and warm-start ACI.

        ``val_preds_path``: path to ``models/forecaster/<f>/seed-<n>/val_preds.npz``.
        """
        z = np.load(val_preds_path)
        preds = z["preds"]  # (N, H, Q)
        targets = z["targets"]  # (N, H)
        q50 = preds[:, horizon_step, preds.shape[-1] // 2]
        y = targets[:, horizon_step]
        scores = np.abs(y - q50)
        aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)
        aci.warm_start(scores)
        return cls(aci=aci, horizon_minutes=horizon_minutes)

    def interval(self, point_pred: float) -> ConformalInterval:
        """Convert a point prediction into a ``ConformalInterval`` via ACI."""
        q = self.aci.quantile()
        if not np.isfinite(q):
            # Fall back to a wide but bounded interval; the verifier will then
            # almost certainly reject. Logged via ACI's diagnostics.
            q = float(self.aci._scores[-1] if self.aci._scores else 1e6)  # type: ignore[arg-type]
        return ConformalInterval(
            horizon_minutes=self.horizon_minutes,
            alpha=self.aci.alpha,
            lower=float(point_pred - q),
            upper=float(point_pred + q),
            method="aci",
        )

    def update(self, realised: float, point_pred: float) -> None:
        """Online update: feed the absolute residual to ACI."""
        self.aci.update(abs(realised - point_pred))


def split_cp_interval(
    val_preds_path: Path,
    point_pred: float,
    *,
    alpha: float = 0.1,
    horizon_step: int = 0,
    horizon_minutes: int = 15,
) -> ConformalInterval:
    """Theorem 1a (finite-sample) interval from val residuals."""
    z = np.load(val_preds_path)
    preds = z["preds"]
    targets = z["targets"]
    q50 = preds[:, horizon_step, preds.shape[-1] // 2]
    y = targets[:, horizon_step]
    scores = np.abs(y - q50)
    cp = SplitConformal.fit(scores, alpha=alpha)
    return ConformalInterval(
        horizon_minutes=horizon_minutes,
        alpha=alpha,
        lower=float(point_pred - cp.quantile),
        upper=float(point_pred + cp.quantile),
        method="split_cp",
    )


__all__ = ["CalibratedForecaster", "split_cp_interval"]
