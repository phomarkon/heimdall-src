"""F8 — patch-transformer + online ACI calibrator.

F8 was trained multivariate (proposal §4.4): the patch-TST input is a
(T × 3) tensor of (imbalance_price_dkk_mwh_15min, load_actual_mw,
da_price_dkk_mwh).  The Theorem-1b ACI calibration sits *around* this
backbone at inference time, not inside the model file itself; the
calibrator service handles the alpha-update loop.

API contract for Tim's agent-runner:

  - If ``history`` is 1-D (length T): the backend broadcasts it onto
    all 3 input channels.  Output is *computed* but the load/DA
    channels carry no real signal; expect degraded accuracy.  A
    UserWarning is emitted.
  - If ``history`` is 2-D and shape (T, 3): used as-is.  Column order
    is (imbalance, load, da).

The univariate fallback is intentional — it keeps the
``Forecaster.predict`` Protocol uniform across F7/F8 so the inference
service doesn't need a special branch.  Production callers should
pass the 2-D multivariate input.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register
from .f7_patch_tst import PATCH_TST_CHECKPOINT_FILES


F8_FEATURE_NAMES = ("imbalance_price_dkk_mwh_15min", "load_actual_mw", "da_price_dkk_mwh")


@dataclass
class F8PatchTST:
    name: str
    seed: int
    model: object
    stats: object
    n_features: int = 3

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        x = np.asarray(list(history) if not isinstance(history, np.ndarray) else history,
                       dtype=np.float64)
        seq_len = self.model.seq_len
        F = self.n_features

        # Reshape / broadcast logic.
        if x.ndim == 1:
            if F > 1:
                warnings.warn(
                    "F8 was trained multivariate (T,3); univariate history is "
                    "being broadcast to all 3 channels which will degrade "
                    "accuracy.  Pass a (T,3) history with columns "
                    f"{F8_FEATURE_NAMES} for production use.",
                    UserWarning,
                    stacklevel=2,
                )
            if x.size < seq_len:
                x = np.concatenate([np.full(seq_len - x.size, x[0] if x.size else 0.0), x])
            x = x[-seq_len:]
            x_input = np.broadcast_to(x[:, None], (seq_len, F)).copy()
        elif x.ndim == 2:
            assert x.shape[1] == F, (
                f"F8 expects (T, {F}); got shape {x.shape}"
            )
            if x.shape[0] < seq_len:
                pad = np.broadcast_to(x[0:1, :], (seq_len - x.shape[0], F))
                x = np.concatenate([pad, x], axis=0)
            x_input = x[-seq_len:]
        else:
            raise ValueError(f"history must be 1-D or 2-D; got ndim={x.ndim}")

        # Normalise per-channel using the stats object.  The stats object
        # carries channel-wise mean/std for multivariate variants.
        # WindowStats (F8b/c/d/F12) carries `.mean` / `.std` per feature;
        # older multivariate stats objects used `feature_means`/`feature_stds`.
        if hasattr(self.stats, "mean") and hasattr(self.stats, "std"):
            means = np.asarray(self.stats.mean)
            stds = np.asarray(self.stats.std)
            stds = np.where(stds == 0.0, 1.0, stds)
            z = (x_input - means) / stds
        elif hasattr(self.stats, "feature_means"):
            means = np.asarray(self.stats.feature_means)
            stds = np.asarray(self.stats.feature_stds)
            z = (x_input - means) / stds
        else:
            z = (x_input - self.stats.target_mean) / self.stats.target_std

        with torch.no_grad():
            yhat = self.model(torch.from_numpy(z).float().reshape(1, seq_len, F))
        yhat = yhat[0].cpu().numpy()
        yhat_dn = self.stats.denormalise_target(yhat)
        n_q = yhat_dn.shape[1]
        out: list[QuantileForecast] = []
        for h in range(min(horizon, yhat_dn.shape[0])):
            n_levels = min(len(levels), n_q)
            out.append(
                QuantileForecast(
                    horizon_minutes=15 * (h + 1),
                    levels=tuple(levels[:n_levels]),
                    values=tuple(float(v) for v in yhat_dn[h, :n_levels]),
                )
            )
        return out


def _load_f8(name: str, seed: int) -> F8PatchTST:
    from heimdall_forecaster.train.model import PatchTransformerQuantile

    d = checkpoint_dir(name, seed, required_files=PATCH_TST_CHECKPOINT_FILES)
    cfg = json.loads((d / "config.json").read_text())
    n_features = int(cfg.get("n_features", 3))
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    use_rin = bool(cfg.get("use_rin", False)) or "rin_gamma" in state or "rin_beta" in state
    model = PatchTransformerQuantile(
        n_features=n_features,
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["horizon"]),
        n_quantiles=3,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=0.0,
        use_rin=use_rin,
    )
    model.load_state_dict(state)
    model.eval()
    with open(d / "stats.pkl", "rb") as fh:
        stats_obj = pickle.load(fh)
    return F8PatchTST(name=name, seed=seed, model=model, stats=stats_obj,
                       n_features=n_features)


@register(
    "f8",
    description=(
        "patch-TST (multivariate, 3 features) + online ACI — Theorem 1b "
        "focal default; survives regime shift.  See backend for input "
        "shape contract (univariate broadcast OK with warning)."
    ),
)
def _load_f8_default(seed: int) -> F8PatchTST:
    return _load_f8("f8", seed)


@register("f8b", description="F8b — rich-feature patch-TST (Plan v2 Track D.1)")
def _load_f8b(seed: int) -> F8PatchTST:
    return _load_f8("f8b", seed)


@register("f8c", description="F8c — kitchen-sink patch-TST (Plan v2 Track D.1)")
def _load_f8c(seed: int) -> F8PatchTST:
    return _load_f8("f8c", seed)


@register("f8d", description="F8d — XAI-lean patch-TST (Plan v2 Track D.4)")
def _load_f8d(seed: int) -> F8PatchTST:
    return _load_f8("f8d", seed)


@register("f8e", description="F8e — F8b + mFRR volumes (audit gap)")
def _load_f8e(seed: int) -> F8PatchTST:
    return _load_f8("f8e", seed)
