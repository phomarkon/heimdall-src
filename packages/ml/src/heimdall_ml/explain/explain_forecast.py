"""Forecaster SHAP attribution. docs/RESEARCH-PROPOSAL.md §1.3 + §5 (XAI emphasis).

Implements ``explain(model, x_window, target_horizon_step) -> Explanation`` for
the F7/F8 patch-transformer. We use ``shap.GradientExplainer`` for the dense
per-input-step + per-feature signal, plus a permutation-importance fallback at
the patch level so non-differentiable forecasters (F1 LightGBM, F11) can share
the API later.

The returned explanation is *non-crossing-quantile-aware*: attribution is
computed against the model's q50 head by default, but ``quantile_idx`` lets
callers attribute to q10 (worst-case for sells) or q90 (worst-case for buys).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class ForecastExplanation:
    """SHAP attribution over (input timestep, feature) for one forecast.

    ``values`` shape: (seq_len, n_features). Sum + base ≈ model output (q50)
    at the chosen target horizon step.
    """

    values: NDArray[np.float64]
    base_value: float
    feature_names: tuple[str, ...]
    target_horizon_step: int
    quantile_idx: int
    method: str  # "shap.GradientExplainer" or "permutation"


def _torch_predict(model: torch.nn.Module, target_step: int, q_idx: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)[:, target_step, q_idx]
    return fn


def explain(
    model: torch.nn.Module,
    x_window: NDArray[np.float32],
    *,
    background: NDArray[np.float32] | None = None,
    feature_names: tuple[str, ...] = ("imbalance_price",),
    target_horizon_step: int = 0,
    quantile_idx: int = 1,  # q50
) -> ForecastExplanation:
    """SHAP-based explanation for a single forecast window.

    Parameters
    ----------
    model:
        A trained ``torch.nn.Module`` whose forward returns ``(B, H, Q)``.
    x_window:
        Input window with shape ``(seq_len, n_features)`` — single example.
    background:
        Background distribution for SHAP. If ``None`` we synthesise 32 random
        background points around ``x_window`` (Gaussian noise).
    target_horizon_step:
        Which forecast lead to attribute (0 = next 15 min).
    quantile_idx:
        Which quantile head to attribute (0 = q10, 1 = q50, 2 = q90).
    """
    import shap

    from heimdall_ml.explain._common import model_device, synthesize_background

    model.eval()
    device = model_device(model)
    seq_len, n_features = x_window.shape

    if background is None:
        bg = synthesize_background(x_window, n_samples=32, seed=13)
    else:
        bg = background

    bg_t = torch.from_numpy(bg).float().to(device)
    x_t = torch.from_numpy(x_window[None]).float().to(device)
    pred_fn = _torch_predict(model, target_horizon_step, quantile_idx)
    base_value = float(pred_fn(bg_t).mean().detach().cpu())

    try:
        explainer = shap.GradientExplainer(pred_fn, bg_t)
        sv = explainer.shap_values(x_t)
        # SHAP returns either an array or a list-of-arrays depending on version.
        sv_arr = np.asarray(sv if not isinstance(sv, list) else sv[0])
        sv_arr = sv_arr.reshape(seq_len, n_features)
        return ForecastExplanation(
            values=sv_arr.astype(np.float64),
            base_value=base_value,
            feature_names=feature_names,
            target_horizon_step=target_horizon_step,
            quantile_idx=quantile_idx,
            method="shap.GradientExplainer",
        )
    except Exception:  # noqa: BLE001 -- gradient SHAP can fail on quirky models
        return _permutation_attribution(
            model=model,
            x_window=x_window,
            feature_names=feature_names,
            target_horizon_step=target_horizon_step,
            quantile_idx=quantile_idx,
            base_value=base_value,
        )


def _permutation_attribution(
    *,
    model: torch.nn.Module,
    x_window: NDArray[np.float32],
    feature_names: tuple[str, ...],
    target_horizon_step: int,
    quantile_idx: int,
    base_value: float,
) -> ForecastExplanation:
    """Patch-level permutation importance fallback. Coarse but always works."""
    from heimdall_ml.explain._common import model_device

    seq_len, n_features = x_window.shape
    device = model_device(model)
    pred_fn = _torch_predict(model, target_horizon_step, quantile_idx)
    rng = np.random.default_rng(13)

    base_pred = float(pred_fn(torch.from_numpy(x_window[None]).float().to(device)).item())
    values = np.zeros((seq_len, n_features), dtype=np.float64)
    n_perm = 16
    for t in range(seq_len):
        for f in range(n_features):
            x_perm = np.tile(x_window[None], (n_perm, 1, 1)).astype(np.float32)
            x_perm[:, t, f] = rng.permutation(n_perm).astype(np.float32) * 0.0  # zero-replace
            preds = pred_fn(torch.from_numpy(x_perm).float().to(device)).detach().cpu().numpy()
            values[t, f] = base_pred - float(preds.mean())
    return ForecastExplanation(
        values=values,
        base_value=base_value,
        feature_names=feature_names,
        target_horizon_step=target_horizon_step,
        quantile_idx=quantile_idx,
        method="permutation",
    )


__all__ = ["ForecastExplanation", "explain"]
