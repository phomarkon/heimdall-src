"""LIME (Ribeiro et al., 2016) attribution for the F7/F8 patch-transformer
forecaster. Sibling of :mod:`heimdall_ml.explain.explain_forecast` (SHAP) with
the same contract; the two methods are reported together in the thesis §3.4 as
a SHAP/LIME agreement cross-check on the model-agnostic side.

LIME's local-linear surrogate gives a per-cell weight over the flattened
(seq_len, n_features) window. Unlike SHAP, weight magnitudes are surrogate
regression coefficients — only sign and rank are interpretable across runs.
We expose the surrogate ``score`` (R²) so callers can flag low-fidelity
explanations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class LimeForecastExplanation:
    """LIME attribution over (input timestep, feature) for one forecast.

    ``values`` shape: ``(seq_len, n_features)``. ``score`` is the R² of the
    sparse local linear surrogate on the perturbation sample; values close to
    1 mean the model is locally linear and the explanation is faithful.
    """

    values: NDArray[np.float64]
    local_pred: float
    score: float
    feature_names: tuple[str, ...]
    target_horizon_step: int
    quantile_idx: int


def explain_lime(
    model: torch.nn.Module,
    x_window: NDArray[np.float32],
    *,
    background: NDArray[np.float32] | None = None,
    feature_names: tuple[str, ...] = ("imbalance_price",),
    target_horizon_step: int = 0,
    quantile_idx: int = 1,
    n_samples: int = 1000,
    seed: int = 13,
) -> LimeForecastExplanation:
    """LIME-tabular explanation flattening the ``(T, F)`` window to one row.

    Parameters
    ----------
    model:
        Trained ``torch.nn.Module`` whose forward returns ``(B, H, Q)``.
    x_window:
        Single window with shape ``(seq_len, n_features)``.
    background:
        Optional ``(N, seq_len, n_features)`` background; defaults to 256
        Gaussian-jittered copies of ``x_window`` (matching SHAP's default).
    feature_names:
        Feature column names; cell names are emitted as ``"<feat>@t-<k>"``.
    target_horizon_step:
        Forecast lead to attribute (0 = next 15 min).
    quantile_idx:
        Quantile head to attribute (0 = q10, 1 = q50, 2 = q90).
    n_samples:
        LIME perturbation sample count.
    seed:
        RNG seed for both the background and LIME's internal sampler.
    """
    from lime.lime_tabular import LimeTabularExplainer

    from heimdall_ml.explain._common import model_device, synthesize_background

    model.eval()
    device = model_device(model)
    seq_len, n_features = x_window.shape
    n_cells = seq_len * n_features

    if background is None:
        bg = synthesize_background(x_window, n_samples=256, seed=seed)
    else:
        bg = background

    bg_flat = bg.reshape(bg.shape[0], n_cells).astype(np.float64)
    x_flat = x_window.reshape(n_cells).astype(np.float64)
    cell_names = [
        f"{feature_names[f] if f < len(feature_names) else f'f{f}'}@t-{seq_len - 1 - t}"
        for t in range(seq_len)
        for f in range(n_features)
    ]

    def predict_fn(X: NDArray[np.float64]) -> NDArray[np.float64]:
        x_t = torch.from_numpy(X.reshape(-1, seq_len, n_features)).float().to(device)
        with torch.no_grad():
            y = model(x_t)[:, target_horizon_step, quantile_idx].cpu().numpy()
        return y.reshape(-1)

    explainer = LimeTabularExplainer(
        training_data=bg_flat,
        feature_names=cell_names,
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    expl = explainer.explain_instance(
        data_row=x_flat,
        predict_fn=predict_fn,
        num_features=n_cells,
        num_samples=n_samples,
    )

    # LIME regression mode keys ``as_map`` by label key (0 or 1 depending on
    # version). Take whichever label is present.
    emap = expl.as_map()
    label_key = next(iter(emap.keys()))
    coefs = np.zeros(n_cells, dtype=np.float64)
    for idx, coef in emap[label_key]:
        coefs[idx] = coef
    values = coefs.reshape(seq_len, n_features)

    local_pred = expl.local_pred
    if hasattr(local_pred, "__len__"):
        local_pred = float(np.asarray(local_pred).reshape(-1)[0])
    else:
        local_pred = float(local_pred)

    return LimeForecastExplanation(
        values=values,
        local_pred=local_pred,
        score=float(expl.score),
        feature_names=feature_names,
        target_horizon_step=target_horizon_step,
        quantile_idx=quantile_idx,
    )


__all__ = ["LimeForecastExplanation", "explain_lime"]
