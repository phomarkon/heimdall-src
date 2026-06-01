"""Tests for forecaster LIME attribution.

On a toy transformer whose forecast depends only on feature 0, LIME
attribution must concentrate on feature-0 cells and treat the other features
as noise. Counterpart to ``test_explain_forecast.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from heimdall_ml.explain import LimeForecastExplanation, explain_lime


class _ToyTransformer(torch.nn.Module):
    """Tiny model: forecast = weighted mean of feature 0 over the window."""

    def __init__(self, n_features: int = 3, horizon: int = 4, n_quantiles: int = 3) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.n_features = n_features
        self.q_w = torch.nn.Parameter(torch.zeros(n_quantiles, n_features))
        with torch.no_grad():
            self.q_w[:, 0] = torch.tensor([0.9, 1.0, 1.1])
        self.q_b = torch.nn.Parameter(torch.zeros(n_quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1)
        out_q = x_mean @ self.q_w.t() + self.q_b
        return out_q.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()


def test_lime_concentrates_on_informative_feature() -> None:
    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    seq_len, n_features = 16, 3
    model = _ToyTransformer(n_features=n_features)

    x = rng.standard_normal((seq_len, n_features)).astype(np.float32)
    bg = rng.standard_normal((128, seq_len, n_features)).astype(np.float32)

    expl = explain_lime(
        model=model,
        x_window=x,
        background=bg,
        feature_names=("informative", "noise_a", "noise_b"),
        target_horizon_step=0,
        quantile_idx=1,
        n_samples=800,
    )
    assert isinstance(expl, LimeForecastExplanation)
    assert expl.values.shape == (seq_len, n_features)

    per_feature = np.abs(expl.values).sum(axis=0)
    assert per_feature[0] > 3 * per_feature[1]
    assert per_feature[0] > 3 * per_feature[2]


