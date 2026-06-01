"""Tests for forecaster SHAP attribution.

We verify that on synthetic data where one input feature is the *only*
informative signal, SHAP attribution concentrates mass on that feature.
"""

from __future__ import annotations

import numpy as np
import torch

from heimdall_ml.explain import explain


class _ToyTransformer(torch.nn.Module):
    """Tiny model: forecast = mean of feature 0 over the window.

    This makes feature 0 ground-truth informative; features 1..F-1 are noise.
    """

    def __init__(self, n_features: int = 3, horizon: int = 4, n_quantiles: int = 3) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.n_features = n_features
        # 3 quantile heads, each sees a per-feature linear weight.
        self.q_w = torch.nn.Parameter(torch.zeros(n_quantiles, n_features))
        with torch.no_grad():
            # Quantile heads place weight only on feature 0.
            self.q_w[:, 0] = torch.tensor([0.9, 1.0, 1.1])
        self.q_b = torch.nn.Parameter(torch.zeros(n_quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Average across T.
        x_mean = x.mean(dim=1)  # (B, F)
        # (B, Q) = x_mean @ q_w.T + q_b
        out_q = x_mean @ self.q_w.t() + self.q_b
        # Repeat across horizon.
        return out_q.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()


def test_explain_concentrates_on_informative_feature() -> None:
    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    seq_len, n_features = 16, 3
    model = _ToyTransformer(n_features=n_features)

    x = rng.standard_normal((seq_len, n_features)).astype(np.float32)
    bg = rng.standard_normal((32, seq_len, n_features)).astype(np.float32)

    expl = explain(
        model=model,
        x_window=x,
        background=bg,
        feature_names=("informative", "noise_a", "noise_b"),
        target_horizon_step=0,
        quantile_idx=1,
    )
    assert expl.values.shape == (seq_len, n_features)
    # Sum of |attribution| per feature
    per_feature = np.abs(expl.values).sum(axis=0)
    # Feature 0 must dominate; the noise features must be ~0.
    assert per_feature[0] > 5 * per_feature[1]
    assert per_feature[0] > 5 * per_feature[2]


