"""Shared utilities for forecast explanation methods."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def synthesize_background(
    x_window: NDArray[np.float32],
    n_samples: int = 32,
    noise_scale: float = 0.1,
    seed: int = 13,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    seq_len, n_features = x_window.shape
    return (x_window + noise_scale * rng.standard_normal((n_samples, seq_len, n_features))).astype(
        np.float32
    )


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device
