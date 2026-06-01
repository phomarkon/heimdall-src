"""Shared training utilities — device resolution and scoring functions."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pinball_loss(y: NDArray, q: NDArray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))
