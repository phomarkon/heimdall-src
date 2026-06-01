"""Frozen seed registry for the project.

Every published result is averaged over these five seeds, paired across
methods. Do not append to this list without a corresponding methodology update.
"""

from __future__ import annotations

import os
import random

import numpy as np

# Frozen for the entire research artefact. See docs/RESEARCH-PROPOSAL.md §5.3.1.
FROZEN_SEEDS: tuple[int, ...] = (13, 42, 137, 1729, 31415)


def seed_everything(seed: int) -> None:
    """Seed Python `random`, NumPy, and (if available) PyTorch.

    Torch is imported lazily so that the ml package does not force a torch
    install when only conformal calibration is used.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
