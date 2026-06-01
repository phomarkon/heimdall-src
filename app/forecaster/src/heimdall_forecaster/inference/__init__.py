"""Heimdall forecaster inference layer.

SOLID-respecting public surface:
- `Forecaster` (protocol) — single inference capability,
- `register` decorator + `get_loader/list_registered/describe` — open/closed
  registry of zoo backends,
- `get_forecaster` — process-local LRU cache over the registry,
- `checkpoint_dir` — single-responsibility HF hydrator,
- side-effect import of `backends/` registers every shipped zoo member.

External callers (the FastAPI service, the focal-orchestrator, the
agent-runner, ablation cells) consume *only this module's surface*; they
never reach into `backends/` or train/* directly.
"""

from .protocol import Forecaster
from .registry import describe, get_loader, list_registered, register
from .cache import clear_cache, get_forecaster
from .hf_hydrator import checkpoint_dir, HF_REPO

# Importing the package side-effect-registers the shipped backends.
from . import backends  # noqa: F401

__all__ = [
    "Forecaster",
    "HF_REPO",
    "checkpoint_dir",
    "clear_cache",
    "describe",
    "get_forecaster",
    "get_loader",
    "list_registered",
    "register",
]
