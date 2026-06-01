"""Process-local LRU cache for loaded forecaster instances.

Single Responsibility: avoid re-loading the same checkpoint twice
within a process while keeping memory bounded.

Concurrency: ``functools.lru_cache`` is thread-safe for hashable args.
We key on ``(name, seed)``; the loader callable is fetched from the
registry inside the cache miss path.
"""

from __future__ import annotations

from functools import lru_cache

from .protocol import Forecaster
from .registry import get_loader


@lru_cache(maxsize=32)
def get_forecaster(name: str, seed: int = 42) -> Forecaster:
    loader = get_loader(name)
    return loader(seed)


def clear_cache() -> None:
    get_forecaster.cache_clear()


__all__ = ["get_forecaster", "clear_cache"]
