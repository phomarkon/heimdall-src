"""Backend registry — open/closed extension point.

A new forecaster lands by adding one file under ``backends/`` and one
``@register("name")`` decorator on its loader function.  Existing service
code does not change — Open–Closed in action.

Design choices:
- Registry stores *loader callables*, not forecaster instances, so we
  pay the load cost lazily (Tim's agent-runner can hold a registry
  reference without paying for every checkpoint at start-up).
- Loaders take only ``(seed: int)`` so the registry stays uniform; any
  backend-specific knobs live in env vars or the backend module's own
  config block.
- ``list_registered()`` is a pure read; useful for the ``/healthz``
  route and for runtime discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .protocol import Forecaster


LoaderFn = Callable[[int], Forecaster]


@dataclass(frozen=True)
class _RegistryEntry:
    name: str
    loader: LoaderFn
    description: str


_REGISTRY: dict[str, _RegistryEntry] = {}


def register(name: str, *, description: str = "") -> Callable[[LoaderFn], LoaderFn]:
    """Decorator: bind a loader callable under ``name`` in the registry."""
    def _wrap(fn: LoaderFn) -> LoaderFn:
        if name in _REGISTRY:
            raise ValueError(f"forecaster name {name!r} already registered")
        _REGISTRY[name] = _RegistryEntry(name=name, loader=fn, description=description)
        return fn
    return _wrap


def get_loader(name: str) -> LoaderFn:
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(
            f"forecaster {name!r} is not registered. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name].loader


def list_registered() -> list[str]:
    return sorted(_REGISTRY)


def describe(name: str) -> str:
    return _REGISTRY[name.lower()].description if name.lower() in _REGISTRY else ""


__all__ = ["register", "get_loader", "list_registered", "describe", "LoaderFn"]
