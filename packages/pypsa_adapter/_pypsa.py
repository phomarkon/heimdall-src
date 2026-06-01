from __future__ import annotations

import importlib
import os
from types import ModuleType


HEIMDALL_PYPSA_ENV_KEYS = ("PYPSA_EUR_DIR", "PYPSA_SOLVER")


def import_pypsa() -> ModuleType:
    saved = {
        key: os.environ.pop(key)
        for key in HEIMDALL_PYPSA_ENV_KEYS
        if key in os.environ
    }
    try:
        return importlib.import_module("pypsa")
    finally:
        os.environ.update(saved)


pypsa = import_pypsa()
