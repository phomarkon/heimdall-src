"""HuggingFace artefact hydrator.

Single Responsibility: download a forecaster's checkpoint directory from
HF on demand and place it at the canonical local path.  Knows nothing
about model architectures or quantile heads — that is the backend
loader's job.

Why a separate module: the FastAPI service must work on a fresh clone
where ``models/forecaster/`` is empty.  Tim's agent-runner must not
have to re-implement HF auth / path conventions.  Centralising them
here keeps the rest of the inference layer pure-Python and torch-free
until a model is actually requested.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_FORECASTER_ROOT = REPO_ROOT / "models" / "forecaster"
HF_REPO = "Phongsakon/heimdall"


def checkpoint_dir(
    name: str,
    seed: int,
    required_files: Iterable[str] | None = None,
) -> Path:
    """Return a local checkpoint directory; pull from HF if required files are missing."""
    target = DEFAULT_FORECASTER_ROOT / name / f"seed-{seed}"
    if _checkpoint_ready(target, required_files):
        return target
    return _hydrate_from_hf(name, seed, target, required_files=required_files)


def _checkpoint_ready(target: Path, required_files: Iterable[str] | None) -> bool:
    if not target.exists():
        return False
    if required_files is None:
        return any(target.iterdir())
    return all((target / filename).is_file() for filename in required_files)


def _missing_files(target: Path, required_files: Iterable[str] | None) -> list[str]:
    if required_files is None:
        return []
    return [filename for filename in required_files if not (target / filename).is_file()]


def _hydrate_from_hf(
    name: str,
    seed: int,
    target: Path,
    required_files: Iterable[str] | None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub not installed; either pip install huggingface_hub "
            "or pre-stage the checkpoint at " + str(target)
        ) from e

    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[
                f"models/forecaster/{name}/seed-{seed}/*",
                f"{name}/seed-{seed}/*",
            ],
            local_dir=REPO_ROOT,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"forecaster {name} seed-{seed} not found locally and HF "
            f"snapshot_download failed: {e!r}"
        ) from e
    flat_source = REPO_ROOT / name / f"seed-{seed}"
    if flat_source.exists() and flat_source != target:
        target.mkdir(parents=True, exist_ok=True)
        for source_file in flat_source.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, target / source_file.name)
    if not target.exists() or not any(target.iterdir()):
        raise FileNotFoundError(
            f"HF download completed but {target} is still empty — checked "
            f"models/forecaster/{name}/seed-{seed}/ and {name}/seed-{seed}/"
        )
    missing = _missing_files(target, required_files)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"HF download completed but {target} is missing required files: {joined}"
        )
    return target


__all__ = ["checkpoint_dir", "HF_REPO", "DEFAULT_FORECASTER_ROOT"]
