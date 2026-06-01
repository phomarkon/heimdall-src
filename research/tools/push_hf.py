"""Push trained forecaster checkpoints to the canonical HF model repo.

Policy (set 2026-05-16):
  - Target repo: ``Phongsakon/heimdall`` (fixed; not date-stamped).
  - **Newest-only**: every push deletes pre-existing files in the repo before
    upload, so the HF mirror always reflects the current local checkpoint set.
  - **Checkpoints only**: model weights + normalization stats + cards.
    Data, val_preds.npz, metrics.json, configs, mlruns, leaderboards, logs
    live in GitHub — they are NOT mirrored to HF (per user directive
    2026-05-16: "everything to GitHub except model checkpoint to HF").

Reads ``HF_TOKEN`` from the environment; never echoes it. Falls back to a
local tarball under ``models/release/`` if the API rejects the token.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "forecaster"
RELEASE_DIR = REPO_ROOT / "models" / "release"
DEFAULT_REPO_ID = "Phongsakon/heimdall"

# Only these filename patterns get mirrored to HF.
CHECKPOINT_PATTERNS = (
    "*.pt", "*.npz",         # model weights (PyTorch; numpy AR/F0)
    "*.safetensors", "*.bin",  # alt weight formats
    "stats.pkl",              # normalization stats — small, needed at inference
    "MODEL_CARD.md", "README.md", "config.json",  # per-model cards + arch config
)
# `val_preds.npz` is the ONE .npz we explicitly exclude (lives in GitHub).
EXCLUDE_NAMES = ("val_preds.npz",)


def _model_card(repo_id: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""---
license: apache-2.0
language: en
library_name: pytorch
tags:
  - forecasting
  - conformal-prediction
  - electricity-markets
  - mFRR
  - DK1
  - quantile-regression
  - patchTST
  - DeepAR
  - TimesFM
  - chronos-bolt
---

# Heimdall forecasters — DK1 imbalance prices

Pre-release research checkpoints for the Heimdall thesis project (BSc Software
Engineering, SDU Sønderborg; industrial partner Danfoss A/S). These weights
predict 15-minute imbalance prices for the Nordic mFRR EAM in bidding zone DK1
under the post-2025-03-04 single-price regime.

This is a **newest-only mirror** — each push replaces the previous revision.
Latest snapshot: **{today}**.

## What's here
Only model weights + normalization stats + cards. Reproducible val/test
metrics, predictions, leaderboard, training logs, and the panel data all live
in the source-of-truth GitHub repository (`phomarkon/heimdall`).

## Per-model layout
```
<F-id>/seed-<n>/
    model.pt | model.npz       # weights
    stats.pkl                  # train-set normalization (mean/std)
    config.json                # architecture + training config
```

## Forecaster zoo
| ID | Family | Notes |
|----|--------|-------|
| F0 | Seasonal AR(24)            | deterministic baseline |
| F1 | Quantile LightGBM          | gradient-boosted quantile trees |
| F2 | Bayesian Linear Regression | closed-form posterior |
| F3 | Deep Ensemble (5× F7)      | aggregation over F7 seeds (ADR-0006) |
| F3-Lite | LSTM DeepAR           | appendix; original day-3 implementation |
| F4 | MC-Dropout transformer     | K=30 over F7 backbones |
| F7 | patchTST + quantile + split-CP | univariate workhorse |
| F8 | patchTST + online ACI      | regime-shift-aware (default in production) |
| F9 | TimesFM-2.0 zero-shot      | foundation model; deterministic |
| F10 | Chronos-Bolt zero-shot    | foundation model; appendix |
| F11 | PriceFM-shaped surrogate  | 8-layer × 192-d patchTST, no public PriceFM weights |

## License
Apache 2.0.

## Citation
```bibtex
@misc{{heimdall2026,
  title={{Heimdall: A Verifier-Guarded LLM Society for Post-EAM Nordic Balancing Markets}},
  author={{Konrad, Phongsakon and Adam, Tim Lukas}},
  year={{2026}},
  note={{BSc thesis, SDU S{{\\o}}nderborg; industrial partner Danfoss A/S.}}
}}
```
"""


def _collect_checkpoint_paths() -> list[tuple[Path, str]]:
    """List (local_path, repo_relative_path) for files we mirror to HF."""
    out: list[tuple[Path, str]] = []
    for f in MODEL_DIR.rglob("*"):
        if not f.is_file() or f.name in EXCLUDE_NAMES:
            continue
        if any(f.match(p) for p in CHECKPOINT_PATTERNS):
            rel = f.relative_to(MODEL_DIR).as_posix()
            out.append((f, rel))
    return out


def push(repo_id: str = DEFAULT_REPO_ID) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return _local_fallback("HF_TOKEN missing")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return _local_fallback("huggingface_hub not installed")

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001
        return _local_fallback(f"whoami failed: {exc}")
    if not who.get("name"):
        return _local_fallback("HF whoami returned no user")

    try:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return _local_fallback(f"create_repo failed: {exc}")

    # Newest-only: list current repo files and delete those not in the new push set.
    new_files = _collect_checkpoint_paths()
    new_set = {rel for _, rel in new_files}
    new_set.add("README.md")  # we always write a fresh card

    try:
        existing = api.list_repo_files(repo_id, repo_type="model")
    except Exception as exc:  # noqa: BLE001
        existing = []
        print(f"[hf] could not list existing files (continuing): {exc}")

    to_delete = [p for p in existing if p not in new_set and not p.startswith(".gitattributes")]

    # Write fresh model card at repo root.
    card_path = MODEL_DIR / "README.md"
    card_path.write_text(_model_card(repo_id))

    # Batched single-commit: one CommitOperationDelete per stale file, one
    # CommitOperationAdd per new file -> ONE API commit (vs. 50+). Avoids
    # the 128-commits/hour rate limit.
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    ops: list = [CommitOperationDelete(path_in_repo=p) for p in to_delete]
    for local_path, rel in new_files:
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(local_path)))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)))
    try:
        api.create_commit(
            repo_id=repo_id, repo_type="model", operations=ops,
            commit_message=f"newest-only mirror @ {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} "
                           f"({len(new_files)} adds, {len(to_delete)} deletes)",
        )
    except Exception as exc:  # noqa: BLE001
        return _local_fallback(f"create_commit failed: {exc}")

    url = f"https://huggingface.co/{repo_id}"
    print(f"[hf] uploaded -> {url}  ({len(new_files)} files)")
    return {"status": "uploaded", "repo_id": repo_id, "url": url,
            "n_files_pushed": len(new_files), "n_files_deleted": len(to_delete)}


def _local_fallback(reason: str) -> dict:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = RELEASE_DIR / f"heimdall-{today}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname="forecaster", filter=_strip_caches)
    print(f"[hf] FALLBACK ({reason}); wrote {out}")
    return {"status": "fallback", "blocker": reason, "tarball": str(out)}


def _strip_caches(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in info.name or info.name.endswith(".DS_Store"):
        return None
    if info.name.endswith("val_preds.npz"):
        return None
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = p.parse_args(argv)
    out = push(args.repo_id)
    return 0 if out.get("status") in ("uploaded", "fallback") else 1


if __name__ == "__main__":
    raise SystemExit(main())
