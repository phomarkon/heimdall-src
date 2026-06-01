"""Probe TimesFM 2.5 availability and load-ability on this box.

Per ADR-0007. Tries a small grid of plausible architectures (200M variant
ships with sm 2.5 weights at `google/timesfm-2.5-200m-pytorch`); if any
loads cleanly and emits predictions on a 96-step DK1 window, write the
result to `models/forecaster/f9/probe_2_5.json` and update the wrapper to
prefer 2.5. Otherwise fall back to 2.0 — already on disk + functional.

The probe is idempotent and cheap: a single load + one forecast call.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "models/forecaster/f9/probe_2_5.json"

CANDIDATES = [
    # (repo_id, num_layers guess) — 2.5-200m variant; layer count documented at hf
    ("google/timesfm-2.5-200m-pytorch", 50),
    ("google/timesfm-2.5-200m-pytorch", 20),
]


def _try_load(repo_id: str, num_layers: int, history: np.ndarray) -> dict:
    import timesfm

    hp = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=4,
        horizon_len=16,
        context_len=96,
        num_layers=num_layers,
        use_positional_embedding=False,
    )
    ck = timesfm.TimesFmCheckpoint(version="torch", huggingface_repo_id=repo_id)
    model = timesfm.TimesFm(hparams=hp, checkpoint=ck)
    mean, full = model.forecast([history.tolist()], freq=[0])
    return {
        "ok": True,
        "repo_id": repo_id,
        "num_layers": num_layers,
        "mean_first": float(mean[0][0]),
        "mean_shape": list(np.asarray(mean[0]).shape),
        "full_shape": list(np.asarray(full[0]).shape),
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    history = np.linspace(-1.0, 1.0, 96)

    results = []
    success: dict | None = None
    for repo_id, n_layers in CANDIDATES:
        try:
            r = _try_load(repo_id, n_layers, history)
            results.append(r)
            success = r
            break
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "ok": False,
                    "repo_id": repo_id,
                    "num_layers": n_layers,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace_tail": traceback.format_exc().splitlines()[-3:],
                }
            )

    payload = {
        "probe_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "candidates": results,
        "selected": success["repo_id"] if success else None,
        "fallback_to_2_0": success is None,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
