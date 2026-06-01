"""Bulk-upload Mark-track forecaster artefacts to Hugging Face.

Uploads one folder per top-level model directory (one commit each), staying
under HF's 128-commits/hour rate cap. Idempotent — re-uploads replace.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login

REPO = Path(__file__).resolve().parents[2]
HF_REPO = "Phongsakon/heimdall-forecasters"

ALLOW_PATTERNS = [
    "model.pt", "boosters.pkl", "booster.pkl", "stats.pkl",
    "regressors.pkl", "stage_a.pkl", "stage_b.pkl",
    "metrics.json", "config.json", "aci_state.json",
    "val_preds.npz", "MODEL_CARD.md",
    "*/model.pt", "*/boosters.pkl", "*/booster.pkl", "*/stats.pkl",
    "*/regressors.pkl", "*/stage_a.pkl", "*/stage_b.pkl",
    "*/metrics.json", "*/config.json", "*/aci_state.json",
    "*/val_preds.npz",
]


def _resolve_token() -> str | None:
    token = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_API_KEY=") and "=" in line[len("HF_API_KEY="):] is False:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("HF_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v and not v.startswith("HF_API_KEY"):
                    return v
    return None


def main() -> int:
    token = _resolve_token()
    if not token:
        print("ERROR: no HF_API_KEY in env or .env", flush=True)
        return 1
    login(token=token, add_to_git_credential=False)
    api = HfApi()

    model_root = REPO / "models/forecaster"
    dirs = sorted([d for d in model_root.iterdir() if d.is_dir()])
    print(f"[hf] uploading {len(dirs)} model dirs to {HF_REPO}", flush=True)

    ok = 0; fail = 0
    for d in dirs:
        rel = d.relative_to(REPO)
        try:
            api.upload_folder(
                folder_path=str(d),
                path_in_repo=str(rel),
                repo_id=HF_REPO,
                repo_type="model",
                allow_patterns=ALLOW_PATTERNS,
                commit_message=f"Sync {rel}",
            )
            ok += 1
            print(f"[hf] {ok}/{len(dirs)}  ok  {rel}", flush=True)
        except Exception as ex:
            fail += 1
            print(f"[hf] FAIL {rel}: {type(ex).__name__} {ex}", flush=True)

    # Plus the EAM data cache as a dataset-shaped file.
    eam = REPO / "data/processed/mfrr_eam_dk1.parquet"
    if eam.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(eam),
                path_in_repo="data/processed/mfrr_eam_dk1.parquet",
                repo_id=HF_REPO, repo_type="model",
                commit_message="Sync mfrr_eam_dk1.parquet",
            )
            print("[hf] ok  data/processed/mfrr_eam_dk1.parquet", flush=True)
        except Exception as ex:
            fail += 1
            print(f"[hf] FAIL eam: {type(ex).__name__} {ex}", flush=True)

    print(f"[hf] done: {ok} ok / {fail} fail", flush=True)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
