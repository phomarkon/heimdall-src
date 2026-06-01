#!/usr/bin/env bash
# Heimdall — one-shot cloud-GPU setup script.
#
# Designed for a fresh uCloud B200 instance (Linux, CUDA ≥ 12.4).  Idempotent:
# safe to re-run; skips work that's already done.  After this script finishes
# you can `source .venv/bin/activate && pytest -q` and everything works.
#
# Usage:
#   bash setup.sh                # default: install everything
#   bash setup.sh --no-data      # skip data fetch
#   bash setup.sh --no-hf        # skip HF artefact download
#   bash setup.sh --no-thesis    # skip texlive (saves ~600 MB)
#   bash setup.sh --no-frontend  # skip bun + frontend deps
#   bash setup.sh --no-db        # skip PostgreSQL provisioning (disk-only run-view)
#   bash setup.sh --cloud-ai-society
#                                # additionally validate UCloud GPU context
#                                # for ai-society native vLLM runtime
#
# After setup, bring up the dashboard with:  bash dev-stack.sh
#
# Environment variables consumed:
#   HF_TOKEN          (optional) — HuggingFace token for private repo pulls
#   ENTSOE_API_TOKEN  (optional) — ENTSO-E free-tier token for the
#                                  ingest pipeline; not required for
#                                  Energinet (no-auth) ingest

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- flag parsing ----------------------------------------------------------
DO_DATA=1
DO_HF=1
DO_THESIS=1
DO_AI_SOCIETY=0
DO_FRONTEND=1
DO_DB=1
for arg in "$@"; do
  case "$arg" in
    --no-data)   DO_DATA=0 ;;
    --no-hf)     DO_HF=0 ;;
    --no-thesis) DO_THESIS=0 ;;
    --no-frontend) DO_FRONTEND=0 ;;
    --no-db)     DO_DB=0 ;;
    --cloud-ai-society) DO_AI_SOCIETY=1 ;;
    --no-ai-society) DO_AI_SOCIETY=0 ;;
    -h|--help)
      sed -n '2,24p' "$0"; exit 0 ;;
    *)
      echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[36m[setup]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. system packages ----------------------------------------------------
log "1/8 system packages"
if have apt-get; then
  sudo apt-get update -y >/dev/null
  sudo apt-get install -y --no-install-recommends \
    build-essential curl git jq make \
    python3-dev python3-venv \
    libgl1 libpq-dev pkg-config \
    >/dev/null
  if [[ $DO_THESIS == 1 ]]; then
    sudo apt-get install -y --no-install-recommends \
      texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra \
      >/dev/null
  fi
fi

# --- 2. uv ------------------------------------------------------------------
log "2/8 uv (Python package manager)"
if ! have uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- 3. Python venv + workspace deps ---------------------------------------
log "3/8 Python venv + workspace deps (uv sync)"
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --frozen
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 4. PyTorch CUDA wheel verification ------------------------------------
log "4/8 PyTorch CUDA wheel verification"
python - <<'PY'
import torch
print(f"  torch  {torch.__version__}")
print(f"  cuda   available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
PY

# --- 5. Extra ML deps (cvxpy, sb3, gymnasium, dvc) -------------------------
log "5/8 Extra deps (cvxpy, stable-baselines3, gymnasium, dvc)"
uv pip install -q \
  cvxpy gymnasium stable-baselines3 \
  prometheus-client \
  dvc dvc-s3 \
  || true

# --- 6. HF artefacts -------------------------------------------------------
if [[ $DO_HF == 1 ]]; then
  log "6/8 HuggingFace artefacts (forecaster zoo + mlruns + B4 PPO)"
  if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
    log "  HF_TOKEN unset and no cached token; skipping HF pull. Run 'hf auth login' to enable."
  else
    if [[ -n "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
      mkdir -p "$HOME/.cache/huggingface"
      printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
      chmod 600 "$HOME/.cache/huggingface/token"
    fi
    REPO=Phongsakon/heimdall-forecasters-2026-05-09
    log "  pulling $REPO"
    hf download "$REPO" --repo-type=model --local-dir "$REPO_ROOT/_hf_snapshot" >/dev/null
    # rsync into the right paths
    [[ -d _hf_snapshot/mlruns ]]               && rsync -a _hf_snapshot/mlruns/ research/ml/mlruns/ || true
    [[ -d _hf_snapshot/models/forecaster ]]    && rsync -a _hf_snapshot/models/forecaster/ research/ml/models/forecaster/ || true
    [[ -d _hf_snapshot/baselines/b4_ppo ]]     && rsync -a _hf_snapshot/baselines/b4_ppo/ research/ml/models/baselines/b4_ppo/ || true
    [[ -d _hf_snapshot/references_papers ]]    && rsync -a _hf_snapshot/references_papers/ references/papers/ || true
    [[ -d _hf_snapshot/experiments_outputs ]]  && rsync -a _hf_snapshot/experiments_outputs/ research/ml/experiments/outputs/ || true
    rm -rf _hf_snapshot
    log "  HF artefacts hydrated"
  fi
else
  log "6/8 HF artefacts: skipped (--no-hf)"
fi

# --- 7a. PyPSA-Eur-Sec cost CSV (no auth) ---------------------------------
log "7a PyPSA-Eur-Sec cost CSV"
mkdir -p data/raw/pypsa_eursec
if [[ ! -f data/raw/pypsa_eursec/costs_2030.csv ]]; then
  curl -L -o data/raw/pypsa_eursec/costs_2030.csv \
    https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2030.csv \
    || log "  WARNING: cost CSV pull failed; A9 + verifier asset spec will be unavailable"
fi
sha256sum data/raw/pypsa_eursec/costs_2030.csv | tee data/raw/pypsa_eursec/costs_2030.csv.sha256

# --- 7. Public-data ingest (free tier; no auth needed for Energinet) ------
if [[ $DO_DATA == 1 ]]; then
  log "7/8 public-data ingest (Energinet EDS, no auth)"
  if [[ -f data/processed/dk1_panel_features_v2.parquet ]]; then
    log "  feature panel already exists; skipping"
  else
    PYTHONPATH=. python research/tools/ingest_public_features.py \
      --start 2024-01-01T00:00 --end 2026-04-29T23:45 --zone DK1 || \
      log "  WARNING: ingest failed (likely network); rerun once online"
  fi
else
  log "7/8 public-data ingest: skipped (--no-data)"
fi

# --- 7b. frontend toolchain (bun) ------------------------------------------
if [[ $DO_FRONTEND == 1 ]]; then
  log "7b frontend toolchain (bun + deps)"
  if ! have bun; then
    if [[ -x "$HOME/.bun/bin/bun" ]]; then
      export PATH="$HOME/.bun/bin:$PATH"
    else
      curl -fsSL https://bun.sh/install | bash || log "  WARNING: bun install failed"
      export PATH="$HOME/.bun/bin:$PATH"
    fi
  fi
  if have bun; then
    log "  bun $(bun --version)"
    ( cd app/frontend && bun install ) || log "  WARNING: bun install (deps) failed"
  else
    log "  WARNING: bun not on PATH; install from https://bun.sh then 'cd app/frontend && bun install'"
  fi
else
  log "7b frontend toolchain: skipped (--no-frontend)"
fi

# --- 7c. PostgreSQL (run-view backend) -------------------------------------
# Provisions the local cluster + heimdall role/database matching the run-view
# DSN. run-view serves runs from disk without this; Postgres adds society-spec
# and agent-template persistence (the dashboard Config page).
PG_DSN="${HEIMDALL_RUN_VIEW_DATABASE_URL:-postgresql://heimdall:heimdall@127.0.0.1:5432/heimdall}"
export HEIMDALL_RUN_VIEW_DATABASE_URL="$PG_DSN"
if [[ $DO_DB == 1 ]]; then
  log "7c PostgreSQL (run-view backend)"
  if have apt-get; then
    if ! have pg_ctlcluster; then
      sudo apt-get install -y --no-install-recommends postgresql postgresql-client >/dev/null \
        || log "  WARNING: postgresql install failed; run-view will use disk-only mode"
    fi
    if have pg_ctlcluster; then
      sudo pg_ctlcluster 16 main start 2>/dev/null || true
      for _ in $(seq 1 20); do pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1 && break; sleep 0.5; done
      sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='heimdall'" 2>/dev/null | grep -q 1 \
        || sudo -u postgres psql -c "CREATE ROLE heimdall LOGIN PASSWORD 'heimdall';" >/dev/null 2>&1 \
        || log "  WARNING: could not create role heimdall"
      sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='heimdall'" 2>/dev/null | grep -q 1 \
        || sudo -u postgres createdb -O heimdall heimdall >/dev/null 2>&1 \
        || log "  WARNING: could not create database heimdall"
      heimdall-run-view init-db >/dev/null 2>&1 \
        && log "  run-view schema ready ($PG_DSN)" \
        || log "  WARNING: run-view init-db failed; check Postgres auth for $PG_DSN"
    else
      log "  WARNING: pg_ctlcluster unavailable; run-view will use disk-only mode"
    fi
  else
    log "  non-Debian host: install PostgreSQL 16 manually (see docs/REPRODUCE.md) or use --no-db"
  fi
else
  log "7c PostgreSQL: skipped (--no-db); run-view will serve runs from disk only"
fi

# --- 8. Smoke tests --------------------------------------------------------
log "8/8 smoke tests"
PYTHONPATH=. pytest -q -x \
  tests/test_data_config.py \
  tests/test_physical_verifier.py \
  tests/test_simulator_replay.py \
  packages/ml/tests/test_aci.py \
  packages/ml/tests/test_split_cp.py \
  || log "  WARNING: some smoke tests failed; check logs above"

# --- thesis build (optional sanity) ---------------------------------------
if [[ $DO_THESIS == 1 ]]; then
  log "thesis: pdflatex sanity build"
  ( cd thesis && make >/dev/null 2>&1 ) || log "  thesis build failed (non-fatal)"
fi

# --- AI society cloud setup (explicit opt-in; server-only) -----------------
if [[ $DO_AI_SOCIETY == 1 ]]; then
  log "ai-society: validating UCloud GPU/native vLLM context"
  bash research/llm/ai-society/setup-cloud.sh
else
  log "ai-society: skipped (use --cloud-ai-society on the UCloud GPU server)"
fi

cat <<'EOF'

=== setup complete ===

Next steps:
  source .venv/bin/activate
  bash dev-stack.sh   # Postgres :5432 + run-view :8091 + frontend :3000 -> http://localhost:3000
  PYTHONPATH=.:research python experiments/baselines_focal_agent.py
  PYTHONPATH=.:research python experiments/ablations/a8b_bocpd_aci.py
  PYTHONPATH=.:research python experiments/ablations/a9_pypsa_vs_custom.py

Useful entry points:
  - Reproducibility runbook       : docs/REPRODUCE.md
  - Reproducibility bundle prep   : research/tools/data_bundle.py
  - AI society UCloud runbook     : research/llm/ai-society/SOCIETY-PLAN.md

Native UCloud vLLM next steps:
  bash research/llm/ai-society/ucloud-vllm/scripts/install_to_work.sh
  cd /work/heimdall-vllm && bash scripts/launch_tmux.sh
  source /work/heimdall-vllm/scripts/export_env.sh

If you hit a CUDA / driver mismatch, see scaling-pearl-ladder/CLOUD.md
in the parent workspace for B200 wheel selection.
EOF
