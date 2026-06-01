# Heimdall

Verifier-guarded LLM-agent society for the post-2025 Nordic 15-minute mFRR balancing market.

BSc Software Engineering thesis (SDU Sonderborg) by Phongsakon "Mark" Konrad and Tim Lukas Adam, with Danfoss A/S as industrial partner.

This is the public source release. Data panels are versioned through DVC under
CC-BY-4.0 and reproducible from public sources; forecaster checkpoints are
mirrored on HuggingFace; the thesis manuscript lives in its own repository.

- Model card and forecaster checkpoints: https://huggingface.co/Phongsakon/heimdall

## What this is

A society of heterogeneous LLM-persona agents simulating the Nordic mFRR balancing market,
built around a focal market-making agent whose every bid passes a two-stage verifier
(physical feasibility + conformal worst-case-profit). Theorem 1a (split-CP) and Theorem 1b
(online ACI) guarantee that accepted bids inherit conformal coverage on realised profit
regardless of LLM hallucinations.

## Repository structure

```
heimdall-src/
  app/                  Production services
    forecaster          FastAPI inference, 20+ backends (F0-F13)
    verifier            Two-stage physical + conformal bid gate
    conformal-calibrator Split-CP / ACI / BOCPD-ACI intervals
    run-view            Replay API (hybrid Postgres + disk)
    market-simulator    mFRR clearing replay engine
    pypsa-scenario      PyPSA-Eur-Sec P2H asset specs
    agent-runner        vLLM/Qwen3-32B tool-use wrapper
    focal-orchestrator  Multi-horizon focal coordinator
    frontend            Next.js 15 + React 19 + Tailwind v4

  packages/             Shared Python libraries
    contracts           Pydantic v2 schemas (zero internal deps)
    ml                  Conformal prediction, XAI, MLflow tracking
    data                ENTSO-E, Energinet, EDS, OpenMeteo loaders
    markets             Shared mFRR profit math
    personas            LLM agent archetypes
    simulator           Replay simulator, agent tool, mFRR engine
    pypsa_adapter       PyPSA-Eur-Sec network wrapper

  research/             Experiment code and the full run-config ledger
    ai-society          Society runners, UCloud configs, ablation matrices
    experiments         Ablations, baselines, test-set evaluation
    tools               DVC pipeline, feature engineering, evaluation scripts

  deploy/               Helm, ArgoCD, Docker, observability
  docs/                 Reproducibility runbook
  tests/                Cross-package integration tests
```

Run outputs (`mlruns/`, society run logs), data panels (`data/`), and model
checkpoints are not committed here: data is pulled with DVC (`uv run dvc pull`)
and checkpoints hydrate from HuggingFace on first use.

## Quick start

Requires Python 3.12. On a fresh Debian/Ubuntu machine `setup.sh` installs `uv`
and `bun`, syncs the environment, pulls data (best-effort), and provisions a
local PostgreSQL (needs `sudo`). Then `dev-stack.sh` brings up the dashboard.

```bash
bash setup.sh                                  # uv env + bun + data + local Postgres
uv run pytest -q -m "not gpu and not slow"     # CPU test suite (what CI runs)
bash dev-stack.sh                              # Postgres :5432 + run-view :8091 + frontend :3000
# open http://localhost:3000
```

Frontend-only / no-Postgres viewing (run-view serves runs from disk; the Config
page's spec/template persistence is unavailable):

```bash
bash setup.sh --no-db
uv run uvicorn heimdall_run_view.service:app --host 127.0.0.1 --port 8091 &
(cd app/frontend && bun run dev)              # http://localhost:3000
```

See `docs/REPRODUCE.md` for the full runbook.

## Services

Each service is independently testable and deployable. All communicate through
the shared `contracts` package (Pydantic v2 schemas).

| Service | Port | Endpoints |
|---|---|---|
| forecaster | 8001 | `POST /forecast` |
| conformal-calibrator | 8002 | `PUT /series/{id}`, `POST /series/{id}/observation`, `GET /series/{id}/interval` |
| verifier | 8003 | `POST /verify` |
| pypsa-scenario | 8005 | `GET /assetspec`, `GET /scenario` |
| run-view | 8091 | `GET /v1/runs`, `GET /v1/runs/{id}/*`, `POST /v1/agent-templates` |
| frontend | 3000 | Next.js dashboard (Live / Runs / Config / Results / Help) |

## ML inference

No checkpoint needed:
- `ar1` -- Gaussian AR(1) fallback
- `f0` -- Seasonal AR(24)

Hydrated from HuggingFace on first use:
- `f7` -- PatchTST quantile + split-CP
- `f8b` / `f8c` -- Rich-feature patch-TST (focal-society default)
- `f9` -- TimesFM-2.0 zero-shot
- `f10` -- Chronos-Bolt zero-shot

Backend registry: `app/forecaster/src/heimdall_forecaster/inference/backends/`

## Frontend

```bash
cd app/frontend
bun install
bun run dev             # :3000
bun run test            # vitest
```

One-command full stack:
```bash
bash dev-stack.sh       # Postgres :5432 + run-view :8091 + Next.js :3000
bash dev-stack.sh --stop
```

## AI society

```bash
# Local dry-run (no GPU)
PYTHONPATH=.:research python -m heimdall_ai_society run \
  --config research/llm/ai-society/configs/local-dryrun.yaml

# UCloud with vLLM (3x B200)
PYTHONPATH=.:research python -m heimdall_ai_society run \
  --config research/llm/ai-society/configs/smoke-5.yaml
```

Outputs land in `research/llm/ai-society/runs/<run_id>/`.

## Testing

```bash
pytest -x                                 # full suite
pytest app/run-view/tests/ -x             # single service
pytest -k test_name                       # single test
ruff check .                              # lint
```

## Hard constraints

- Hardware: 2-3x NVIDIA B200 (384-576 GB VRAM). vLLM 0.21.0 + Qwen3-32B.
- Frozen seeds: [13, 42, 137, 1729, 31415]. Every result averages over five.
- Pre/post split: 2025-03-04 00:00 UTC (mFRR EAM go-live). See ADR-0002.
- Train/val/test: train <= 2025-02-28; val 2025-03-04 -- 2025-04-30; test 2025-05-01 -- 2026-04-30.

## Documentation

- `docs/REPRODUCE.md` -- reproducibility runbook

## License

- Code: Apache 2.0 (`LICENSE`)
- Data bundle: CC-BY-4.0 (DVC, reproducible from public sources)
- Models: Apache 2.0 (HuggingFace)
- Frontend: MIT
