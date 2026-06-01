# Reproducing Heimdall results

This document gives **exact commands** to reproduce every table, figure, and
numerical claim in `manuscript.tex` and the BSc thesis `thesis.tex`.

For standing up the **dashboard stack** (Postgres + run-view + frontend), see
[Spinning up the stack](#spinning-up-the-stack) below. For the **results**
pipeline, continue to [Prerequisites](#prerequisites).

## Spinning up the stack

Verified on Ubuntu 24.04, Python 3.12. Produces the running dashboard with the
society runs loaded.

```bash
bash setup.sh                                  # uv env + bun + data + local Postgres
uv run pytest -q -m "not gpu and not slow"     # CPU tests (296 pass)
bash dev-stack.sh                              # Postgres + run-view + frontend
# open http://localhost:3000
```

`setup.sh` is idempotent and (relevant flags): `--no-db` skips PostgreSQL,
`--no-frontend` skips bun/frontend deps, `--no-data`/`--no-hf` skip the heavy
data and checkpoint pulls. Steps 7b/7c need network and `sudo` (apt). On
non-Debian hosts, install PostgreSQL 16 and `bun` manually.

### PostgreSQL (run-view backend)

`setup.sh` provisions this. By hand (Debian/Ubuntu):

```bash
sudo apt-get install -y postgresql postgresql-client
sudo pg_ctlcluster 16 main start
sudo -u postgres psql -c "CREATE ROLE heimdall LOGIN PASSWORD 'heimdall';"
sudo -u postgres createdb -O heimdall heimdall
uv run heimdall-run-view init-db        # create the run-view schema
```

DSN defaults to `postgresql://heimdall:heimdall@127.0.0.1:5432/heimdall`
(override via `HEIMDALL_RUN_VIEW_DATABASE_URL`). Run artifacts stay on disk;
Postgres only holds society specs and agent templates. run-view degrades
gracefully when Postgres is absent: the run catalogue still serves from disk,
only spec/template persistence (the Config page) returns 503.

### The dashboard

```bash
bash dev-stack.sh              # foreground; Ctrl-C stops run-view + frontend (Postgres stays up)
bash dev-stack.sh --detach    # all three detached, returns the shell
bash dev-stack.sh --status    # report what is up
bash dev-stack.sh --stop      # stop run-view + frontend
```

- run-view: `http://127.0.0.1:8091` (`/v1/runs`, `/v1/runs/{id}/*`).
- frontend: `http://127.0.0.1:3000` (single-route SPA; nav tabs are client-side).

run-view auto-discovers the catalogue at `research/llm/ai-society/runs` (1347
runs; override with `HEIMDALL_RUNS_DIR`), so no extra config is needed to see
them. The frontend reads `NEXT_PUBLIC_HEIMDALL_API_URL` (default
`http://127.0.0.1:8091`), so no `.env.local` is required for local use.

Disk-only (no Postgres):

```bash
bash setup.sh --no-db
uv run uvicorn heimdall_run_view.service:app --host 127.0.0.1 --port 8091 &
(cd app/frontend && bun run dev)
```

### Society dry-run (CPU, no GPU)

```bash
PYTHONPATH=.:research:research/llm/ai-society/src \
  uv run python -m heimdall_ai_society run \
  --config research/llm/ai-society/configs/local-dryrun.yaml
```

Outputs land in `research/llm/ai-society/runs/<run_id>/` and run-view picks them
up automatically.

---

## Prerequisites

## Prerequisites

```bash
git clone https://github.com/phomarkon/heimdall.git
cd heimdall
git checkout dev   # or the tagged manuscript submission commit
uv sync            # installs the workspace + dev deps
export HF_TOKEN=...  # not required for code; required if you want HF mirror
```

Hardware:
- Forecaster training: NVIDIA B200 (1× sufficient; 183 GB VRAM means all
  forecaster training fits in a single GPU)
- Conformal calibration, leaderboard, classical baselines: CPU
- Total wall time for a from-scratch reproduction on B200: ~6-8 hours

Python: **3.12**. Strict, enforced via `pyproject.toml`.

## Step 1. Data

```bash
# Re-fetch monthly DK1 panels from Energinet + ENTSO-E. Requires
# ENTSOE_API_TOKEN. Idempotent: skips months already on disk.
bash tools/get_data.sh

# Build the canonical train/val/test splits (frozen at 2025-03-04 EAM break)
PYTHONPATH=packages/data/src uv run python tools/build_dk1_panels.py
```

If you do not have `ENTSOE_API_TOKEN`, the processed splits are committed
to git at `data/processed/dk1_panel_{train,val,test}.parquet` and you can
skip this step.

## Step 2. Forecaster zoo (5 seeds each)

```bash
# F0 / F7 / F8 (and F3-Lite via --models f3)
uv run python experiments/seed_sweep.py --models f0 f7 f8

# F3-Lite (LSTM DeepAR; appendix-only forecaster)
uv run python experiments/seed_sweep.py --models f3

# F1 (Quantile LightGBM), CPU-bound, ~12 min/seed × 5
uv run python -m heimdall_forecaster.train.f1_lgbm

# F2 (Bayesian Linear Regression), closed-form, ~30s/seed × 5
uv run python -m heimdall_forecaster.train.f2_blr

# F3-ensemble (aggregation of F7 seeds; pure CPU)
uv run python -m heimdall_forecaster.train.f3_ensemble

# F4 MC-Dropout K=30 over F7 backbones
uv run python -m heimdall_forecaster.train.f4_mc_dropout

# F8b / F8c / F8d / F8e (multivariate variants on rich panel, 20 epochs)
bash tools/run_f8_multivariate_20ep.sh

# F9 (TimesFM-2.0 zero-shot, full val window)
PYTHONPATH=. uv run python experiments/eval_f9_timesfm_zoo.py --backend gpu

# F11 (PriceFM-shaped surrogate, 5 seeds)
bash tools/run_f11_seeds.sh

# F12-EBM (Karras-EDM, 30 epochs × 5 seeds)
uv run python -m heimdall_forecaster.train.f12_ebm
```

After this step you should have `models/forecaster/<name>/seed-<n>/{model.pt,
stats.pkl, val_preds.npz, metrics.json, aci_state.json}` for every forecaster
in the zoo.

## Step 3. Finalize metrics + rebuild leaderboard

```bash
uv run python tools/finalize_metrics.py
PYTHONPATH=. uv run python experiments/build_leaderboard.py
```

Output: `notes/forecaster_leaderboard.md` (Table 1 in the manuscript, §8.1).

## Step 4. Ablations and Feynman-check experiments

```bash
# A8-PyPSA physics-grounded regime shift (§8.3.5 / Table 3 of manuscript)
PYTHONPATH=. uv run python experiments/ablations/a8_pypsa_shift.py

# Verifier-discriminates-by-safety (§8.5b / Table 4)
PYTHONPATH=. uv run python experiments/feynman_verifier_discriminates.py

# F12-EBM heavy-tail ablation (§8.5c / Table 5)
PYTHONPATH=. uv run python experiments/feynman_f12_heavy_tail.py

# Classical baselines (val only)
uv run python experiments/baselines_classical.py --baselines b5 b6 b8 b9
```

## Step 5. Single-shot test-set evaluation (LEDGER-GATED)

```bash
# This is a one-shot evaluation per (model, seed, config_hash). Re-runs
# require --allow-rerun --reason and a logged provenance entry.
PYTHONPATH=. uv run python experiments/test_set_evaluation.py \
    --models f0 f1_lgbm f2_blr f3_ensemble f3_lite f4_mc_dropout \
             f7 f8 f8b f8c f8d f8e f9 f11 f12_ebm \
    --commit
```

Output: `experiments/outputs/test_set_results.json` (Table 2, §8.2).
Ledger: `experiments/outputs/test_set_ledger.json`.

## Step 6. Build the PDFs

```bash
# BSc thesis (book class, ~70 pages)
pdflatex thesis.tex
pdflatex thesis.tex  # second pass for references

# Q1 manuscript (article class, ~13-25 pages)
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Both PDFs use `bibliography/heimdall.bib`.

## Mapping every table/figure to its source

| Artifact | Section | Source |
|---|---|---|
| Table 1 (val leaderboard) | §8.1 | `notes/forecaster_leaderboard.md` ← `models/forecaster/*/seed-*/metrics.json` |
| Table 2 (test leaderboard) | §8.2 | `experiments/outputs/test_set_results.json` |
| Table 3 (A8-PyPSA) | §8.3.5 | `experiments/outputs/a8_pypsa_shift.json` |
| Table 4 (verifier discriminates) | §8.5b | `experiments/outputs/feynman_verifier_discriminates.json` |
| Table 5 (heavy-tail) | §8.5c | `experiments/outputs/feynman_f12_heavy_tail.json` |
| Theorem 1a/1b/1c/1d | §5 | `chapters/05_theorems.tex` |
| ACI panel claim | §8.1 + §8.5c | aggregate of `metrics.json` `aci_empirical_coverage` |

## Hardware notes

- **GPU utilisation** is bottlenecked on data-loading and Python overhead,
  not on B200 compute. For the published configuration, expect <10\% GPU
  util on a single B200. Multi-GPU or larger batch sizes would speed up
  training but the numbers in the paper come from `batch_size = 64`.
- **CPU usage** is significant for F1 (LightGBM single-thread per booster)
  and the classical baselines (SARIMAX in particular is slow).

## Known-broken paths (out of scope for this manuscript)

- F3-Lite and F4 do not have registered HF-Hydrator inference backends;
  their test-path predictions therefore appear as zero in
  `test_set_results.json`. Val numbers via `val_preds.npz` are valid.
- F10 (Chronos-Bolt) has a `huggingface-hub` / `transformers` version
  conflict with our repo pin (`huggingface-hub>=1.14.0`). F10 is
  appendix-only; see `models/forecaster/f10/MODEL_CARD.md`.

## License + citation

- Code: Apache-2.0 (`LICENSE`).
- Data: Energinet Open Data + ENTSO-E Transparency (both free-tier, free
  to redistribute).
- Models: Apache-2.0, mirrored at `Phongsakon/heimdall` on
  HuggingFace.

If you use Heimdall in your work, please cite the BSc thesis and the
forthcoming preprint (DOI to be assigned at submission).
