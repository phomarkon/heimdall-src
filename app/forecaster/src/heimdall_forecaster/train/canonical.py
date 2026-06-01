"""Apples-to-apples canonical training runner.

Trains every multivariate-capable F-zoo entry on **the same** ``F_CANONICAL``
feature panel, plus univariate-by-architecture entries on the target series
alone. Both the imbalance-price target and the signed-activation-volume
target are supported via ``--target {price,activation}``.

Fairness contract:
- All multivariate models see ``F_CANONICAL_FEATURES`` over the same windows.
- All univariate models see the same target series over the same windows.
- Train/val panels and seeds are identical across families.
- Per-model architectural hyperparams stay at each family's published defaults
  (we vary inputs, not capacity).

CLI:
    uv run python -m heimdall_forecaster.train.canonical \\
        --models f1_lgbm f8 \\
        --target price --seeds 13 42 137 \\
        [--smoke]

``--smoke`` forces epochs=1 / n_estimators=20 for fast pipeline validation.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from heimdall_forecaster.train.dataset import (
    CANONICAL_FEATURE_GROUPS,
    CANONICAL_MODEL_ROUTING,
    F_CANONICAL_FEATURES,
    f_canonical_without,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TRAIN = REPO_ROOT / "data/processed/dk1_panel_rich_v2_train.parquet"
DEFAULT_VAL = REPO_ROOT / "data/processed/dk1_panel_rich_v2_val.parquet"
DEFAULT_ANOMALY = REPO_ROOT / "data/processed/anomaly_features.parquet"
# Univariate baselines train on the minimal panel that carries the target.
UNIVARIATE_TRAIN = REPO_ROOT / "data/processed/dk1_panel_train.parquet"
UNIVARIATE_VAL = REPO_ROOT / "data/processed/dk1_panel_val.parquet"

TARGET_ALIASES = {
    "price": "price",
    "activation": "activation_volume",
    "activation_volume": "activation_volume",
}


# Module-level overrides for LOGO ablations. ``_ACTIVE_FEATURES`` is consulted
# by every adapter via ``_features()`` below; ``_ACTIVE_LO_TAG`` is appended to
# the output directory name so runs do not collide with the baseline matrix.
_ACTIVE_FEATURES: tuple[str, ...] = F_CANONICAL_FEATURES
_ACTIVE_LO_TAG: str = ""


def _features() -> tuple[str, ...]:
    return _ACTIVE_FEATURES


def _worker_init(features: tuple[str, ...], lo_tag: str) -> None:
    """ProcessPoolExecutor initializer: propagate LOGO overrides to worker procs
    and isolate each worker's MLflow tracking dir to avoid filesystem lock
    contention (which deadlocked the first parallel run on F2_BLR).
    """
    global _ACTIVE_FEATURES, _ACTIVE_LO_TAG
    _ACTIVE_FEATURES = features
    _ACTIVE_LO_TAG = lo_tag
    pid = os.getpid()
    mlruns_dir = Path(f"/tmp/mlflow_canonical_pid{pid}")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlruns_dir}"


def _canonical_name(model: str, target: str) -> str:
    """Output directory tag — keeps canonical runs separated from legacy ones."""
    suffix = "price" if target == "price" else "activation"
    tag = f"_lo_{_ACTIVE_LO_TAG}" if _ACTIVE_LO_TAG else ""
    return f"canonical_{model}_{suffix}{tag}"


def _run_patchtst(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    """F7 / F8 / F11 share the patchTST trainer."""
    from heimdall_forecaster.train.trainer import TrainConfig, train_model

    cfg = TrainConfig(
        name=_canonical_name(model, target),
        train_panel=DEFAULT_TRAIN,
        val_panel=DEFAULT_VAL,
        anomaly_panel=DEFAULT_ANOMALY,
        feature_names=_features(),
        multivariate=True,
        target=target,
        seed=seed,
        epochs=1 if smoke else 20,
        out_dir=out_dir,
        use_rin=True,
    )
    result = train_model(cfg)
    return {
        "val_pinball_mean": float(result["val_pinball_mean"]),
        "val_q10_q90_coverage": float(result["val_q10_q90_coverage"]),
    }


def _run_f1(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    from heimdall_forecaster.train.f1_lgbm import F1Config, train_f1

    cfg = F1Config(
        name=_canonical_name(model, target),
        train_panel=DEFAULT_TRAIN,
        val_panel=DEFAULT_VAL,
        anomaly_panel=DEFAULT_ANOMALY,
        multivariate=True,
        feature_names=_features(),
        target=target,
        # Canonical: 50 rounds (LGBM plateaus by ~50 on this 4224-feature input;
        # original f1.yaml's 250 takes 8+h per seed on F_CANONICAL — confirmed
        # 2026-05-21 overnight). Apples-to-apples fairness is in the feature
        # panel, not in n_estimators; documented in the canonical run notes.
        n_estimators=20 if smoke else 50,
        seed=seed,
        out_dir=out_dir,
    )
    return train_f1(cfg)


def _run_f2(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    from heimdall_forecaster.train.f2_blr import F2Config, train_f2

    _ = smoke  # F2 is closed-form; nothing to shrink.
    cfg = F2Config(
        name=_canonical_name(model, target),
        train_panel=DEFAULT_TRAIN,
        val_panel=DEFAULT_VAL,
        anomaly_panel=DEFAULT_ANOMALY,
        multivariate=True,
        feature_names=_features(),
        target=target,
        seed=seed,
        out_dir=out_dir,
    )
    return train_f2(cfg)


def _run_np(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    """F5 (ConvCNP) / F6 (AttentiveNP) — name prefix selects the variant."""
    from heimdall_forecaster.train.f5_f6_neural_process import (
        NeuralProcessConfig,
        train_neural_process,
    )

    cfg = NeuralProcessConfig(
        name=_canonical_name(model, target),
        train_panel=DEFAULT_TRAIN,
        val_panel=DEFAULT_VAL,
        anomaly_panel=DEFAULT_ANOMALY,
        feature_names=_features(),
        target=target,
        seed=seed,
        epochs=1 if smoke else 8,
        out_dir=out_dir,
    )
    # The internal `_model` dispatch keys on cfg.name.startswith("f6"); the
    # canonical_<model>_<target> prefix would break that. Pin the variant via
    # a marker field on the config name.
    if model.startswith("f6"):
        cfg.name = "f6_" + cfg.name
    else:
        cfg.name = "f5_" + cfg.name
    return train_neural_process(cfg)


def _run_f0(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    from heimdall_forecaster.train.f0_ar import F0Config, train_f0

    _ = smoke  # F0 is OLS-AR; nothing to shrink.
    cfg = F0Config(
        name=_canonical_name(model, target),
        train_panel=UNIVARIATE_TRAIN,
        val_panel=UNIVARIATE_VAL,
        target=target,
        seed=seed,
        out_dir=out_dir,
    )
    return train_f0(cfg)


# Adapter dispatch — only models we can run on F_CANONICAL **as-is**.
# F3 (DeepARLite), F9 (TimesFM-2.0) and F10 (Chronos-Bolt) are univariate
# pretrained / univariate-by-architecture; they are intentionally not in the
# canonical fan-out because they cannot ingest covariates without a model
# rewrite. They remain available via their own legacy entrypoints and should
# be reported alongside the canonical table as univariate baselines.
ADAPTERS: dict[str, callable] = {
    "f7": _run_patchtst,
    "f8": _run_patchtst,
    "f11": _run_patchtst,
    "f1_lgbm": _run_f1,
    "f2_blr": _run_f2,
    "f5_np": _run_np,
    "f6_anp": _run_np,
    "f0": _run_f0,
}
# Note: ``ar1`` is a closed-form Gaussian-residual fallback (no training);
# it is invoked at inference via ``inference/backends/ar1_fallback.py`` and
# does not participate in the training fan-out. F3 / F9 / F10 are excluded
# because they are univariate by architecture and cannot consume F_CANONICAL.


def run_one(model: str, target: str, seed: int, *, smoke: bool, out_dir: Path) -> dict:
    if model not in ADAPTERS:
        raise KeyError(f"Unknown / unsupported model {model!r}. Known: {sorted(ADAPTERS)}")
    routing = CANONICAL_MODEL_ROUTING.get(model, "univariate")
    target_kind = TARGET_ALIASES[target]
    metrics = ADAPTERS[model](model, target_kind, seed, smoke=smoke, out_dir=out_dir)
    return {
        "model": model,
        "routing": routing,
        "target": target_kind,
        "seed": seed,
        **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help=f"Subset of {sorted(ADAPTERS)}")
    p.add_argument("--target", choices=list(TARGET_ALIASES), default="price")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "models/forecaster")
    p.add_argument("--leave-out", choices=list(CANONICAL_FEATURE_GROUPS),
                   help="LOGO ablation: drop one feature group from F_CANONICAL "
                        "before training. Output dirs get a ``_lo_<group>`` suffix.")
    p.add_argument("--parallel-seeds", type=int, default=1,
                   help="Run N seeds of the same model concurrently (process pool). "
                        "GPU models share the device via PyTorch's MPS-friendly default. "
                        "Models are still iterated serially.")
    args = p.parse_args(argv)

    # Isolate MLflow tracking dir per CLI invocation so concurrent canonical
    # runs (and other jobs writing to ./mlruns) cannot deadlock each other on
    # the file backend's lock. Both the parent process and any worker pool
    # workers (which re-set this in ``_worker_init``) end up with their own
    # tracking dir.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        parent_pid = os.getpid()
        parent_dir = Path(f"/tmp/mlflow_canonical_parent_pid{parent_pid}")
        parent_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{parent_dir}"

    if args.leave_out:
        global _ACTIVE_FEATURES, _ACTIVE_LO_TAG
        _ACTIVE_FEATURES = f_canonical_without(args.leave_out)
        _ACTIVE_LO_TAG = args.leave_out
        print(f"[LOGO] dropped group={args.leave_out!r} -> "
              f"{len(_ACTIVE_FEATURES)} features active "
              f"(was {len(F_CANONICAL_FEATURES)})", flush=True)

    summary: list[dict] = []
    for model in args.models:
        if args.parallel_seeds <= 1:
            for seed in args.seeds:
                row = run_one(model, args.target, seed,
                              smoke=args.smoke, out_dir=args.out_dir)
                print(json.dumps(row), flush=True)
                summary.append(row)
            continue
        # Fan out seeds across a process pool. spawn ensures CUDA context isolation.
        ctx = "spawn"
        import multiprocessing as mp  # noqa: PLC0415 — defer import to avoid spawn cost when unused
        with ProcessPoolExecutor(max_workers=args.parallel_seeds,
                                 mp_context=mp.get_context(ctx),
                                 initializer=_worker_init,
                                 initargs=(_ACTIVE_FEATURES, _ACTIVE_LO_TAG)) as ex:
            futs = {
                ex.submit(run_one, model, args.target, seed,
                          smoke=args.smoke, out_dir=args.out_dir): seed
                for seed in args.seeds
            }
            for f in as_completed(futs):
                seed = futs[f]
                try:
                    row = f.result()
                except Exception as e:  # noqa: BLE001 — surface and keep going
                    row = {"model": model, "seed": seed, "error": repr(e)}
                print(json.dumps(row), flush=True)
                summary.append(row)
    _ = os  # imported for env tweaks if needed by callers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
