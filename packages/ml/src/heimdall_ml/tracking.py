"""MLflow tracking helper.

Every experiment (forecaster training, conformal calibration, ablation cell,
counterfactual run) MUST log via this module. Centralising the contract here
keeps the `mlruns/` directory at a single repo-root location and prevents
"un-logged shadow runs" — which §10 explicitly rules out.
"""

from __future__ import annotations

import os
import platform
import resource
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

# Resolve the repo root via this file's location: packages/ml/src/heimdall_ml/tracking.py
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
DEFAULT_TRACKING_URI = (_REPO_ROOT / "mlruns").as_uri()


_INITIALISED = False


def init(tracking_uri: str | None = None, experiment: str = "heimdall") -> None:
    """Idempotently configure MLflow's tracking URI + active experiment.

    Behaviour:
      - If `tracking_uri` is provided, set it (last call wins; supports tests
        that pin a tmp path).
      - On first call without `tracking_uri`, set the env-var URI or the
        repo-root `mlruns/` default. Subsequent calls without a URI leave
        the tracking URI alone.
    """
    global _INITIALISED
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        _INITIALISED = True
    elif not _INITIALISED:
        uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(uri)
        _INITIALISED = True
    mlflow.set_experiment(experiment)


@contextmanager
def run(
    name: str,
    *,
    experiment: str | None = None,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
):
    """Standard run context. Always sets seeds tag if `seed` is in params.

    If `experiment` is None we leave the active experiment as set by the
    most recent `init(...)` call; otherwise we re-init for that experiment
    while preserving the tracking URI.
    """
    if experiment is not None:
        init(experiment=experiment)
    with mlflow.start_run(run_name=name) as active:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
            if "seed" in params:
                mlflow.set_tag("heimdall.seed", str(params["seed"]))
        yield active


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


# ---- compute/energy footprint (per docs/RESEARCH-PROPOSAL.md §10) ---------------


@contextmanager
def track_compute(
    name: str,
    *,
    gpu_index: int | None = 0,
    grid_intensity_g_co2_per_kwh: float = 90.0,
):
    """Lightweight GPU-energy / wall-time tracker.

    Wraps a code block: samples nvidia-smi power draw at start/end, computes
    average draw × wall time → kWh, multiplies by ``grid_intensity`` (Danish
    grid 2025: ~90 g CO₂/kWh, see Energinet declaration) → CO₂e, and logs
    the trio (kWh, CO₂e, wall_seconds) to the active MLflow run.

    Falls back gracefully when no GPU / nvidia-smi is available — the block
    still runs and only wall-time is logged.
    """
    import shutil
    import subprocess
    import time

    smi = shutil.which("nvidia-smi") if gpu_index is not None else None

    def _power_w() -> float | None:
        if smi is None:
            return None
        try:
            out = subprocess.check_output(
                [smi, f"-i={gpu_index}", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"], text=True, timeout=2.0,
            ).strip()
            return float(out.splitlines()[0])
        except Exception:
            return None

    p_start = _power_w()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        p_end = _power_w()
        metrics = {f"heimdall.{name}.wall_seconds": float(wall)}
        if p_start is not None and p_end is not None:
            avg_w = 0.5 * (p_start + p_end)
            kwh = avg_w * wall / 3600.0 / 1000.0
            co2e_g = kwh * grid_intensity_g_co2_per_kwh
            metrics.update({
                f"heimdall.{name}.gpu_kwh": round(float(kwh), 6),
                f"heimdall.{name}.co2e_grams": round(float(co2e_g), 4),
                f"heimdall.{name}.gpu_avg_w": round(float(avg_w), 2),
            })
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            # No active run → just print (still better than silent loss).
            print(metrics)


@contextmanager
def track_experiment_compute(
    name: str,
    *,
    gpu_indices: tuple[int, ...] | None = (0,),
    grid_intensity_g_co2_per_kwh: float = 90.0,
):
    """Wall/GPU/host compute accounting for thesis experiments.

    Uses ``nvidia-smi`` when available and degrades to CPU/wall/RSS metrics on
    CPU-only machines. All values are logged to the active MLflow run.
    """
    import shutil
    import subprocess
    import time

    smi = shutil.which("nvidia-smi") if gpu_indices else None

    def _power_w(index: int) -> float | None:
        if smi is None:
            return None
        try:
            out = subprocess.check_output(
                [
                    smi,
                    f"-i={index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2.0,
            ).strip()
            return float(out.splitlines()[0])
        except Exception:
            return None

    indices = tuple(gpu_indices or ())
    p_start = {idx: _power_w(idx) for idx in indices}
    ru_start = resource.getrusage(resource.RUSAGE_SELF)
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    try:
        try:
            mlflow.set_tags(
                {
                    f"heimdall.{name}.host": socket.gethostname(),
                    f"heimdall.{name}.platform": platform.platform(),
                    f"heimdall.{name}.gpu_indices": ",".join(str(i) for i in indices),
                    f"heimdall.{name}.grid_intensity_g_co2_per_kwh": str(
                        grid_intensity_g_co2_per_kwh
                    ),
                }
            )
        except Exception:
            pass
        yield
    finally:
        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        ru_end = resource.getrusage(resource.RUSAGE_SELF)
        metrics = {
            f"heimdall.{name}.wall_seconds": float(wall),
            f"heimdall.{name}.cpu_seconds": float(cpu),
            f"heimdall.{name}.max_rss_mb": float(ru_end.ru_maxrss) / 1024.0,
            f"heimdall.{name}.user_cpu_seconds": float(ru_end.ru_utime - ru_start.ru_utime),
            f"heimdall.{name}.system_cpu_seconds": float(ru_end.ru_stime - ru_start.ru_stime),
        }
        total_kwh = 0.0
        sampled = 0
        for idx in indices:
            p0 = p_start.get(idx)
            p1 = _power_w(idx)
            if p0 is None or p1 is None:
                continue
            avg_w = 0.5 * (p0 + p1)
            kwh = avg_w * wall / 3600.0 / 1000.0
            total_kwh += kwh
            sampled += 1
            metrics[f"heimdall.{name}.gpu{idx}_avg_w"] = round(float(avg_w), 2)
            metrics[f"heimdall.{name}.gpu{idx}_kwh"] = round(float(kwh), 6)
        metrics[f"heimdall.{name}.gpu_sampled_count"] = float(sampled)
        metrics[f"heimdall.{name}.gpu_kwh"] = round(float(total_kwh), 6)
        metrics[f"heimdall.{name}.co2e_grams"] = round(
            float(total_kwh * grid_intensity_g_co2_per_kwh), 4
        )
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            print(metrics)
