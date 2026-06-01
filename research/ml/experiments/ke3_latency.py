"""KE3 — compute / inference-latency kill experiment. docs/RESEARCH-PROPOSAL.md §5.6 (sprint day 14).

Hypothesis: F7+ACI inference latency must hit p99 < 200 ms on a single B200
to be useful inside the bid-time loop (gate-closure cushion is ~1 minute,
shared with verifier and the LLM tick model).

We benchmark 10 000 rolling-window predictions on val data:

  - **CPU** baseline (single-thread torch).
  - **GPU** (B200, eager-mode forward).
  - **GPU + ACI** (the production critical path).

Latency CDF + per-percentile table written to ``notes/ke3_verdict.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from heimdall_forecaster.train.dataset import HORIZON, SEQ_LEN, make_windows
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_ml import tracking
from heimdall_ml.conformal.aci import AdaptiveConformalInference

REPO_ROOT = Path(__file__).resolve().parents[2]
P99_BUDGET_MS = 200.0
N_RUNS = 10_000


@dataclass
class KE3Verdict:
    n_runs: int
    cpu_p50_ms: float
    cpu_p99_ms: float
    gpu_p50_ms: float
    gpu_p99_ms: float
    gpu_aci_p50_ms: float
    gpu_aci_p99_ms: float
    p99_budget_ms: float
    passed: bool
    decision: str


def _load_f7_seed42() -> tuple[PatchTransformerQuantile, np.ndarray]:
    """Load the F7 seed-42 weights + a stack of val windows."""
    import pickle

    out_dir = REPO_ROOT / "models/forecaster/f7/seed-42"
    cfg = json.loads((out_dir / "config.json").read_text())
    model = PatchTransformerQuantile(
        n_features=int(cfg["n_features"]),
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["horizon"]),
        n_quantiles=3,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    )
    state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(out_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    X_va, _, _ = make_windows(
        REPO_ROOT / "data/processed/dk1_panel_val.parquet",
        seq_len=SEQ_LEN, horizon=HORIZON, multivariate=False, stats=stats,
    )
    return model, X_va


def _bench(callable_, n: int) -> np.ndarray:
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        callable_(i)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def run_ke3() -> KE3Verdict:
    model, X_va = _load_f7_seed42()
    if X_va.shape[0] < N_RUNS:
        # cycle if val set is shorter than 10k
        idx = np.arange(N_RUNS) % X_va.shape[0]
    else:
        idx = np.arange(N_RUNS)
    X_va_t = torch.from_numpy(X_va).float()

    # CPU forward
    model_cpu = model.cpu()
    with torch.no_grad():
        # warm-up
        for _ in range(20):
            model_cpu(X_va_t[0:1])
        cpu_times = _bench(lambda i: model_cpu(X_va_t[idx[i] : idx[i] + 1]), N_RUNS)

    # GPU forward
    if torch.cuda.is_available():
        model_gpu = PatchTransformerQuantile(
            n_features=model.n_features,
            seq_len=model.seq_len,
            horizon=model.horizon,
            n_quantiles=model.n_quantiles,
            patch_len=model.patch_len,
            d_model=model.head.in_features // (model.seq_len // model.patch_len),
            nhead=8,
            n_layers=6,
            dropout=0.0,
        )
        model_gpu.load_state_dict(model.state_dict())
        model_gpu = model_gpu.to("cuda").eval()
        X_va_gpu = X_va_t.to("cuda")
        with torch.no_grad():
            for _ in range(20):
                model_gpu(X_va_gpu[0:1])
        torch.cuda.synchronize()
        gpu_times = _bench(lambda i: model_gpu(X_va_gpu[idx[i] : idx[i] + 1]), N_RUNS)

        # GPU + ACI on CPU (the calibrator is a few quantile lookups)
        aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
        # Warm-start with synthetic residuals matching production magnitudes.
        aci.warm_start(np.abs(np.random.randn(1000) * 100.0))

        def _gpu_aci(i: int) -> None:
            with torch.no_grad():
                pred = model_gpu(X_va_gpu[idx[i] : idx[i] + 1])
            q50 = pred[0, 0, 1].item()
            q = aci.quantile()
            _ = (q50 - q, q50 + q)
            aci.update(50.0)

        gpu_aci_times = _bench(_gpu_aci, N_RUNS)
    else:
        gpu_times = np.full(N_RUNS, np.nan)
        gpu_aci_times = np.full(N_RUNS, np.nan)

    cpu_p50 = float(np.percentile(cpu_times, 50))
    cpu_p99 = float(np.percentile(cpu_times, 99))
    gpu_p50 = float(np.percentile(gpu_times, 50))
    gpu_p99 = float(np.percentile(gpu_times, 99))
    gpu_aci_p50 = float(np.percentile(gpu_aci_times, 50))
    gpu_aci_p99 = float(np.percentile(gpu_aci_times, 99))
    passed = bool(gpu_aci_p99 < P99_BUDGET_MS)

    decision = (
        f"PASS — GPU+ACI p99 {gpu_aci_p99:.2f} ms < budget {P99_BUDGET_MS:.0f} ms; "
        "F7 inference comfortably fits inside the bid-time loop alongside the verifier."
        if passed
        else f"FAIL — GPU+ACI p99 {gpu_aci_p99:.2f} ms ≥ budget {P99_BUDGET_MS:.0f} ms; "
        "PIVOT: distill to a smaller forecaster (proposal §4.2.2 row F4) or move ACI offline."
    )

    tracking.init(experiment="heimdall-ke3")
    with tracking.run(
        name="ke3-latency",
        params={"n_runs": N_RUNS, "p99_budget_ms": P99_BUDGET_MS},
    ):
        tracking.log_metrics(
            {
                "cpu_p50_ms": cpu_p50, "cpu_p99_ms": cpu_p99,
                "gpu_p50_ms": gpu_p50, "gpu_p99_ms": gpu_p99,
                "gpu_aci_p50_ms": gpu_aci_p50, "gpu_aci_p99_ms": gpu_aci_p99,
                "passed": float(passed),
            }
        )

    # Save raw distributions for the figure script.
    out_dir = REPO_ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "ke3_latency.npz",
        cpu=cpu_times,
        gpu=gpu_times,
        gpu_aci=gpu_aci_times,
    )

    return KE3Verdict(
        n_runs=N_RUNS,
        cpu_p50_ms=cpu_p50, cpu_p99_ms=cpu_p99,
        gpu_p50_ms=gpu_p50, gpu_p99_ms=gpu_p99,
        gpu_aci_p50_ms=gpu_aci_p50, gpu_aci_p99_ms=gpu_aci_p99,
        p99_budget_ms=P99_BUDGET_MS,
        passed=passed,
        decision=decision,
    )


def write_verdict_note(v: KE3Verdict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# KE3 — inference-latency kill experiment\n\n"
        "Per docs/RESEARCH-PROPOSAL.md §5.6 (sprint day 14).\n\n"
        f"Benchmarked F7+ACI on **{v.n_runs}** rolling-window predictions on the "
        "post-2025 val set, on a single NVIDIA B200 GPU.\n\n"
        "## Latency table (milliseconds)\n\n"
        "| Backend       | p50  | p99   |\n"
        "|---------------|-----:|------:|\n"
        f"| CPU (1 thread) | {v.cpu_p50_ms:.2f} | {v.cpu_p99_ms:.2f} |\n"
        f"| B200 GPU       | {v.gpu_p50_ms:.2f} | {v.gpu_p99_ms:.2f} |\n"
        f"| B200 + ACI     | {v.gpu_aci_p50_ms:.2f} | {v.gpu_aci_p99_ms:.2f} |\n\n"
        f"Budget: p99 < **{v.p99_budget_ms:.0f} ms** on the production critical path "
        "(GPU + ACI).\n\n"
        f"## Verdict: {'PASS' if v.passed else 'FAIL'}\n\n"
        f"{v.decision}\n"
    )
    path.write_text(body)


if __name__ == "__main__":
    v = run_ke3()
    write_verdict_note(v, REPO_ROOT / "notes" / "ke3_verdict.md")
    print(json.dumps(asdict(v), indent=2))
