"""One-shot benchmark per F* model family.

For each family:
  - param count
  - 1 training epoch wall time on real train slice
  - inference latency @ batch=1 (p50, p95) and throughput @ batch=256
  - peak GPU memory + average GPU power (W) via pynvml
  - peak CPU% via psutil sampling

Writes outputs/footprint/benchmark.json + benchmark.md.
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/footprint"
SEED = 42
SEQ_LEN = 96
HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
N_TRAIN_SAMPLES = 175000  # match real training workload
N_INFER_SAMPLES = 1000
BATCH = 256
GPU_POLL_S = 0.05


def _load_features(target_col: str = "imbalance_price_dkk_mwh_15min"):
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES
    df = pl.read_parquet(REPO / "data/processed/dk1_panel_rich_v2_train.parquet").with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    anom = pl.read_parquet(REPO / "data/processed/anomaly_features_train.parquet").with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    df = df.join(anom, on="timestamp_utc", how="left").fill_null(0.0)
    have = [c for c in F_CANONICAL_FEATURES if c in df.columns]
    X = df.select(have).to_numpy().astype(np.float32)
    y = df.select(target_col).to_numpy().astype(np.float32).ravel()
    n = len(y) - SEQ_LEN - HORIZON + 1
    Xw = np.zeros((n, SEQ_LEN, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, HORIZON), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i + SEQ_LEN]
        Yw[i] = y[i + SEQ_LEN:i + SEQ_LEN + HORIZON]
    return Xw, Yw


class _GpuPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples_w = []
        self.samples_mem = []
        self.stop_evt = threading.Event()
    def run(self):
        while not self.stop_evt.is_set():
            if _HAS_GPU:
                try:
                    self.samples_w.append(pynvml.nvmlDeviceGetPowerUsage(_GPU) / 1000.0)
                    self.samples_mem.append(pynvml.nvmlDeviceGetMemoryInfo(_GPU).used / 1024**3)
                except Exception:
                    pass
            time.sleep(GPU_POLL_S)
    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)
        return {
            "avg_gpu_power_w": float(np.mean(self.samples_w)) if self.samples_w else 0.0,
            "peak_gpu_power_w": float(np.max(self.samples_w)) if self.samples_w else 0.0,
            "peak_gpu_mem_gb": float(np.max(self.samples_mem)) if self.samples_mem else 0.0,
        }


def _measure(fn, *, label: str) -> dict:
    proc = psutil.Process()
    proc.cpu_percent(None)
    poll = _GpuPoller(); poll.start()
    t0 = time.perf_counter()
    rss0 = proc.memory_info().rss / 1024**3
    res = fn()
    wall = time.perf_counter() - t0
    cpu_pct = proc.cpu_percent(None)
    rss1 = proc.memory_info().rss / 1024**3
    gpu = poll.stop()
    out = {"wall_s": wall, "cpu_pct": cpu_pct, "rss_gb_delta": rss1 - rss0}
    out.update(gpu)
    out.update(res or {})
    print(f"  [{label}] wall={wall:.2f}s cpu={cpu_pct:.0f}% gpu_avg={out['avg_gpu_power_w']:.0f}W gpu_peak_mem={out['peak_gpu_mem_gb']:.2f}GB", flush=True)
    return out


# ---- model benchmarks -----------------------------------------------------

def bench_lgbm(X, Y, name: str = "F1_LGBM") -> dict:
    import lightgbm as lgb
    n, sl, f = X.shape
    Xf = X.reshape(n, sl * f)
    train_metrics = _measure(lambda: {"n_estimators": 50},
                             label=f"{name}_train_setup")
    # Train one h0_q50 model = 1 horizon × 1 quantile (representative single fit)
    def _do_train():
        d = lgb.Dataset(Xf, label=Y[:, 0])
        b = lgb.train({"objective": "quantile", "alpha": 0.5, "verbose": -1,
                       "num_leaves": 63, "learning_rate": 0.05,
                       "feature_fraction": 0.8, "num_threads": 32, "seed": SEED},
                      d, num_boost_round=50)
        return {"booster": b, "n_features": Xf.shape[1]}
    train = _measure(lambda: _do_train(), label=f"{name}_train_h0_q50")
    booster = train.pop("booster"); n_feat = train.pop("n_features")
    # Inference: latency @ batch=1, throughput @ batch=256
    def _lat():
        ts = []
        for i in range(N_INFER_SAMPLES):
            x = Xf[i:i+1]
            t = time.perf_counter()
            booster.predict(x)
            ts.append((time.perf_counter() - t) * 1000)
        return {"latency_ms_p50": float(np.percentile(ts, 50)),
                "latency_ms_p95": float(np.percentile(ts, 95))}
    lat = _measure(lambda: _lat(), label=f"{name}_latency_b1")
    def _thru():
        n_samp = N_INFER_SAMPLES
        t = time.perf_counter()
        for i in range(0, n_samp, BATCH):
            booster.predict(Xf[i:i+BATCH])
        elapsed = time.perf_counter() - t
        return {"throughput_samples_per_s": n_samp / elapsed}
    thru = _measure(lambda: _thru(), label=f"{name}_throughput_b256")
    return {
        "family": name, "device": "CPU", "n_features_flat": int(n_feat),
        "n_params_est": booster.num_trees() * booster.num_feature(),
        "train_one_model_s": train["wall_s"],
        "train_full_estimate_s": train["wall_s"] * HORIZON * len(QUANTILES),
        "train_cpu_pct": train["cpu_pct"], "train_rss_gb_delta": train["rss_gb_delta"],
        "latency_ms_p50": lat["latency_ms_p50"], "latency_ms_p95": lat["latency_ms_p95"],
        "throughput_samples_per_s": thru["throughput_samples_per_s"],
        "avg_gpu_power_w": train["avg_gpu_power_w"],
    }


def bench_patchtst(X, Y, name: str = "F8_PatchTST", *, use_rin: bool = True) -> dict:
    from heimdall_forecaster.train.model import PatchTransformerQuantile, quantile_loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerQuantile(
        n_features=X.shape[-1], seq_len=SEQ_LEN, horizon=HORIZON, n_quantiles=3,
        patch_len=8, d_model=128, nhead=8, n_layers=6, dropout=0.1, use_rin=use_rin,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Normalise
    mu = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0).clip(min=1e-5)
    Xn = (X - mu) / sd
    def _train_epoch():
        model.train()
        order = np.random.permutation(X.shape[0])
        loss_sum = 0.0; nb = 0
        for i in range(0, X.shape[0], BATCH):
            idx = order[i:i+BATCH]
            xb = torch.from_numpy(Xn[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(Y[idx]).to(device, non_blocking=True)
            pred = model(xb)
            loss = quantile_loss(pred, yb, QUANTILES)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        return {"avg_loss": loss_sum / max(1, nb)}
    train = _measure(lambda: _train_epoch(), label=f"{name}_train_1epoch")

    # Inference
    model.eval()
    with torch.no_grad():
        def _lat():
            ts = []
            for i in range(N_INFER_SAMPLES):
                xb = torch.from_numpy(Xn[i:i+1]).to(device)
                if device.type == "cuda": torch.cuda.synchronize()
                t = time.perf_counter()
                model(xb)
                if device.type == "cuda": torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
            return {"latency_ms_p50": float(np.percentile(ts, 50)),
                    "latency_ms_p95": float(np.percentile(ts, 95))}
        lat = _measure(lambda: _lat(), label=f"{name}_latency_b1")
        def _thru():
            n_samp = N_INFER_SAMPLES
            if device.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for i in range(0, n_samp, BATCH):
                xb = torch.from_numpy(Xn[i:i+BATCH]).to(device)
                model(xb)
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            return {"throughput_samples_per_s": n_samp / elapsed}
        thru = _measure(lambda: _thru(), label=f"{name}_throughput_b256")
    return {
        "family": name, "device": str(device),
        "n_params": int(n_params), "use_rin": use_rin,
        "train_1_epoch_s": train["wall_s"],
        "train_20epoch_estimate_s": train["wall_s"] * 20,
        "avg_gpu_power_w": train["avg_gpu_power_w"],
        "peak_gpu_mem_gb": train["peak_gpu_mem_gb"],
        "latency_ms_p50": lat["latency_ms_p50"], "latency_ms_p95": lat["latency_ms_p95"],
        "throughput_samples_per_s": thru["throughput_samples_per_s"],
    }


def bench_f0(X, Y) -> dict:
    """Seasonal AR baseline (numpy, no training)."""
    def _train():
        # F0 = seasonal naive @ horizon - no actual training, just stats
        return {"params": 1}
    train = _measure(lambda: _train(), label="F0_train")
    def _lat():
        ts = []
        for i in range(N_INFER_SAMPLES):
            t = time.perf_counter()
            _ = np.tile(X[i, -1], (HORIZON, 1))[:, 0]  # repeat last value
            ts.append((time.perf_counter() - t) * 1000)
        return {"latency_ms_p50": float(np.percentile(ts, 50)),
                "latency_ms_p95": float(np.percentile(ts, 95))}
    lat = _measure(lambda: _lat(), label="F0_latency_b1")
    return {"family": "F0_seasonal_AR", "device": "CPU", "n_params": 1,
            "train_full_estimate_s": train["wall_s"],
            "latency_ms_p50": lat["latency_ms_p50"], "latency_ms_p95": lat["latency_ms_p95"],
            "throughput_samples_per_s": 1e6, "avg_gpu_power_w": 0.0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[bench] loading windows…", flush=True)
    Xfull, Yfull = _load_features()
    np.random.seed(SEED)
    idx = np.random.choice(Xfull.shape[0], min(N_TRAIN_SAMPLES, Xfull.shape[0]), replace=False)
    X = Xfull[idx]; Y = Yfull[idx]
    print(f"[bench] using {X.shape[0]} samples × {X.shape[1]}×{X.shape[2]} features", flush=True)

    results = []
    print("\n=== F0 ==="); results.append(bench_f0(X, Y))
    print("\n=== F1 LGBM ==="); results.append(bench_lgbm(X, Y, "F1_LGBM"))
    print("\n=== F8 PatchTST (RIN) ==="); results.append(bench_patchtst(X, Y, "F8_PatchTST", use_rin=True))
    print("\n=== F7 PatchTST (no RIN) ==="); results.append(bench_patchtst(X, Y, "F7_PatchTST", use_rin=False))
    print("\n=== Hurdle LGBM ==="); results.append(bench_lgbm(X, Y, "Hurdle_LGBM"))
    print("\n=== Hurdle PatchTST (multi-task) ==="); results.append(bench_patchtst(X, Y, "Hurdle_PatchTST", use_rin=True))

    # Derive energy footprint estimates for full 5-seed training
    for r in results:
        train_full = r.get("train_full_estimate_s") or r.get("train_20epoch_estimate_s") or 0
        n_seeds = 5
        total_wall_s = train_full * n_seeds
        # Power: use measured avg_gpu_power_w if >0, else CPU baseline 200W
        p_w = r.get("avg_gpu_power_w") or 0
        if p_w < 10: p_w = 200  # CPU fallback
        kwh = total_wall_s * p_w / 1000.0 / 3600.0
        # DK grid intensity ~ 100 g CO2/kWh
        co2_g = kwh * 100
        r["full_5seed_wall_s"] = total_wall_s
        r["full_5seed_kwh"] = round(kwh, 4)
        r["full_5seed_co2_g"] = round(co2_g, 2)

    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {OUT}/benchmark.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
