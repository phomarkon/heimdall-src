"""PatchTST hurdle for activation forecasting (GPU multi-task).

Multi-task PatchTST head per direction:
  - Stage A: H sigmoid logits (event probability per horizon)
  - Stage B: H*Q quantile outputs (volume conditional on event)
Loss = BCE(stage_a) + alpha * masked_pinball(stage_b, mask=event)

Trains 5 seeds × 2 directions on B200. Two trainers run in parallel on a
single GPU (each ~30-50 GB). Writes outputs/hurdle_patchtst/{train,test}_leaderboard.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
OUT_BASE = REPO / "outputs"
MODEL_BASE = REPO / "models/forecaster"
# Variant config: (name, use_rin, d_model, n_layers)
VARIANTS = {
    "f8": ("hurdle_patchtst", True, 128, 6),
    "f7": ("hurdle_f7_patchtst_norin", False, 128, 6),
    "f11": ("hurdle_f11_patchtst_deep", True, 192, 10),
}
VARIANT = "f8"  # overridden by --variant CLI
OUT = OUT_BASE / VARIANTS[VARIANT][0]
MODEL_DIR = MODEL_BASE / VARIANTS[VARIANT][0]
SEEDS = (13, 42, 137, 1729, 31415)
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 96
HORIZON = 16
PATCH_LEN = 8
D_MODEL = 128
NHEAD = 8
N_LAYERS = 6
DROPOUT = 0.1
BATCH = 256
EPOCHS = 20
LR = 1e-3
ALPHA_REG = 0.5  # weight for stage-B pinball relative to BCE
THRESH = 0.0


class PatchTSTHurdle(nn.Module):
    def __init__(self, n_features: int, *, use_rin: bool = True,
                 d_model: int = 128, n_layers: int = 6):
        super().__init__()
        self.use_rin = use_rin
        if use_rin:
            self.rin_gamma = nn.Parameter(torch.ones(n_features))
            self.rin_beta = nn.Parameter(torch.zeros(n_features))
        self.patch = nn.Conv1d(n_features, d_model, kernel_size=PATCH_LEN, stride=PATCH_LEN)
        n_patches = SEQ_LEN // PATCH_LEN
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, NHEAD, 4 * d_model, DROPOUT,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        flat = d_model * n_patches
        self.head_cls = nn.Linear(flat, HORIZON)
        self.head_reg = nn.Linear(flat, HORIZON * len(QUANTILES))

    def forward(self, x):
        if self.use_rin:
            m = x.mean(1, keepdim=True); s = x.std(1, keepdim=True).clamp(min=1e-5)
            x = (x - m) / s * self.rin_gamma + self.rin_beta
        h = self.patch(x.transpose(1, 2)).transpose(1, 2) + self.pos
        h = self.norm(self.encoder(h))
        flat = h.reshape(h.shape[0], -1)
        logits = self.head_cls(flat)
        qout = self.head_reg(flat).reshape(-1, HORIZON, len(QUANTILES))
        return logits, qout


def _windows(panel_path, anom_path, *, feature_names, target_col):
    df = pl.read_parquet(panel_path).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    if anom_path.exists():
        anom = pl.read_parquet(anom_path).with_columns(
            pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
        )
        df = df.join(anom, on="timestamp_utc", how="left").fill_null(0.0)
    have = [c for c in feature_names if c in df.columns]
    X = df.select(have).to_numpy().astype(np.float32)
    y = df.select(target_col).to_numpy().astype(np.float32).ravel()
    n = len(y) - SEQ_LEN - HORIZON + 1
    Xw = np.zeros((n, SEQ_LEN, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, HORIZON), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i + SEQ_LEN]
        Yw[i] = y[i + SEQ_LEN:i + SEQ_LEN + HORIZON]
    return Xw, Yw


def _pinball_masked(pred, target, mask, quantiles):
    # pred: (B, H, Q), target: (B, H), mask: (B, H)
    target = target.unsqueeze(-1)
    err = target - pred
    qs = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    loss = torch.maximum(qs * err, (qs - 1.0) * err)  # (B, H, Q)
    m = mask.unsqueeze(-1).expand_as(loss)
    n = m.sum().clamp(min=1.0)
    return (loss * m).sum() / n


def _joint_pinball_eval(logits, qout, target, level=0.5):
    p = torch.sigmoid(logits)
    q50 = qout[..., len(QUANTILES) // 2]
    err_pos = target - q50
    pin_pos = torch.maximum(level * err_pos, (level - 1.0) * err_pos)
    pin_zero = torch.maximum(level * target, (level - 1.0) * target)
    return (p * pin_pos + (1 - p) * pin_zero).mean()


def _train_one(seed: int, direction: str, X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
               gpu_id: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(f"cuda:{gpu_id}")
    _, use_rin, d_model_, n_layers_ = VARIANTS[VARIANT]
    model = PatchTSTHurdle(n_features=X_tr.shape[-1], use_rin=use_rin,
                           d_model=d_model_, n_layers=n_layers_).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Global z-score normalisation on X
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(0).clip(min=1e-5)
    Xtr_n = (X_tr - mu) / sd; Xva_n = (X_va - mu) / sd; Xte_n = (X_te - mu) / sd
    # Target normalisation for stage B (per-direction global mean/std of nonzero)
    mask_pos_tr = Y_tr > THRESH
    if mask_pos_tr.any():
        ymu = Y_tr[mask_pos_tr].mean(); ysd = Y_tr[mask_pos_tr].std().clip(min=1e-5)
    else:
        ymu, ysd = 0.0, 1.0
    Ytr_n = (Y_tr - ymu) / ysd; Yva_n = (Y_va - ymu) / ysd; Yte_n = (Y_te - ymu) / ysd

    bce = nn.BCEWithLogitsLoss()
    n = X_tr.shape[0]
    best_val = float("inf"); best_state = None
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)
        losses = []
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            xb = torch.from_numpy(Xtr_n[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(Y_tr[idx]).to(device, non_blocking=True)
            yb_n = torch.from_numpy(Ytr_n[idx]).to(device, non_blocking=True)
            event = (yb > THRESH).float()
            logits, qout = model(xb)
            loss_a = bce(logits, event)
            loss_b = _pinball_masked(qout, yb_n, event, QUANTILES)
            loss = loss_a + ALPHA_REG * loss_b
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # Val
        model.eval()
        with torch.no_grad():
            ps, qs_, ys = [], [], []
            for i in range(0, X_va.shape[0], BATCH):
                xb = torch.from_numpy(Xva_n[i:i + BATCH]).to(device)
                logits, qout = model(xb)
                ps.append(logits.cpu()); qs_.append(qout.cpu())
            logits_va = torch.cat(ps); qout_va = torch.cat(qs_)
            yva = torch.from_numpy(Y_va)
            # Joint pinball on REAL scale
            q50_real = qout_va[..., 1] * ysd + ymu
            p_va = torch.sigmoid(logits_va)
            err = yva - q50_real
            pin_pos = torch.maximum(0.5 * err, -0.5 * err)
            pin_zero = torch.maximum(0.5 * yva, -0.5 * yva)
            val_joint = float((p_va * pin_pos + (1 - p_va) * pin_zero).mean())
        if val_joint < best_val:
            best_val = val_joint
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[ptst-h] seed={seed} dir={direction} gpu={gpu_id} ep={ep} train={np.mean(losses):.3f} val_joint={val_joint:.3f}",
              flush=True)
    model.load_state_dict(best_state)

    # Test eval (single shot)
    model.eval()
    with torch.no_grad():
        ps, qs_ = [], []
        for i in range(0, X_te.shape[0], BATCH):
            xb = torch.from_numpy(Xte_n[i:i + BATCH]).to(device)
            logits, qout = model(xb)
            ps.append(logits.cpu()); qs_.append(qout.cpu())
        logits_te = torch.cat(ps); qout_te = torch.cat(qs_)
        yte = torch.from_numpy(Y_te)
        q50_real = qout_te[..., 1] * ysd + ymu
        p_te = torch.sigmoid(logits_te)
        err = yte - q50_real
        pin_pos = torch.maximum(0.5 * err, -0.5 * err)
        pin_zero = torch.maximum(0.5 * yte, -0.5 * yte)
        test_joint = float((p_te * pin_pos + (1 - p_te) * pin_zero).mean())
        # AUC + Brier per horizon
        events = (yte > THRESH).float()
        probs = p_te
        brier = float(((probs - events) ** 2).mean())
        # Simple macro AUC across horizons
        aucs = []
        for h in range(HORIZON):
            y_h = events[:, h].numpy(); p_h = probs[:, h].numpy()
            pos = y_h.sum(); neg = len(y_h) - pos
            if pos == 0 or neg == 0: continue
            order = np.argsort(p_h)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            aucs.append((ranks[y_h == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
        auc_mean = float(np.mean(aucs)) if aucs else float("nan")

    md = MODEL_DIR / f"{direction}_seed-{seed}"
    md.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, md / "model.pt")
    metrics = {
        "seed": seed, "direction": direction,
        "val_joint_pinball": best_val,
        "test_joint_pinball": test_joint,
        "test_brier": brier,
        "test_auc_mean": auc_mean,
        "ymu": float(ymu), "ysd": float(ysd),
    }
    (md / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--directions", nargs="+", default=["up", "down"])
    p.add_argument("--max-workers", type=int, default=2, help="Parallel GPU trainers")
    p.add_argument("--variant", choices=list(VARIANTS), default="f8")
    args = p.parse_args(argv)
    global VARIANT, OUT, MODEL_DIR
    VARIANT = args.variant
    OUT = OUT_BASE / VARIANTS[VARIANT][0]
    MODEL_DIR = MODEL_BASE / VARIANTS[VARIANT][0]

    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES
    OUT.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    test_panel = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
    anom_train = REPO / "data/processed/anomaly_features_train.parquet"
    anom_val = REPO / "data/processed/anomaly_features_val.parquet"
    anom_test = REPO / "data/processed/anomaly_features_test.parquet"

    data: dict[str, tuple] = {}
    for d in args.directions:
        tcol = f"mfrr_{d}_volume_mw"
        print(f"[ptst-h] windows dir={d}", flush=True)
        X_tr, Y_tr = _windows(train_panel, anom_train, feature_names=F_CANONICAL_FEATURES, target_col=tcol)
        X_va, Y_va = _windows(val_panel, anom_val, feature_names=F_CANONICAL_FEATURES, target_col=tcol)
        X_te, Y_te = _windows(test_panel, anom_test, feature_names=F_CANONICAL_FEATURES, target_col=tcol)
        print(f"  Xtr={X_tr.shape} Xva={X_va.shape} Xte={X_te.shape}", flush=True)
        data[d] = (X_tr, Y_tr, X_va, Y_va, X_te, Y_te)

    jobs = [(seed, d) for d in args.directions for seed in args.seeds]
    print(f"[ptst-h] {len(jobs)} jobs, {args.max_workers} parallel GPU trainers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        futs = {}
        for k, (seed, d) in enumerate(jobs):
            gpu_id = 0  # only one B200 visible
            futs[exe.submit(_train_one, seed, d, *data[d], gpu_id)] = (seed, d)
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            print(f"[ptst-h] DONE seed={r['seed']} dir={r['direction']} "
                  f"val={r['val_joint_pinball']:.3f} test={r['test_joint_pinball']:.3f} "
                  f"auc={r['test_auc_mean']:.3f}", flush=True)

    lb = []
    for d in args.directions:
        sub = [r for r in results if r["direction"] == d]
        lb.append({
            "direction": d, "n_seeds": len(sub),
            "val_joint_pinball_mean": float(np.mean([r["val_joint_pinball"] for r in sub])),
            "val_joint_pinball_std": float(np.std([r["val_joint_pinball"] for r in sub])),
            "test_joint_pinball_mean": float(np.mean([r["test_joint_pinball"] for r in sub])),
            "test_joint_pinball_std": float(np.std([r["test_joint_pinball"] for r in sub])),
            "test_auc_mean": float(np.mean([r["test_auc_mean"] for r in sub])),
            "test_brier_mean": float(np.mean([r["test_brier"] for r in sub])),
        })
    (OUT / "leaderboard.json").write_text(json.dumps(lb, indent=2))
    (OUT / "per_direction.json").write_text(json.dumps(results, indent=2))
    print(f"[ptst-h] wrote {OUT}/leaderboard.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
