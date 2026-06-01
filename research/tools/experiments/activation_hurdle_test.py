"""Single-shot test-set eval for the activation hurdle model (§5.7).

Loads stage A + stage B boosters from models/forecaster/hurdle/{dir}_seed-*/,
scores on dk1_panel_rich_v2_test.parquet, writes
outputs/hurdle/test_leaderboard.json + outputs/hurdle/test_per_direction.json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/hurdle"
MODEL_DIR = REPO / "models/forecaster/hurdle"
SEEDS = (13, 42, 137, 1729, 31415)
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 96
HORIZON = 16
THRESH = 0.0


def _windows(panel_path: Path, anom_path: Path, *, feature_names, target_col):
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


def _score(direction: str, seed: int, Xt, Yt):
    sd = MODEL_DIR / f"{direction}_seed-{seed}"
    a = pickle.load(open(sd / "stage_a.pkl", "rb"))
    b = pickle.load(open(sd / "stage_b.pkl", "rb"))
    n, sl, f = Xt.shape
    Xf = Xt.reshape(n, sl * f)
    horizon_metrics = []
    joint = []
    for h in range(HORIZON):
        key = f"h{h}"
        if key not in a:
            continue
        clf = lgb.Booster(model_str=a[key])
        p = clf.predict(Xf)
        y_event = (Yt[:, h] > THRESH).astype(np.int32)
        brier = float(np.mean((p - y_event) ** 2))
        pos = y_event.sum(); neg = len(y_event) - pos
        if pos > 0 and neg > 0:
            order = np.argsort(p)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            auc = float((ranks[y_event == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
        else:
            auc = float("nan")
        pred = (p >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (y_event == 1)).sum())
        fp = int(((pred == 1) & (y_event == 0)).sum())
        fn = int(((pred == 0) & (y_event == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        # Stage B
        bk_q50 = f"h{h}_q50"
        if bk_q50 not in b:
            horizon_metrics.append({"h": h, "brier": brier, "auc": auc, "f1": f1,
                                    "pos_rate_test": float(y_event.mean())})
            continue
        preds_b = {q: lgb.Booster(model_str=b[f"h{h}_q{int(q*100)}"]).predict(Xf).astype(np.float32)
                   for q in QUANTILES}
        mask = y_event == 1
        cond_pinballs = []
        if mask.any():
            yt = Yt[mask, h]
            for q in QUANTILES:
                err = yt - preds_b[q][mask]
                cond_pinballs.append(float(np.mean(np.maximum(q*err, (q-1.0)*err))))
        # Joint expected pinball at q50
        q50 = preds_b[0.5]
        err_pos = Yt[:, h] - q50
        pinball_pos = np.maximum(0.5*err_pos, -0.5*err_pos)
        pinball_zero = np.maximum(0.5*Yt[:, h], -0.5*Yt[:, h])
        joint.append(float(np.mean(p * pinball_pos + (1-p) * pinball_zero)))
        horizon_metrics.append({"h": h, "brier": brier, "auc": auc, "f1": f1,
                                "pos_rate_test": float(y_event.mean()),
                                "cond_pinball_mean": float(np.mean(cond_pinballs)) if cond_pinballs else None,
                                "n_event_test": int(mask.sum())})
    return {
        "seed": seed, "direction": direction,
        "horizon_metrics": horizon_metrics,
        "joint_pinball_mean": float(np.mean(joint)) if joint else float("nan"),
        "stage_a_auc_mean": float(np.nanmean([m["auc"] for m in horizon_metrics if "auc" in m])),
        "stage_a_brier_mean": float(np.mean([m["brier"] for m in horizon_metrics])),
        "stage_a_f1_mean": float(np.mean([m["f1"] for m in horizon_metrics])),
    }


def main() -> int:
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES
    OUT.mkdir(parents=True, exist_ok=True)
    test_panel = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
    anom_test = REPO / "data/processed/anomaly_features_test.parquet"
    if not anom_test.exists():
        anom_test = REPO / "data/processed/anomaly_features.parquet"

    rows = []
    for direction in ("up", "down"):
        target_col = f"mfrr_{direction}_volume_mw"
        print(f"[hurdle-test] windows direction={direction}", flush=True)
        Xt, Yt = _windows(test_panel, anom_test, feature_names=F_CANONICAL_FEATURES,
                          target_col=target_col)
        print(f"  Xt={Xt.shape} event_rate={(Yt>THRESH).any(axis=1).mean():.3f}", flush=True)
        for seed in SEEDS:
            r = _score(direction, seed, Xt, Yt)
            rows.append(r)
            print(f"  seed={seed} dir={direction} joint_pinball={r['joint_pinball_mean']:.3f} "
                  f"AUC={r['stage_a_auc_mean']:.3f} F1={r['stage_a_f1_mean']:.3f}", flush=True)

    leaderboard = []
    for direction in ("up", "down"):
        sub = [r for r in rows if r["direction"] == direction]
        leaderboard.append({
            "direction": direction,
            "n_seeds": len(sub),
            "test_joint_pinball_mean": float(np.mean([r["joint_pinball_mean"] for r in sub])),
            "test_joint_pinball_std": float(np.std([r["joint_pinball_mean"] for r in sub])),
            "test_stage_a_auc_mean": float(np.mean([r["stage_a_auc_mean"] for r in sub])),
            "test_stage_a_brier_mean": float(np.mean([r["stage_a_brier_mean"] for r in sub])),
            "test_stage_a_f1_mean": float(np.mean([r["stage_a_f1_mean"] for r in sub])),
        })
    (OUT / "test_leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    (OUT / "test_per_direction.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"[hurdle-test] wrote {OUT}/test_leaderboard.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
