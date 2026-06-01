"""Fresh test-set evaluation for the canonical (apples-to-apples) zoo.

Independent of the legacy ``experiments/test_set_evaluation.py`` ledger —
re-loads each ``models/forecaster/canonical_<model>_<target>/seed-<S>/``
checkpoint, runs inference on the held-out test panel (post 2025-05-01),
and writes a fresh leaderboard ``outputs/test_canonical/leaderboard.json``.

Authorized re-run per user directive 2026-05-22: the new canonical training
runs need first-time test-set evaluation; this is not a re-evaluation of
prior configs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics as st
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/test_canonical"
TEST_PANEL_RICH = REPO / "data/processed/dk1_panel_rich_v2_test.parquet"
ANOM_TEST = REPO / "data/processed/anomaly_features_test.parquet"
TEST_PANEL_MIN = REPO / "data/processed/dk1_panel_test.parquet"
QUANTILES = (0.1, 0.5, 0.9)
TARGET_PRICE = "imbalance_price_dkk_mwh_15min"

CANON_RE = re.compile(r"^(?:f5_|f6_)?canonical_(?P<model>.+?)_(?P<target>price|activation)(?:_lo_.+)?$")


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _load_test_panel(target: str, feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(TEST_PANEL_RICH).sort("timestamp_utc")
    anom_df = pl.read_parquet(ANOM_TEST)
    df = df.join(anom_df, on="timestamp_utc", how="left")
    # rich_v2_test.parquet dropped mfrr_up/down volumes during regeneration —
    # for activation target we hydrate them from the v1 rich panel which still
    # has them. Same timestamps so the join is exact.
    if target != "price":
        v1 = pl.read_parquet(REPO / "data/processed/dk1_panel_rich_test.parquet")
        if "mfrr_up_volume_mw" in v1.columns and "mfrr_down_volume_mw" in v1.columns:
            df = df.drop("mfrr_up_volume_mw", "mfrr_down_volume_mw").join(
                v1.select(["timestamp_utc", "mfrr_up_volume_mw", "mfrr_down_volume_mw"]),
                on="timestamp_utc", how="left",
            )
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise RuntimeError(f"test panel missing canonical cols: {missing}")
    X = df.select(list(feature_names)).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    if target == "price":
        y = df[TARGET_PRICE].to_numpy().astype(np.float64)
    else:
        up = df["mfrr_up_volume_mw"].to_numpy().astype(np.float64)
        dn = df["mfrr_down_volume_mw"].to_numpy().astype(np.float64)
        y = (up - dn) * 0.25
    return X, y


def _slide_windows(X: np.ndarray, y: np.ndarray, *,
                    seq_len: int, horizon: int, stats) -> tuple[np.ndarray, np.ndarray]:
    n_full = max(0, X.shape[0] - seq_len - horizon)
    Xw = np.empty((n_full, seq_len, X.shape[1]), dtype=np.float64)
    Yw = np.empty((n_full, horizon), dtype=np.float64)
    for i in range(n_full):
        Xw[i] = X[i : i + seq_len]
        Yw[i] = y[i + seq_len : i + seq_len + horizon]
    # Apply frozen train stats (same normalisation as training).
    Xw_n = (Xw - stats.mean) / np.maximum(stats.std, 1e-6)
    return Xw_n, Yw


# ----------------------------------------------------------------------------
# patchTST loader (canonical_f7/f8/f11 share this).
# ----------------------------------------------------------------------------

def _eval_patchtst(seed_dir: Path, X_test: np.ndarray, y_test: np.ndarray, stats) -> dict:
    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES  # noqa: PLC0415
    from heimdall_forecaster.train.model import PatchTransformerQuantile  # noqa: PLC0415

    cfg = json.loads((seed_dir / "config.json").read_text())
    Xw, Yw = _slide_windows(X_test, y_test,
                            seq_len=cfg["seq_len"], horizon=cfg["horizon"], stats=stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerQuantile(
        n_features=Xw.shape[-1], seq_len=cfg["seq_len"], horizon=cfg["horizon"],
        n_quantiles=3, patch_len=cfg["patch_len"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        use_rin=True,
    ).to(device).eval()
    sd = torch.load(seed_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    preds = []
    with torch.no_grad():
        for i in range(0, Xw.shape[0], 256):
            t = torch.from_numpy(Xw[i:i + 256]).float().to(device)
            preds.append(model(t).cpu().numpy())
    P = np.concatenate(preds, axis=0)
    P_dn = stats.denormalise_target(P)
    return _score(P_dn, Yw)


def _eval_lgbm(seed_dir: Path, X_test: np.ndarray, y_test: np.ndarray, stats) -> dict:
    import lightgbm as lgb  # noqa: PLC0415

    with open(seed_dir / "boosters.pkl", "rb") as fh:
        boosters = pickle.load(fh)
    # F1_LGBM trainer doesn't write config.json; fall back to canonical SEQ_LEN/HORIZON.
    cfg = (json.loads((seed_dir / "config.json").read_text())
           if (seed_dir / "config.json").exists() else {"seq_len": 96, "horizon": 16})
    Xw, Yw = _slide_windows(X_test, y_test,
                            seq_len=cfg["seq_len"], horizon=cfg["horizon"], stats=stats)
    n, sl, f = Xw.shape
    Xf = Xw.reshape(n, sl * f).astype(np.float32)
    P = np.zeros((n, cfg["horizon"], 3), dtype=np.float32)
    for h in range(cfg["horizon"]):
        for qi, q in enumerate(QUANTILES):
            booster = lgb.Booster(model_str=boosters[f"h{h}_q{int(q * 100)}"])
            P[:, h, qi] = booster.predict(Xf).astype(np.float32)
    return _score(P, Yw)


def _eval_blr(seed_dir: Path, X_test: np.ndarray, y_test: np.ndarray, stats) -> dict:
    with open(seed_dir / "regressors.pkl", "rb") as fh:
        regs = pickle.load(fh)
    cfg = (json.loads((seed_dir / "config.json").read_text())
           if (seed_dir / "config.json").exists() else {"seq_len": 96, "horizon": 16})
    Xw, Yw = _slide_windows(X_test, y_test,
                            seq_len=cfg["seq_len"], horizon=cfg["horizon"], stats=stats)
    n, sl, f = Xw.shape
    Xf = Xw.reshape(n, sl * f).astype(np.float32)
    from scipy.stats import norm  # noqa: PLC0415
    z_levels = np.array([norm.ppf(q) for q in QUANTILES], dtype=np.float32)
    P = np.zeros((n, cfg["horizon"], 3), dtype=np.float32)
    for h in range(cfg["horizon"]):
        reg = regs[f"h{h}"]
        mu, sigma = reg.predict(Xf, return_std=True)
        # Predictions are in normalized target units; denormalise.
        mu_dn = mu * stats.target_std + stats.target_mean
        sigma_dn = sigma * stats.target_std
        for qi, z in enumerate(z_levels):
            P[:, h, qi] = mu_dn + z * sigma_dn
    # Already denormalised.
    return _score(P, Yw, already_denorm=True)


def _score(P: np.ndarray, y: np.ndarray, *, already_denorm: bool = False) -> dict:
    P_dn = P  # caller hands us either denorm or about-to-denorm
    per_q = {f"test_pinball_q{int(q * 100)}": _pinball(y, P_dn[..., qi], q)
             for qi, q in enumerate(QUANTILES)}
    srt = np.sort(P_dn, axis=-1)
    cov = float(np.mean((y >= srt[..., 0]) & (y <= srt[..., -1])))
    return {
        "test_n_windows": int(P_dn.shape[0]),
        **per_q,
        "test_pinball_mean_dkk": float(np.mean(list(per_q.values()))),
        "test_q10_q90_coverage": cov,
    }


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------

PATCHTST_PREFIXES = ("canonical_f7_", "canonical_f8_", "canonical_f11_")


def _classify(model_dir: Path) -> str | None:
    if (model_dir / "seed-42/boosters.pkl").exists():
        return "lgbm"
    if (model_dir / "seed-42/regressors.pkl").exists():
        return "blr"
    if (model_dir / "seed-42/model.pt").exists():
        # Distinguish patchTST from F5/F6 NP variants which use different model classes.
        try:
            sd = torch.load(model_dir / "seed-42/model.pt",
                            map_location="cpu", weights_only=True)
            keys = set(sd.keys())
            if "rin_gamma" in keys and "head.weight" in keys:
                return "patchtst"
            return "other_pt"   # F12 EDM / F5 NP / F6 ANP — skip in this runner
        except Exception:
            return "other_pt"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO / "models/forecaster")
    p.add_argument("--target", choices=["price", "activation", "both"], default="price")
    args = p.parse_args(argv)

    from heimdall_forecaster.train.dataset import F_CANONICAL_FEATURES  # noqa: PLC0415

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    targets = ("price", "activation") if args.target == "both" else (args.target,)
    for tgt in targets:
        X_test, y_test = _load_test_panel(tgt, F_CANONICAL_FEATURES)
        for sub in sorted(args.root.iterdir()):
            if not sub.is_dir():
                continue
            m = CANON_RE.match(sub.name)
            if not m or m["target"] != tgt or "_lo_" in sub.name:
                continue
            kind = _classify(sub)
            if kind is None:
                continue
            for sd in sorted(sub.glob("seed-*/")):
                try:
                    with open(sd / "stats.pkl", "rb") as fh:
                        stats = pickle.load(fh)
                except FileNotFoundError:
                    continue
                try:
                    if kind == "patchtst":
                        score = _eval_patchtst(sd, X_test, y_test, stats)
                    elif kind == "lgbm":
                        score = _eval_lgbm(sd, X_test, y_test, stats)
                    elif kind == "blr":
                        score = _eval_blr(sd, X_test, y_test, stats)
                    else:
                        # other_pt = EDM / NP — not supported by this lightweight
                        # evaluator (different architectures). Skip cleanly.
                        rows.append({"model": sub.name, "seed": sd.name,
                                     "target": tgt,
                                     "error": f"unsupported kind={kind}"})
                        continue
                    score.update({"model": sub.name, "kind": kind,
                                  "seed": sd.name, "target": tgt})
                    rows.append(score)
                    print(f"{sub.name:<35s} {sd.name:<12s} kind={kind:<8s} "
                          f"pinball={score['test_pinball_mean_dkk']:.1f} "
                          f"cov={score['test_q10_q90_coverage']:.2f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{sub.name}/{sd.name} ERROR: {e!r}", flush=True)
                    rows.append({"model": sub.name, "seed": sd.name,
                                 "target": tgt, "error": repr(e)})

    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    # Aggregate per (model, target) — mean +- pstdev over seeds, only successful rows.
    agg: dict[str, list[float]] = {}
    for r in rows:
        if "error" in r:
            continue
        v = r.get("test_pinball_mean_dkk")
        if v is None or not np.isfinite(v):
            continue
        k = f"{r['model']}__{r['target']}"
        agg.setdefault(k, []).append(float(v))
    leaderboard = []
    for k, vs in agg.items():
        leaderboard.append({
            "model_target": k,
            "n_seeds": len(vs),
            "test_pinball_mean": float(np.mean(vs)),
            "test_pinball_std": float(np.std(vs)) if len(vs) > 1 else 0.0,
        })
    leaderboard.sort(key=lambda r: r["test_pinball_mean"])
    (OUT / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    print(f"\nwrote {len(rows)} rows, {len(leaderboard)} aggregated entries -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
