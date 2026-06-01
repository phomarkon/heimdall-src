"""Consolidated XAI runner for the canonical (apples-to-apples) zoo.

Three explanation views, applied to the canonical winners:

1. **TreeSHAP on F1 LGBM (canonical winner, price)** — exact per-prediction
   Shapley values, fast for trees. Plus LGBM's native gain/split importance
   as a sanity check.
2. **Permutation importance** on F8 patchTST (canonical multivariate
   deep model). Uses the FROZEN train stats; shuffles one feature column at a
   time over the val window stream and reads off Δval pinball.
3. **LASSO sparsity** on the F_CANONICAL flattened feature matrix as the
   linear-floor baseline (Trinity-style).

Outputs land under ``outputs/xai/`` as JSON + Markdown, ready to drop into
the thesis Methods chapter.

Usage:
    uv run python tools/experiments/xai_canonical.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/xai"
SEED = 42


def _pinball_mean(y: np.ndarray, q: np.ndarray, levels=(0.1, 0.5, 0.9)) -> float:
    per = []
    for qi, lv in enumerate(levels):
        err = y - q[..., qi]
        per.append(float(np.mean(np.maximum(lv * err, (lv - 1.0) * err))))
    return float(np.mean(per))


# ----------------------------------------------------------------------------
# F1 LGBM: TreeSHAP + native importance.
# ----------------------------------------------------------------------------

def run_f1_lgbm_treeshap() -> dict:
    import lightgbm as lgb  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, SEQ_LEN, HORIZON, make_windows,
    )

    seed_dir = REPO / f"models/forecaster/canonical_f1_lgbm_price/seed-{SEED}"
    if not (seed_dir / "boosters.pkl").exists():
        print(f"[f1] missing boosters at {seed_dir}, skipping")
        return {}
    with open(seed_dir / "boosters.pkl", "rb") as fh:
        booster_strs = pickle.load(fh)
    with open(seed_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)

    # Same feature layout F1 was trained on: F_CANONICAL × seq_len.
    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    Xva, _Yva_norm, _ = make_windows(
        val_panel, seq_len=SEQ_LEN, horizon=HORIZON,
        multivariate=True, feature_names=F_CANONICAL_FEATURES,
        anomaly_panel_path=anom, stats=stats,
    )
    n_va, sl, f = Xva.shape
    Xf = Xva.reshape(n_va, sl * f).astype(np.float32)
    flat_names = [f"t-{sl - 1 - t}__{name}" for t in range(sl) for name in F_CANONICAL_FEATURES]

    # Use the median-quantile booster at horizon-1 as the representative model
    # for SHAP — gives one ranking instead of 48.
    key = "h0_q50"
    if key not in booster_strs:
        print(f"[f1] booster {key!r} missing")
        return {}
    booster = lgb.Booster(model_str=booster_strs[key])

    # Native importance (gain) aggregated to per-feature (sum over lags).
    raw_gain = booster.feature_importance(importance_type="gain")
    raw_split = booster.feature_importance(importance_type="split")
    agg_gain: dict[str, float] = {}
    agg_split: dict[str, float] = {}
    for i, fn in enumerate(flat_names):
        canon = fn.split("__", 1)[1]
        agg_gain[canon] = agg_gain.get(canon, 0.0) + float(raw_gain[i])
        agg_split[canon] = agg_split.get(canon, 0.0) + float(raw_split[i])

    # TreeSHAP on a 1000-window subsample for tractability.
    import shap  # noqa: PLC0415
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_va, size=min(1000, n_va), replace=False)
    expl = shap.TreeExplainer(booster)
    sv = expl.shap_values(Xf[idx])
    # Aggregate |SHAP| over lags to per-feature contributions.
    mean_abs = np.abs(sv).mean(axis=0)
    agg_shap: dict[str, float] = {}
    for i, fn in enumerate(flat_names):
        canon = fn.split("__", 1)[1]
        agg_shap[canon] = agg_shap.get(canon, 0.0) + float(mean_abs[i])

    ranking = sorted(
        F_CANONICAL_FEATURES,
        key=lambda c: agg_shap.get(c, 0.0),
        reverse=True,
    )
    out = {
        "model": "canonical_f1_lgbm_price", "seed": SEED,
        "booster_used": key,
        "method": "TreeSHAP on h0_q50 booster + native LGBM gain/split importance",
        "n_subsample": int(len(idx)),
        "ranking": [
            {
                "feature": c,
                "shap_mean_abs": agg_shap.get(c, 0.0),
                "lgb_gain": agg_gain.get(c, 0.0),
                "lgb_split": agg_split.get(c, 0.0),
            }
            for c in ranking
        ],
    }
    (OUT / "f1_lgbm_treeshap.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# F8 patchTST: permutation importance.
# ----------------------------------------------------------------------------

def run_patchtst_permutation(model_dir_name: str = "canonical_f8_price") -> dict:
    import torch  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, SEQ_LEN, HORIZON, make_windows,
    )
    from heimdall_forecaster.train.model import PatchTransformerQuantile  # noqa: PLC0415

    seed_dir = REPO / f"models/forecaster/{model_dir_name}/seed-{SEED}"
    if not (seed_dir / "model.pt").exists():
        print(f"[{model_dir_name}] missing checkpoint, skipping")
        return {}
    cfg = json.loads((seed_dir / "config.json").read_text())
    with open(seed_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)

    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    Xva, Yva_norm, _ = make_windows(
        val_panel, seq_len=SEQ_LEN, horizon=HORIZON,
        multivariate=True, feature_names=F_CANONICAL_FEATURES,
        anomaly_panel_path=anom, stats=stats,
    )
    Yva_dn = stats.denormalise_target(Yva_norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerQuantile(
        n_features=Xva.shape[-1], seq_len=cfg["seq_len"], horizon=cfg["horizon"],
        n_quantiles=3, patch_len=cfg["patch_len"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        use_rin=True,  # canonical models always use RIN
    ).to(device).eval()
    sd = torch.load(seed_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)

    def _eval(X: np.ndarray) -> float:
        with torch.no_grad():
            preds: list[np.ndarray] = []
            for i in range(0, X.shape[0], 256):
                t = torch.from_numpy(X[i:i + 256]).float().to(device)
                p = model(t).cpu().numpy()
                preds.append(p)
        p_dn = stats.denormalise_target(np.concatenate(preds, axis=0))
        return _pinball_mean(Yva_dn, p_dn)

    baseline = _eval(Xva)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for fi, fname in enumerate(F_CANONICAL_FEATURES):
        Xshuf = Xva.copy()
        perm = rng.permutation(Xshuf.shape[0])
        Xshuf[:, :, fi] = Xshuf[perm, :, fi]
        perturbed = _eval(Xshuf)
        rows.append({"feature": fname, "delta_pinball": float(perturbed - baseline),
                     "shuffled_pinball": perturbed})
    rows.sort(key=lambda r: -r["delta_pinball"])
    out = {
        "model": model_dir_name, "seed": SEED,
        "method": "Permutation importance (single-shuffle per feature)",
        "baseline_pinball": baseline,
        "runtime_seconds": time.time() - t0,
        "ranking": rows,
    }
    (OUT / f"{model_dir_name}_permutation.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# F2 BLR / linear floor: LASSO sparsity (Trinity-style).
# ----------------------------------------------------------------------------

def run_lasso_sparsity() -> dict:
    from sklearn.linear_model import LassoCV  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, SEQ_LEN, HORIZON, make_windows,
    )

    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    Xtr, Ytr_norm, stats = make_windows(
        train_panel, seq_len=SEQ_LEN, horizon=HORIZON,
        multivariate=True, feature_names=F_CANONICAL_FEATURES,
        anomaly_panel_path=anom,
    )
    # Target = horizon-1 step (closest to bid time), de-normalised.
    Y_h0 = stats.denormalise_target(Ytr_norm)[:, 0]
    n, sl, f = Xtr.shape
    Xf = Xtr.reshape(n, sl * f).astype(np.float32)
    flat_names = [f"t-{sl - 1 - t}__{name}" for t in range(sl) for name in F_CANONICAL_FEATURES]
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Xf)
    Xs = scaler.transform(Xf)
    lasso = LassoCV(cv=3, n_alphas=20, max_iter=2000, random_state=SEED, n_jobs=-1)
    lasso.fit(Xs, Y_h0)
    coefs = lasso.coef_
    nonzero_mask = np.abs(coefs) > 1e-8
    agg: dict[str, float] = {}
    for i, fn in enumerate(flat_names):
        canon = fn.split("__", 1)[1]
        agg[canon] = agg.get(canon, 0.0) + float(abs(coefs[i]))
    ranking = sorted(F_CANONICAL_FEATURES, key=lambda c: agg.get(c, 0.0), reverse=True)
    out = {
        "model": "lasso_floor_on_F_CANONICAL", "seed": SEED,
        "method": "LassoCV (3-fold) on standardized flattened feature matrix",
        "alpha_selected": float(lasso.alpha_),
        "n_nonzero_coefs": int(nonzero_mask.sum()),
        "n_total_coefs": int(coefs.size),
        "ranking": [
            {"feature": c, "abs_coef_sum": agg.get(c, 0.0)} for c in ranking
        ],
    }
    (OUT / "lasso_floor.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# F8 patchTST: Integrated Gradients (Captum).
# ----------------------------------------------------------------------------

def run_patchtst_integrated_gradients(model_dir_name: str = "canonical_f8_price",
                                       n_samples: int = 200) -> dict:
    """Captum IG on a 200-window subsample, attributing the q50 horizon-1 output.
    Aggregates |attr| over (lag, feature) to per-feature contribution.
    """
    import torch  # noqa: PLC0415
    from captum.attr import IntegratedGradients  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, SEQ_LEN, HORIZON, make_windows,
    )
    from heimdall_forecaster.train.model import PatchTransformerQuantile  # noqa: PLC0415

    seed_dir = REPO / f"models/forecaster/{model_dir_name}/seed-{SEED}"
    if not (seed_dir / "model.pt").exists():
        print(f"[ig] missing checkpoint {seed_dir}")
        return {}
    cfg = json.loads((seed_dir / "config.json").read_text())
    with open(seed_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)

    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    Xva, _, _ = make_windows(
        val_panel, seq_len=SEQ_LEN, horizon=HORIZON,
        multivariate=True, feature_names=F_CANONICAL_FEATURES,
        anomaly_panel_path=anom, stats=stats,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerQuantile(
        n_features=Xva.shape[-1], seq_len=cfg["seq_len"], horizon=cfg["horizon"],
        n_quantiles=3, patch_len=cfg["patch_len"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        use_rin=True,
    ).to(device).eval()
    sd = torch.load(seed_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(Xva.shape[0], size=min(n_samples, Xva.shape[0]), replace=False)
    Xs = torch.from_numpy(Xva[idx]).float().to(device).requires_grad_(True)
    # Captum wants a scalar per sample; attribute the q50 (index 1) at horizon-1.
    def _scalar(x):
        return model(x)[:, 0, 1]
    ig = IntegratedGradients(_scalar)
    attr, _ = ig.attribute(Xs, baselines=torch.zeros_like(Xs), n_steps=32,
                           return_convergence_delta=True)
    attr_np = attr.detach().cpu().numpy()
    # |attr| aggregated over time then averaged over samples → per-feature score.
    per_feat = np.abs(attr_np).sum(axis=1).mean(axis=0)
    rows = [{"feature": F_CANONICAL_FEATURES[i], "mean_abs_attr": float(per_feat[i])}
            for i in range(len(F_CANONICAL_FEATURES))]
    rows.sort(key=lambda r: -r["mean_abs_attr"])
    out = {
        "model": model_dir_name, "seed": SEED,
        "method": f"Captum IntegratedGradients on q50 horizon-1, n_steps=32, "
                  f"n_subsample={len(idx)}, baseline=zeros",
        "ranking": rows,
    }
    (OUT / f"{model_dir_name}_integrated_gradients.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# F1 LGBM: DiCE counterfactuals on spike vs normal.
# ----------------------------------------------------------------------------

def run_dice_counterfactuals(n_cf: int = 5) -> dict:
    """Threshold the F_CANONICAL panel into spike (>90th pct of train prices)
    vs normal, fit a LightGBM classifier, then ask DiCE for diverse minimal
    perturbations that flip class for n_cf spike instances.
    """
    import lightgbm as lgb  # noqa: PLC0415
    import dice_ml  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_FEATURES, TARGET_COL,
    )
    import polars as pl  # noqa: PLC0415

    # Use the rich_v2 panel directly (not windowed) for a tabular spike classifier.
    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    val_panel = REPO / "data/processed/dk1_panel_rich_v2_val.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"

    df_tr = pl.read_parquet(train_panel).join(
        pl.read_parquet(anom), on="timestamp_utc", how="left"
    ).drop_nulls(F_CANONICAL_FEATURES + (TARGET_COL,))
    df_va = pl.read_parquet(val_panel).join(
        pl.read_parquet(anom), on="timestamp_utc", how="left"
    ).drop_nulls(F_CANONICAL_FEATURES + (TARGET_COL,))

    cols = list(F_CANONICAL_FEATURES)
    spike_thresh = float(df_tr[TARGET_COL].quantile(0.90))
    y_tr = (df_tr[TARGET_COL] > spike_thresh).to_numpy().astype(int)
    y_va = (df_va[TARGET_COL] > spike_thresh).to_numpy().astype(int)
    X_tr = df_tr.select(cols).to_pandas().astype(float)
    X_va = df_va.select(cols).to_pandas().astype(float)

    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05,
                             random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_va) == y_va).mean())

    d = dice_ml.Data(dataframe=pd.concat([X_tr, pd.Series(y_tr, name="spike")], axis=1),
                     continuous_features=cols, outcome_name="spike")
    m = dice_ml.Model(model=clf, backend="sklearn", model_type="classifier")
    exp = dice_ml.Dice(d, m, method="random")

    rng = np.random.default_rng(SEED)
    spike_idx = np.where(y_va == 1)[0]
    if len(spike_idx) == 0:
        print("[dice] no spikes in val window — skipping")
        return {}
    pick = rng.choice(spike_idx, size=min(n_cf, len(spike_idx)), replace=False)
    cfs_per_instance = []
    for i in pick:
        query = X_va.iloc[[int(i)]]
        try:
            cf = exp.generate_counterfactuals(query, total_CFs=3, desired_class=0)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            # Report the delta in feature values from the spike to its nearest CF.
            deltas = (cf_df[cols].iloc[0] - query[cols].iloc[0]).to_dict()
            cfs_per_instance.append({
                "instance_idx": int(i),
                "original_pred_class": int(clf.predict(query)[0]),
                "cf_pred_class": int(clf.predict(cf_df[cols].iloc[[0]])[0]),
                "feature_deltas": {k: float(v) for k, v in deltas.items() if abs(v) > 1e-6},
            })
        except Exception as e:
            cfs_per_instance.append({"instance_idx": int(i), "error": repr(e)})

    # Aggregate: mean |delta| per feature across instances (Trinity-style).
    agg: dict[str, list[float]] = {c: [] for c in cols}
    for rec in cfs_per_instance:
        for k, v in rec.get("feature_deltas", {}).items():
            agg[k].append(abs(v))
    mean_abs = {c: float(np.mean(agg[c])) if agg[c] else 0.0 for c in cols}
    ranking = sorted(cols, key=lambda c: -mean_abs[c])

    out = {
        "method": "DiCE random counterfactuals on LGBM spike classifier (>90th pct)",
        "spike_threshold_dkk": spike_thresh,
        "classifier_val_accuracy": acc,
        "n_counterfactuals": len(cfs_per_instance),
        "per_instance": cfs_per_instance,
        "ranking_by_mean_abs_perturbation": [
            {"feature": c, "mean_abs_perturbation": mean_abs[c]} for c in ranking
        ],
    }
    (OUT / "dice_counterfactuals.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# ----------------------------------------------------------------------------
# Granger causality on anomaly features.
# ----------------------------------------------------------------------------

def run_granger_anomaly() -> dict:
    """For each anomaly feature, test whether it Granger-causes the imbalance
    price at lag 1-4, controlling for the price's own autoregressive structure.
    Defends the anomaly-feature inclusion against the "just noise" critique.
    """
    from statsmodels.tsa.stattools import grangercausalitytests  # noqa: PLC0415
    import polars as pl  # noqa: PLC0415
    from heimdall_forecaster.train.dataset import (  # noqa: PLC0415
        F_CANONICAL_ANOMALY_COLS, TARGET_COL,
    )

    train_panel = REPO / "data/processed/dk1_panel_rich_v2_train.parquet"
    anom = REPO / "data/processed/anomaly_features.parquet"
    df = pl.read_parquet(train_panel).join(
        pl.read_parquet(anom), on="timestamp_utc", how="left"
    ).drop_nulls([TARGET_COL] + list(F_CANONICAL_ANOMALY_COLS))

    y = df[TARGET_COL].to_numpy()
    results = []
    for col in F_CANONICAL_ANOMALY_COLS:
        x = df[col].to_numpy()
        # statsmodels expects (T, 2) with [y, x] columns; tests if x → y.
        try:
            arr = np.column_stack([y, x])
            # Downsample to make it tractable (~20k rows is plenty for stable p-values).
            if arr.shape[0] > 20000:
                step = arr.shape[0] // 20000
                arr = arr[::step]
            res = grangercausalitytests(arr, maxlag=4, verbose=False)
            row = {"feature": col}
            for lag in (1, 2, 3, 4):
                ftest = res[lag][0]["ssr_ftest"]
                row[f"lag{lag}_f"] = float(ftest[0])
                row[f"lag{lag}_p"] = float(ftest[1])
            results.append(row)
        except Exception as e:
            results.append({"feature": col, "error": repr(e)})
    results.sort(key=lambda r: r.get("lag1_p", 1.0))
    out = {
        "method": "Granger causality F-test (ssr_ftest), max lag 4, "
                  "subsampled to ≤20k rows on pre-2025-03-04 train window",
        "results": results,
    }
    (OUT / "granger_anomaly.json").write_text(json.dumps(out, indent=2))
    return out


def _md_summary(treeshap: dict, perm: dict, lasso: dict,
                ig: dict | None = None, dice: dict | None = None,
                granger: dict | None = None) -> str:
    lines = ["# Canonical-zoo XAI summary", ""]
    if treeshap:
        lines.append(f"## F1 LGBM TreeSHAP (canonical price winner, seed={treeshap['seed']})")
        lines.append("")
        lines.append("| rank | feature | mean(|SHAP|) | LGB gain | LGB split |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, r in enumerate(treeshap["ranking"][:15], 1):
            lines.append(f"| {i} | `{r['feature']}` | {r['shap_mean_abs']:.3f} | "
                         f"{r['lgb_gain']:.0f} | {r['lgb_split']:.0f} |")
        lines.append("")
    if perm:
        lines.append(f"## {perm['model']} permutation importance (seed={perm['seed']})")
        lines.append(f"Baseline val pinball: **{perm['baseline_pinball']:.2f}**")
        lines.append("")
        lines.append("| rank | feature | Δ pinball (shuffled − baseline) |")
        lines.append("|---:|---|---:|")
        for i, r in enumerate(perm["ranking"][:15], 1):
            lines.append(f"| {i} | `{r['feature']}` | {r['delta_pinball']:+.2f} |")
        lines.append("")
    if ig:
        lines.append(f"## {ig['model']} Integrated Gradients (Captum)")
        lines.append("")
        lines.append("| rank | feature | mean(|attr|) |")
        lines.append("|---:|---|---:|")
        for i, r in enumerate(ig["ranking"][:15], 1):
            lines.append(f"| {i} | `{r['feature']}` | {r['mean_abs_attr']:.3f} |")
        lines.append("")
    if dice:
        lines.append(f"## DiCE counterfactuals (spike vs normal, threshold {dice['spike_threshold_dkk']:.0f} DKK)")
        lines.append(f"Classifier val accuracy: **{dice['classifier_val_accuracy']:.3f}**, "
                     f"n counterfactuals: **{dice['n_counterfactuals']}**")
        lines.append("")
        lines.append("| rank | feature | mean perturbation to flip spike → normal |")
        lines.append("|---:|---|---:|")
        for i, r in enumerate(dice["ranking_by_mean_abs_perturbation"][:15], 1):
            lines.append(f"| {i} | `{r['feature']}` | {r['mean_abs_perturbation']:.3f} |")
        lines.append("")
    if granger:
        lines.append("## Granger causality: anomaly features → imbalance price")
        lines.append("")
        lines.append("| feature | lag-1 F | lag-1 p | lag-4 F | lag-4 p |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in granger["results"]:
            if "error" in r:
                continue
            lines.append(f"| `{r['feature']}` | {r['lag1_f']:.1f} | {r['lag1_p']:.3g} "
                         f"| {r['lag4_f']:.1f} | {r['lag4_p']:.3g} |")
        lines.append("")
    if lasso:
        lines.append("## LASSO sparsity on F_CANONICAL (linear floor)")
        lines.append(f"Selected α: **{lasso['alpha_selected']:.4f}**, "
                     f"non-zero coefs: **{lasso['n_nonzero_coefs']} / {lasso['n_total_coefs']}**")
        lines.append("")
        lines.append("| rank | feature | Σ|β| over lags |")
        lines.append("|---:|---|---:|")
        for i, r in enumerate(lasso["ranking"][:15], 1):
            lines.append(f"| {i} | `{r['feature']}` | {r['abs_coef_sum']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="skip TreeSHAP subsample expansion + skip LASSO")
    p.add_argument("--patchtst-model", default="canonical_f8_price")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("[xai] TreeSHAP on F1 LGBM ...")
    ts = run_f1_lgbm_treeshap()
    print("[xai] permutation importance on patchTST ...")
    pm = run_patchtst_permutation(args.patchtst_model)
    print("[xai] Integrated Gradients on patchTST ...")
    ig = run_patchtst_integrated_gradients(args.patchtst_model)
    print("[xai] DiCE counterfactuals on spike classifier ...")
    dice = run_dice_counterfactuals()
    print("[xai] Granger causality on anomaly features ...")
    granger = run_granger_anomaly()
    lasso = {}
    if not args.quick:
        print("[xai] LASSO sparsity on F_CANONICAL ...")
        lasso = run_lasso_sparsity()
    md = _md_summary(ts, pm, lasso, ig=ig, dice=dice, granger=granger)
    (OUT / "summary.md").write_text(md)
    print(f"wrote outputs to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
