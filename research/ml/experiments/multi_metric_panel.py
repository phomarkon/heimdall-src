"""Phase 6: Multi-metric fair comparison panel.

For every forecaster in the zoo, computes:
- pinball loss (per-quantile + mean) [DKK]
- CRPS — Continuous Ranked Probability Score (approximated from quantiles)
- Winkler interval score at α=0.10 (90% interval)
- Raw [q10, q90] band coverage
- ACI coverage + mean width
- Left-tail miscoverage (P[y < q10])
- Right-tail miscoverage (P[y > q90])
- Pred std (sanity check vs collapse)

All measured from val_preds.npz; deterministic. Output:
  experiments/outputs/multi_metric_panel.json
  notes/forecaster_multi_metric.md
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / "models/forecaster"
QUANTILES = (0.1, 0.5, 0.9)
ALPHA = 0.10


def _pinball(y, q, level):
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _crps_from_quantiles(y, preds_quantiles, levels):
    """Approximate CRPS from a small set of quantile predictions.

    For each y, CRPS ≈ Σ_q (q_i - y_i)·(F_indicator - q_level)·dt  but with
    a finite set of q-levels we use the discrete-CRPS = pinball-mean × 2.
    (Wilks 2011 §7.6.1 and Gneiting & Raftery 2007 show: for 3 quantiles
    at {0.1, 0.5, 0.9}, the trapezoidal approximation to CRPS equals
    2 × mean pinball loss within a multiplicative constant.)
    """
    per_q = [_pinball(y, preds_quantiles[..., qi], q) for qi, q in enumerate(levels)]
    return 2.0 * float(np.mean(per_q))


def _winkler_score(y, lower, upper, alpha):
    """Winkler interval score (Gneiting & Raftery 2007 §5.1)."""
    width = upper - lower
    below = (lower - y) * (y < lower) * (2.0 / alpha)
    above = (y - upper) * (y > upper) * (2.0 / alpha)
    return float(np.mean(width + below + above))


def _metrics_for_seed(npz_path: Path) -> dict | None:
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    preds = z["preds"].astype(np.float64)
    targets = z["targets"].astype(np.float64)
    sorted_p = np.sort(preds, axis=-1)
    lower, upper = sorted_p[..., 0], sorted_p[..., -1]
    per_q = {f"pinball_q{int(q*100)}": _pinball(targets, preds[..., qi], q)
             for qi, q in enumerate(QUANTILES)}
    pinball_mean = float(np.mean(list(per_q.values())))
    raw_cov = float(np.mean((targets >= lower) & (targets <= upper)))
    crps = _crps_from_quantiles(targets, sorted_p, QUANTILES)
    winkler = _winkler_score(targets, lower, upper, ALPHA)
    # Tail miscoverage
    below_q10 = float(np.mean(targets < sorted_p[..., 0]))
    above_q90 = float(np.mean(targets > sorted_p[..., -1]))
    pred_std = float(preds[:, 0, 1].std())
    return {
        **per_q,
        "pinball_mean": pinball_mean,
        "crps": crps,
        "winkler_90": winkler,
        "raw_cov": raw_cov,
        "below_q10_freq": below_q10,
        "above_q90_freq": above_q90,
        "pred_std": pred_std,
        "n_windows": int(preds.shape[0]),
    }


def _agg(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if isinstance(m.get(k), (int, float))]
        if not vals:
            continue
        if len(vals) > 1:
            out[k + "_mean"] = float(mean(vals))
            out[k + "_std"] = float(stdev(vals))
        else:
            out[k + "_mean"] = float(vals[0])
            out[k + "_std"] = 0.0
    return out


def main() -> int:
    panel = {}
    for md in sorted(MODEL_ROOT.iterdir()):
        if not md.is_dir() or md.name == "release":
            continue
        per_seed = []
        for sd in sorted(md.glob("seed-*")):
            m = _metrics_for_seed(sd / "val_preds.npz")
            if m is not None:
                per_seed.append(m)
        if not per_seed:
            continue
        # Pull ACI from existing metrics.json (already computed)
        aci_vals = []
        for sd in sorted(md.glob("seed-*")):
            mp = sd / "metrics.json"
            if mp.exists():
                d = json.loads(mp.read_text())
                if "aci_empirical_coverage" in d:
                    aci_vals.append(d["aci_empirical_coverage"])
        agg = _agg(per_seed)
        agg["n_seeds"] = len(per_seed)
        if aci_vals:
            agg["aci_cov_mean"] = float(mean(aci_vals))
            agg["aci_cov_std"] = float(stdev(aci_vals)) if len(aci_vals) > 1 else 0.0
        panel[md.name] = agg

    out_path = REPO / "experiments/outputs/multi_metric_panel.json"
    out_path.write_text(json.dumps(panel, indent=2))

    # Markdown ranked by pinball_mean
    md_lines = ["# Multi-metric forecaster panel (Phase 6)", "",
                "All metrics on val window, mean ± std over 5 frozen seeds (where applicable).",
                "Pinball / CRPS / Winkler in DKK/MWh, lower=better. Coverage in fraction (target 0.90).",
                "",
                "| model | n | pinball | CRPS | Winkler-90 | raw cov | ACI cov | pred std (q50) | below_q10 | above_q90 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def _fmt(d: dict, k: str, prec: int = 1) -> str:
        m = d.get(k + "_mean")
        s = d.get(k + "_std")
        if m is None: return "—"
        if s and d["n_seeds"] > 1: return f"{m:.{prec}f} ± {s:.{prec}f}"
        return f"{m:.{prec}f}"

    rows = sorted(panel.items(), key=lambda x: x[1].get("pinball_mean_mean", 1e9))
    for name, p in rows:
        md_lines.append(
            f"| {name} | {p['n_seeds']} | {_fmt(p, 'pinball_mean')} | {_fmt(p, 'crps')} | "
            f"{_fmt(p, 'winkler_90')} | {_fmt(p, 'raw_cov', 3)} | {_fmt(p, 'aci_cov', 3)} | "
            f"{_fmt(p, 'pred_std', 3)} | {_fmt(p, 'below_q10_freq', 3)} | {_fmt(p, 'above_q90_freq', 3)} |"
        )
    md = REPO / "notes/forecaster_multi_metric.md"
    md.write_text("\n".join(md_lines))
    print(f"Wrote {out_path}")
    print(f"Wrote {md}")
    print(f"\nTop 5 by pinball:")
    for name, p in rows[:5]:
        print(f"  {name:25s} pinball={p.get('pinball_mean_mean', 0):.1f} CRPS={p.get('crps_mean', 0):.1f} "
              f"Winkler={p.get('winkler_90_mean', 0):.1f} pred_std={p.get('pred_std_mean', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
