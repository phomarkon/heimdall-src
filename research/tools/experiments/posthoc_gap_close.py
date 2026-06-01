"""Post-hoc gap-close analyses that need only the saved val_preds.npz files.

Runs in one pass over every ``models/forecaster/*/seed-*/val_preds.npz``:
  - Reliability diagram (multi-alpha empirical coverage).
  - Regime-stratified pinball + coverage (spike/normal/negative + wind/storm
    when context columns are present).
  - Long-run ACI rolling-window audit (Theorem 1b).

All output as JSON for downstream plotting / table-gen. No markdown chaff.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/posthoc"
SKIP_NAMES = {"README.md", "summary_all.json", "sweep_summary.json", "coverage_audit.json"}
LOGO_PAT = re.compile(r"_lo_")
ACT_PAT = re.compile(r"(?:^activation_|_activation(?:_|$))")
QUANTILES = (0.1, 0.5, 0.9)
ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.3)


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _empirical_coverage(targets: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((targets >= lo) & (targets <= hi)))


def _multi_alpha_from_q10_q90(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Treat the trained q10/q90 as a Gaussian-equivalent symmetric band and
    derive multi-alpha coverage by linearly scaling the half-width.

    This is the standard 'one-shot' reliability check for any model that only
    trained 3 quantiles — it asks: if we asked for alpha=0.05 / 0.10 / ... ,
    how often would the rescaled band actually contain the truth?
    """
    from scipy.stats import norm  # noqa: PLC0415

    q50 = preds[..., 1]
    half = (preds[..., 2] - preds[..., 0]) / 2.0  # |q90 - q10| / 2
    # Reference half-width at alpha=0.10 is the empirical half-band.
    ref_z = norm.ppf(1 - 0.05)  # 1.645 (q90 of N(0,1))
    coverages = {}
    for a in ALPHAS:
        z = norm.ppf(1 - a / 2.0)
        scale = z / ref_z
        hw = half * scale
        cov = _empirical_coverage(targets, q50 - hw, q50 + hw)
        coverages[f"alpha_{a:.2f}"] = {"target": 1 - a, "empirical": cov,
                                       "delta": cov - (1 - a)}
    return coverages


def _regime_split(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Split val by target magnitude into 4 regimes and report pinball/coverage."""
    out = {}
    spike_thr = float(np.quantile(targets, 0.95))
    neg_thr = 0.0
    masks = {
        "spike": targets > spike_thr,
        "negative": targets < neg_thr,
        "high_vol": np.abs(targets - np.median(targets)) > np.quantile(np.abs(targets - np.median(targets)), 0.9),
        "normal": (targets <= spike_thr) & (targets >= neg_thr),
    }
    for name, mask in masks.items():
        if mask.sum() < 5:
            out[name] = {"n": int(mask.sum())}
            continue
        per = [_pinball(targets[mask], preds[mask][..., qi], q) for qi, q in enumerate(QUANTILES)]
        srt = np.sort(preds[mask], axis=-1)
        cov = _empirical_coverage(targets[mask], srt[..., 0], srt[..., -1])
        out[name] = {
            "n": int(mask.sum()),
            "pinball_mean": float(np.mean(per)),
            "raw_q10_q90_coverage": cov,
        }
    return out


def _rolling_aci_audit(preds: np.ndarray, targets: np.ndarray, *,
                       alpha: float = 0.1, gamma: float = 0.05,
                       window: int = 96) -> dict:
    """Walk-forward ACI on the flat (N*H,) score stream. Reports rolling
    empirical coverage in `window`-step buckets and the final alpha_t drift.
    """
    q50 = preds[..., 1]
    half = (preds[..., 2] - preds[..., 0]) / 2.0
    scores = np.abs(targets - q50).reshape(-1)
    width = half.reshape(-1)
    n = scores.size
    if n < window * 2:
        return {"n": int(n), "skipped": "series too short"}
    # Init: empirical alpha quantile of half-widths on first window.
    q_init = float(np.quantile(scores[:window], 1 - alpha))
    a_t = alpha
    in_band = np.zeros(n, dtype=bool)
    a_trace = np.zeros(n, dtype=np.float64)
    for t in range(n):
        z = scores[t]
        hw = width[t] if width[t] > 0 else q_init
        in_band[t] = z <= hw
        # ACI update (Gibbs & Candes 2021).
        err_t = 0.0 if in_band[t] else 1.0
        a_t = float(np.clip(a_t + gamma * (alpha - err_t), 1e-3, 1 - 1e-3))
        a_trace[t] = a_t
    # Bucketed coverage.
    edges = np.linspace(0, n, num=9, dtype=int)
    buckets = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        buckets.append({
            "start": s, "end": e, "n": e - s,
            "empirical_coverage": float(in_band[s:e].mean()) if e > s else float("nan"),
            "alpha_t_mean": float(a_trace[s:e].mean()) if e > s else float("nan"),
        })
    return {
        "n": int(n), "alpha_target": alpha, "gamma": gamma,
        "alpha_t_final": float(a_t),
        "overall_empirical_coverage": float(in_band.mean()),
        "buckets": buckets,
    }


def analyse_one(npz_path: Path) -> dict:
    d = np.load(npz_path)
    if "preds" not in d or "targets" not in d:
        return {}
    preds, targets = d["preds"], d["targets"]
    if preds.ndim != 3 or preds.shape[-1] != 3:
        return {"skipped": f"unexpected shape preds={preds.shape}"}
    per_q = {f"pinball_q{int(q * 100)}": _pinball(targets, preds[..., qi], q)
             for qi, q in enumerate(QUANTILES)}
    return {
        "pinball_per_q": per_q,
        "pinball_mean": float(np.mean(list(per_q.values()))),
        "raw_q10_q90_coverage": _empirical_coverage(
            targets, np.sort(preds, axis=-1)[..., 0], np.sort(preds, axis=-1)[..., -1]
        ),
        "reliability": _multi_alpha_from_q10_q90(preds, targets),
        "regime": _regime_split(preds, targets),
        "long_run_aci": _rolling_aci_audit(preds, targets),
    }


def _target_kind(name: str) -> str:
    return "activation" if ACT_PAT.search(name) else "price"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO / "models/forecaster")
    p.add_argument("--include-logo", action="store_true",
                   help="Also analyse canonical_*_lo_* runs")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for model_dir in sorted(args.root.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_NAMES:
            continue
        if (not args.include_logo) and LOGO_PAT.search(model_dir.name):
            continue
        per_seed: list[dict] = []
        for sd in sorted(model_dir.glob("seed-*/")):
            npz = sd / "val_preds.npz"
            if not npz.exists():
                continue
            r = analyse_one(npz)
            if r:
                r["seed"] = sd.name
                per_seed.append(r)
        if per_seed:
            results[model_dir.name] = {
                "target": _target_kind(model_dir.name),
                "n_seeds": len(per_seed),
                "per_seed": per_seed,
            }

    (OUT / "posthoc.json").write_text(json.dumps(results, indent=2))
    # Compact summary per (model, target): mean pinball + per-alpha reliability.
    compact = {}
    for name, rec in results.items():
        if not rec["per_seed"]:
            continue
        pinballs = [s["pinball_mean"] for s in rec["per_seed"]]
        cov_by_alpha = {
            k: float(np.mean([s["reliability"][k]["empirical"] for s in rec["per_seed"]]))
            for k in rec["per_seed"][0]["reliability"]
        }
        regimes = rec["per_seed"][0]["regime"]
        compact[name] = {
            "target": rec["target"],
            "n_seeds": rec["n_seeds"],
            "pinball_mean_dkk": float(np.mean(pinballs)),
            "pinball_std_dkk": float(np.std(pinballs)) if len(pinballs) > 1 else 0.0,
            "reliability_by_alpha": cov_by_alpha,
            "regime_n": {k: v.get("n", 0) for k, v in regimes.items()},
            "regime_pinball": {k: v.get("pinball_mean") for k, v in regimes.items()
                               if "pinball_mean" in v},
            "long_run_aci_final_alpha": rec["per_seed"][0]["long_run_aci"].get("alpha_t_final"),
            "long_run_aci_coverage": rec["per_seed"][0]["long_run_aci"].get("overall_empirical_coverage"),
        }
    (OUT / "posthoc_compact.json").write_text(json.dumps(compact, indent=2))
    print(f"wrote {OUT}/posthoc.json + posthoc_compact.json ({len(results)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
