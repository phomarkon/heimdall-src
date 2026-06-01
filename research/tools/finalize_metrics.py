"""Post-train finalizer: ensure every model/seed dir has metrics.json + aci_state.json.

For models trained via run.py (e.g. F11) the trainer writes val_preds.npz but
no metrics.json. This script computes per-quantile pinball + ACI wrap from
val_preds.npz and writes the canonical metrics.json + aci_state.json the
leaderboard expects.

Idempotent: skips dirs that already have a fresh metrics.json newer than the
val_preds.npz it would summarise.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / "models/forecaster"

# Match the rest of the zoo.
QUANTILES = (0.1, 0.5, 0.9)


def _pinball(y, q, level):
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _aci_wrap(preds, targets, alpha=0.1, gamma=0.05):
    """Run the real ACI wrap (the same one seed_sweep uses).

    Delegates to `heimdall_forecaster.train.wrap_aci.aci_coverage_from_val`,
    which fits split-CP cal on a prefix and runs online α_t updates on the
    tail. Returns (alpha_target, empirical_coverage, mean_width).
    """
    import tempfile, os
    from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name, preds=preds, targets=targets)
        tmp_path = tmp.name
    try:
        aci = aci_coverage_from_val(tmp_path, alpha=alpha, gamma=gamma)
    finally:
        os.unlink(tmp_path)
    return aci.alpha_target, aci.empirical_coverage, aci.mean_width


def finalize(seed_dir: Path, model_name: str) -> bool:
    npz = seed_dir / "val_preds.npz"
    if not npz.exists():
        return False
    metrics_path = seed_dir / "metrics.json"
    # Always refresh — earlier versions of this script wrote raw band cov as
    # `aci_empirical_coverage` which is wrong. We now delegate to the real
    # ACI wrap. Re-running is idempotent + cheap.
    data = np.load(npz)
    preds = data["preds"].astype(np.float64)
    targets = data["targets"].astype(np.float64)
    seed = int(seed_dir.name.removeprefix("seed-"))
    per_q = {}
    for qi, q in enumerate(QUANTILES):
        per_q[f"val_pinball_q{int(q*100)}"] = _pinball(targets, preds[..., qi], q)
    pinball_mean = float(np.mean(list(per_q.values())))
    # RAW band coverage: fraction of (N,H) cells where target falls in [q10,q90].
    sorted_p = np.sort(preds, axis=-1)
    raw_cov = float(np.mean((targets >= sorted_p[..., 0]) & (targets <= sorted_p[..., -1])))
    # ACI-wrapped coverage (post-hoc online α_t adjustment).
    alpha, aci_cov, aci_width = _aci_wrap(preds, targets)
    metrics = {
        "seed": seed,
        **per_q,
        "val_pinball_mean": pinball_mean,
        "val_q10_q90_coverage": raw_cov,
        "aci_alpha_target": alpha,
        "aci_empirical_coverage": aci_cov,
        "aci_mean_width": aci_width,
        "model_name": model_name,
        "_source": "finalize_metrics.py (post-train backfill)",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    aci_state = seed_dir / "aci_state.json"
    if not aci_state.exists():
        aci_state.write_text(json.dumps({
            "alpha": alpha, "gamma": 0.05,
            "warm_start_n": int(targets.shape[0]),
            "empirical_coverage": aci_cov,
        }, indent=2))
    return True


def main() -> int:
    n_done = 0
    for model_dir in sorted(MODEL_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"release"}:
            continue
        for seed_dir in sorted(model_dir.glob("seed-*")):
            if finalize(seed_dir, model_dir.name):
                print(f"[finalize] wrote {seed_dir}/metrics.json")
                n_done += 1
    print(f"[finalize] done; {n_done} dirs finalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
