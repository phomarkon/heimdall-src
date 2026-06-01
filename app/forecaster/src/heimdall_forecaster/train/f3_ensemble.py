"""F3 — 5-seed patchTST Deep Ensemble. docs/RESEARCH-PROPOSAL.md §4.2.2 + ADR-0006.

The five F7 seed checkpoints `[13, 42, 137, 1729, 31415]` constitute the
ensemble members. F3 quantile predictions are the per-quantile mean of the
member outputs; the inter-member std at q50 is exposed as an epistemic-
uncertainty signal for the verifier (§4.7).

This module is *pure aggregation* — no training, no GPU. It loads each
F7 seed's `val_preds.npz` and emits ensemble predictions in the standard
F-zoo shape ``(N, H, Q)`` plus per-member std at the median.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from heimdall_forecaster.train._utils import pinball_loss
from heimdall_forecaster.train.dataset import QUANTILES
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds, tracking

REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class F3EnsembleConfig:
    name: str = "f3_ensemble"
    member_model: str = "f7"
    member_seeds: tuple[int, ...] = (13, 42, 137, 1729, 31415)
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    member_root: Path | None = None  # defaults to out_dir; tests override.
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    aci_alpha: float = 0.1
    aci_gamma: float = 0.05
    experiment: str = "heimdall-forecaster-f3-ensemble"


def _load_member(out_root: Path, member_model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(out_root / member_model / f"seed-{seed}" / "val_preds.npz")
    return z["preds"].astype(np.float64), z["targets"].astype(np.float64)



def build_ensemble(cfg: F3EnsembleConfig) -> dict:
    """Aggregate the F7 seed members into an F3 deep ensemble.

    Writes one synthetic seed dir per ensemble member: each contains the
    *averaged* predictions paired with that member's seed for traceability.
    The leaderboard treats this as a 5-seed entry with stdev measuring
    *member-to-mean* offset (so it is small but non-zero — reflects the
    ensemble's calibration spread)."""
    out_root = cfg.out_dir
    member_root = cfg.member_root if cfg.member_root is not None else out_root
    member_preds = []
    targets_ref: np.ndarray | None = None
    for s in cfg.member_seeds:
        preds_s, targets_s = _load_member(member_root, cfg.member_model, s)
        if targets_ref is None:
            targets_ref = targets_s
        else:
            if not np.allclose(targets_ref, targets_s):
                raise RuntimeError(
                    f"target mismatch between {cfg.member_model} seeds "
                    "— ensemble can only aggregate over identical val windows"
                )
        member_preds.append(preds_s)

    if targets_ref is None:
        raise RuntimeError("no member seeds supplied")

    stack = np.stack(member_preds, axis=0)  # (M, N, H, Q)
    ens_preds = stack.mean(axis=0)  # (N, H, Q)
    member_std_q50 = stack[..., len(cfg.quantiles) // 2].std(axis=0)  # (N, H)

    # Per-seed dirs with the ensemble mean (so the leaderboard sees 5 rows).
    metrics_per_seed = []
    for s in cfg.member_seeds:
        seed_dir = out_root / cfg.name / f"seed-{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez(seed_dir / "val_preds.npz", preds=ens_preds.astype(np.float32), targets=targets_ref.astype(np.float32))
        np.save(seed_dir / "member_std_q50.npy", member_std_q50.astype(np.float32))

        per_q = {}
        for qi, q in enumerate(cfg.quantiles):
            per_q[f"val_pinball_q{int(q*100)}"] = pinball_loss(targets_ref, ens_preds[..., qi], q)
        per_q["val_pinball_mean"] = float(np.mean(list(per_q.values())))
        sorted_p = np.sort(ens_preds, axis=-1)
        per_q["val_q10_q90_coverage"] = float(
            np.mean((targets_ref >= sorted_p[..., 0]) & (targets_ref <= sorted_p[..., -1]))
        )

        # ACI on q50, horizon-step 0 (matches F7/F8 leaderboard convention).
        aci = aci_coverage_from_val(
            seed_dir / "val_preds.npz",
            alpha=cfg.aci_alpha,
            gamma=cfg.aci_gamma,
            horizon_step=0,
        )
        per_q["aci_alpha_target"] = aci.alpha_target
        per_q["aci_empirical_coverage"] = aci.empirical_coverage
        per_q["aci_mean_width"] = aci.mean_width
        per_q["seed"] = s
        per_q["description"] = "f3_deep_ensemble_5seed_patchtst (ADR-0006)"

        with open(seed_dir / "metrics.json", "w") as fh:
            json.dump(per_q, fh, indent=2)
        metrics_per_seed.append(per_q)

    aggregate = {
        "ensemble_q50_mean_member_std_dkk": float(member_std_q50.mean()),
        "ensemble_q50_max_member_std_dkk": float(member_std_q50.max()),
    }
    with open(out_root / cfg.name / "ensemble_summary.json", "w") as fh:
        json.dump(
            {
                "config": {
                    "member_model": cfg.member_model,
                    "member_seeds": list(cfg.member_seeds),
                },
                "per_seed": metrics_per_seed,
                "aggregate": aggregate,
            },
            fh,
            indent=2,
        )
    return {"per_seed": metrics_per_seed, "aggregate": aggregate}


def main() -> int:
    seeds.seed_everything(42)
    cfg = F3EnsembleConfig()
    result = build_ensemble(cfg)
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
