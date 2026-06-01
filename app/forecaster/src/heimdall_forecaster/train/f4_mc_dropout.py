"""F4 — MC Dropout transformer. docs/RESEARCH-PROPOSAL.md §4.2.2 row F4.

Per Gal & Ghahramani (2016, ICML): a deterministic neural net with dropout
trained as usual *can* be reinterpreted as a Bayesian-approximate posterior
predictive when dropout is left active at inference and predictions are
averaged over K stochastic forward passes.

We do *not* re-train; we reuse the F7 patchTST checkpoints (which were
already trained with `dropout=0.1`) and Monte-Carlo-sample at inference.
This satisfies the proposal's UQ paradigm coverage at near-zero compute.

Methodology disclosure:
- The F7 dropout location (transformer encoder layer + FFN) is "concrete"
  not "variational" dropout. The Gal-Ghahramani guarantee strictly applies
  to dropout *before each weight matrix*; transformer-encoder dropout is a
  pragmatic approximation. We label F4 as MC-Dropout-approximate in §5
  rather than fully-Bayesian.
- Per-MC-sample we obtain a (q10, q50, q90) triple. The reported point
  prediction is the mean over K samples per quantile. The MC-spread of q50
  is preserved as `mc_q50_std.npy` for the verifier's epistemic-uncertainty
  channel (§4.7).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    QuantilePanelDataset,
    WindowStats,
    make_windows,
)
from heimdall_forecaster.train._utils import pinball_loss, resolve_device
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
from heimdall_ml import seeds, tracking


REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class F4Config:
    name: str = "f4_mc_dropout"
    member_seeds: tuple[int, ...] = (13, 42, 137, 1729, 31415)
    member_model: str = "f7"
    n_mc_samples: int = 30
    val_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_val.parquet")
    train_panel: Path = field(default_factory=lambda: REPO_ROOT / "data/processed/dk1_panel_train.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    batch_size: int = 256
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "models/forecaster")
    device: str = "auto"
    experiment: str = "heimdall-forecaster-f4"



def _load_member(model_dir: Path, n_features: int, device: torch.device) -> tuple[PatchTransformerQuantile, WindowStats, dict]:
    cfg_path = model_dir / "config.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    model = PatchTransformerQuantile(
        n_features=n_features,
        seq_len=cfg.get("seq_len", SEQ_LEN),
        horizon=cfg.get("horizon", HORIZON),
        n_quantiles=len(QUANTILES),
        patch_len=cfg.get("patch_len", 8),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 8),
        n_layers=cfg.get("n_layers", 6),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    with open(model_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return model, stats, cfg



def _mc_predict(
    model: PatchTransformerQuantile,
    dataloader: DataLoader,
    *,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    """Returns (S, N, H, Q) MC samples in normalised target space."""
    # Force dropout active: switch to train mode but disable BatchNorm update —
    # the transformer encoder uses LayerNorm only, so train() is safe here.
    model.train()
    samples = []
    with torch.no_grad():
        for s in range(n_samples):
            preds = []
            for x, _ in dataloader:
                x = x.to(device)
                p = model.predict_quantiles(x).cpu().numpy()
                preds.append(p)
            samples.append(np.concatenate(preds, axis=0))
    return np.stack(samples, axis=0)


def train_f4(cfg: F4Config) -> dict:
    """Per-member: load F7 weights at seed S, run MC dropout, save.

    *Critical*: each F7 seed was trained with its own (mean, std) normalisation
    — passed via ``stats.pkl``. We must build val windows using *that* seed's
    stats so the model sees inputs in the distribution it was trained on.
    """
    device = resolve_device(cfg.device)

    metrics_per_seed = []
    for seed in cfg.member_seeds:
        seeds.seed_everything(seed)
        member_dir = cfg.out_dir / cfg.member_model / f"seed-{seed}"
        if not (member_dir / "model.pt").exists():
            raise FileNotFoundError(f"missing F7 checkpoint at {member_dir}")
        n_features = 1  # F7 univariate
        model, stats, _ = _load_member(member_dir, n_features=n_features, device=device)
        # Re-normalise val data using *this* member's training stats.
        X_va, Y_va_norm, _ = make_windows(
            cfg.val_panel,
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            multivariate=False,
            stats=stats,
        )
        val_ds = QuantilePanelDataset(X_va, Y_va_norm)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        mc_norm = _mc_predict(model, val_loader, n_samples=cfg.n_mc_samples, device=device)
        # mc_norm: (S, N, H, Q) in normalised target units; sort over Q axis (predict_quantiles already sorts).
        mc_dn = stats.denormalise_target(mc_norm)
        # Aggregate.
        ens_preds = mc_dn.mean(axis=0).astype(np.float32)  # (N, H, Q)
        mc_q50_std = mc_dn[..., len(cfg.quantiles) // 2].std(axis=0).astype(np.float32)  # (N, H)
        targets = stats.denormalise_target(Y_va_norm).astype(np.float32)

        out = cfg.out_dir / cfg.name / f"seed-{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "val_preds.npz", preds=ens_preds, targets=targets)
        np.save(out / "mc_q50_std.npy", mc_q50_std)

        with tracking.run(
            name=f"{cfg.name}-seed{seed}",
            experiment=cfg.experiment,
            params={"seed": seed, "n_mc_samples": cfg.n_mc_samples, "horizon": cfg.horizon},
        ):
            per_q = {}
            for qi, q in enumerate(cfg.quantiles):
                per_q[f"val_pinball_q{int(q*100)}"] = pinball_loss(targets, ens_preds[..., qi], q)
            per_q["val_pinball_mean"] = float(np.mean(list(per_q.values())))
            sorted_p = np.sort(ens_preds, axis=-1)
            per_q["val_q10_q90_coverage"] = float(
                np.mean((targets >= sorted_p[..., 0]) & (targets <= sorted_p[..., -1]))
            )
            aci = aci_coverage_from_val(out / "val_preds.npz", alpha=0.1, gamma=0.05, horizon_step=0)
            per_q["aci_alpha_target"] = aci.alpha_target
            per_q["aci_empirical_coverage"] = aci.empirical_coverage
            per_q["aci_mean_width"] = aci.mean_width
            per_q["mc_q50_std_mean_dkk"] = float(mc_q50_std.mean())
            per_q["mc_q50_std_max_dkk"] = float(mc_q50_std.max())
            per_q["seed"] = seed
            per_q["description"] = f"f4_mc_dropout_K{cfg.n_mc_samples}_over_f7_seed{seed}"
            tracking.log_metrics({k: v for k, v in per_q.items() if isinstance(v, (int, float))})
            with open(out / "metrics.json", "w") as fh:
                json.dump(per_q, fh, indent=2)
        metrics_per_seed.append(per_q)
    return {"per_seed": metrics_per_seed}


def main() -> int:
    cfg = F4Config()
    res = train_f4(cfg)
    for m in res["per_seed"]:
        print(f"seed={m['seed']} mean_pinball={m['val_pinball_mean']:.1f}  ACI={m['aci_empirical_coverage']:.3f}  mc_q50_std={m['mc_q50_std_mean_dkk']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
