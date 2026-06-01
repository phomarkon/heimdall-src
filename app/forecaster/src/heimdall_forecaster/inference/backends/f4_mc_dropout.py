"""F4 — MC-Dropout-over-F7 backend (Gal & Ghahramani 2016, ICML).

F4 has no model.pt of its own — it loads the F7 checkpoint at the same seed
and runs K stochastic forward passes with dropout active. Returns quantiles
from the empirical sample distribution at each horizon step.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from heimdall_contracts import QuantileForecast

from ..hf_hydrator import checkpoint_dir
from ..registry import register


@dataclass
class _F4Backend:
    name: str
    seed: int
    model: object
    stats: object
    seq_len: int
    n_mc: int

    def predict(
        self,
        history: list[float] | np.ndarray | Iterable[float],
        horizon: int = 16,
        levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[QuantileForecast]:
        x = np.asarray(list(history), dtype=np.float64)
        if x.size < self.seq_len:
            x = np.concatenate([np.full(self.seq_len - x.size, x[0] if x.size else 0.0), x])
        x = x[-self.seq_len:]
        z = (x - self.stats.target_mean) / self.stats.target_std
        device = next(self.model.parameters()).device
        x_t = torch.from_numpy(z).float().reshape(1, self.seq_len, 1).to(device)
        # Force dropout active.
        self.model.train()
        samples = []
        with torch.no_grad():
            # Single batched pass: replicate input n_mc times so dropout draws
            # are independent per row.  Avoids n_mc serial transformer calls.
            x_rep = x_t.expand(self.n_mc, -1, -1)
            preds = self.model.predict_quantiles(x_rep).cpu().numpy()  # (n_mc, H, 3)
            samples = [preds[i] for i in range(self.n_mc)]
        self.model.eval()
        # samples shape: (n_mc, H, 3) — q10/q50/q90 per pass.
        S = np.stack(samples, axis=0)
        # Median across MC samples for each underlying quantile head; that
        # collapses K dropout draws into a single set of three quantiles.
        # Then take the empirical quantile across (n_mc, head) jointly per
        # horizon to widen the band with epistemic uncertainty.
        flat = S.reshape(S.shape[0] * S.shape[2], S.shape[1])  # (n_mc*3, H)
        q = np.quantile(flat, list(levels), axis=0).T  # (H, len(levels))
        q_dn = self.stats.denormalise_target(q)
        out: list[QuantileForecast] = []
        for h in range(min(horizon, q_dn.shape[0])):
            out.append(QuantileForecast(
                horizon_minutes=15 * (h + 1),
                levels=tuple(levels),
                values=tuple(float(v) for v in q_dn[h]),
            ))
        return out


@register("f4_mc_dropout", description="MC-Dropout over F7 weights (Gal & Ghahramani 2016)")
def _load_f4(seed: int) -> _F4Backend:
    from heimdall_forecaster.train.model import PatchTransformerQuantile

    d = checkpoint_dir("f7", seed)
    cfg = json.loads((d / "config.json").read_text())
    model = PatchTransformerQuantile(
        n_features=int(cfg["n_features"]),
        seq_len=int(cfg["seq_len"]), horizon=int(cfg["horizon"]),
        n_quantiles=3, patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(d / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return _F4Backend(name="f4_mc_dropout", seed=seed, model=model, stats=stats,
                     seq_len=int(cfg["seq_len"]), n_mc=30)
