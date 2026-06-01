"""patchTST-style transformer encoder with quantile heads. docs/RESEARCH-PROPOSAL.md §4.4.

Architecture (~1.5M params; well under the 50M cap):
  - Patch embedding: chunk the input sequence into non-overlapping patches of
    length ``patch_len`` and project each to ``d_model`` (Nie et al. 2023).
  - 6-layer transformer encoder, 8 heads, ``d_model=128`` (small for B200 + CPU).
  - Quantile head: linear ``d_model → horizon * n_quantiles``; reshaped at output.

The quantile loss enforces non-crossing implicitly via separate heads per quantile;
we additionally clamp at inference for hard monotonicity (callers can opt-in).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Non-overlapping patch projection of a (B, T, F) tensor."""

    def __init__(self, n_features: int, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if t % self.patch_len != 0:
            raise ValueError(f"seq_len={t} must be divisible by patch_len={self.patch_len}")
        n_patches = t // self.patch_len
        # (B, n_patches, patch_len * F)
        patches = x.reshape(b, n_patches, self.patch_len * f)
        return self.proj(patches)


class PatchTransformerQuantile(nn.Module):
    """Patch-transformer with multi-quantile output. F7 (n_features=1) / F8 (n_features=3).

    `use_rin`: if True, applies Reversible Instance Normalization (Kim et al.,
    ICLR 2022) to each input window. RIN normalises each (batch, feature)
    series by its own mean/std, processes via the transformer, then de-
    normalises the output. Without RIN, high-dim multivariate patchTST
    variants on this task collapse to constant predictors (verified
    2026-05-17 on F8b/c/d/e). Default False to preserve F7/F8 baseline
    semantics; F8b/c/d/e/F13 should set True.
    """

    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int = 96,
        horizon: int = 16,
        n_quantiles: int = 3,
        patch_len: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        use_rin: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.patch_len = patch_len
        self.use_rin = use_rin
        # RIN learnable affine on the target feature (col 0 = imbalance price).
        if use_rin:
            self.rin_gamma = nn.Parameter(torch.ones(n_features))
            self.rin_beta = nn.Parameter(torch.zeros(n_features))

        self.patch = PatchEmbedding(n_features, patch_len, d_model)
        n_patches = seq_len // patch_len
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Pool over patches → linear head
        self.head = nn.Linear(d_model * n_patches, horizon * n_quantiles)

    def _rin_normalise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-window per-feature normalisation (Kim et al. ICLR 2022)."""
        mean = x.mean(dim=1, keepdim=True)              # (B, 1, F)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1, F)
        x_norm = (x - mean) / std * self.rin_gamma + self.rin_beta
        return x_norm, mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if self.use_rin:
            # FIX 2026-05-17: trainer already applies GLOBAL z-score
            # normalization, so the input is already mean~0 / std~1
            # globally. Applying per-window centering on top removes the
            # within-window mean shift (which is exactly the
            # "regime-dependent" component we want to expose to the
            # transformer). Critically: we DO NOT denormalize the model
            # output via per-window stats here — the trainer's existing
            # `stats.denormalise_target()` handles the final scale at
            # eval time. This avoids the double-normalization bug that
            # produced F13 q50 range [-7800, +39500] EUR.
            x_norm, mean, std = self._rin_normalise(x)
            x = x_norm
        h = self.patch(x) + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        b = h.shape[0]
        out = self.head(h.reshape(b, -1))
        out = out.reshape(b, self.horizon, self.n_quantiles)
        # No denormalization at model output: out stays in global z-score
        # space (matching the trainer's target normalization). The trainer
        # denormalizes at eval time via stats.denormalise_target().
        return out

    @torch.no_grad()
    def predict_quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """Sort over the quantile axis for monotonic output (Chernozhukov et al.)."""
        out = self.forward(x)
        return out.sort(dim=-1).values


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Pinball loss per (B, H, Q) -> scalar mean. ``target`` is (B, H)."""
    target = target.unsqueeze(-1)  # (B, H, 1)
    err = target - pred  # (B, H, Q)
    qs = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    loss = torch.maximum(qs * err, (qs - 1.0) * err)
    return loss.mean()


class ConvCNPForecaster(nn.Module):
    """Small ConvCNP-style sequence-to-Gaussian forecaster for F5."""

    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int = 96,
        horizon: int = 16,
        d_model: int = 96,
        n_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon
        self.input = nn.Linear(n_features + 1, d_model)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.query = nn.Parameter(torch.linspace(1.0 / horizon, 1.0, horizon).reshape(1, horizon, 1))
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        pos = torch.linspace(0.0, 1.0, t, device=x.device, dtype=x.dtype).reshape(1, t, 1).expand(b, -1, -1)
        h = self.input(torch.cat([x, pos], dim=-1)).transpose(1, 2)
        h = self.encoder(h).transpose(1, 2)
        context = h.mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
        q = self.query.to(device=x.device, dtype=x.dtype).expand(b, -1, -1)
        out = self.head(torch.cat([context, q], dim=-1))
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + 1e-4
        return mu, sigma


class AttentiveNPForecaster(nn.Module):
    """Attentive Neural Process sequence forecaster for F6."""

    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int = 96,
        horizon: int = 16,
        d_model: int = 96,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon
        self.context_proj = nn.Linear(n_features + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        pos = torch.linspace(0.0, 1.0, t, device=x.device, dtype=x.dtype).reshape(1, t, 1).expand(b, -1, -1)
        context = self.context_proj(torch.cat([x, pos], dim=-1))
        context = self.context_encoder(context)
        qpos = torch.linspace(
            1.0 / self.horizon,
            1.0,
            self.horizon,
            device=x.device,
            dtype=x.dtype,
        ).reshape(1, self.horizon, 1).expand(b, -1, -1)
        query = self.query_proj(qpos)
        attended, _ = self.attn(query, context, context, need_weights=False)
        out = self.head(attended)
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + 1e-4
        return mu, sigma


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target).mean()


__all__ = [
    "AttentiveNPForecaster",
    "ConvCNPForecaster",
    "PatchEmbedding",
    "PatchTransformerQuantile",
    "gaussian_nll",
    "quantile_loss",
]
