"""F11 — PriceFM fine-tuned for DK1 + split-CP wrap.

Per docs/RESEARCH-PROPOSAL.md §4.2.2.  PriceFM (arXiv:2508.04875) is a
probabilistic foundation model pretrained on day-ahead prices across 24
European countries / 38 regions, with graph-based inductive biases for
transmission topology.

**Status as of 2026-05-10**: the PriceFM authors have not published
weights to a public HuggingFace repo we could resolve, and the paper's
artefact section points at private code.  This backend:
1. Looks for a fine-tuned local checkpoint at
   ``models/forecaster/f11/seed-<seed>/{config.json,model.pt,stats.pkl}``.
2. If absent, raises a clear FileNotFoundError pointing at the
   ``MODEL_CARD.md`` recipe for fine-tuning the F7 patch-TST on DK1
   data as a *PriceFM-shaped surrogate* (same input shape, same
   quantile head, same conformal wrap).

The surrogate loads through the same patch-TST loader as F7/F8, so the
backend is *zero-overhead* once a checkpoint dir lands; until then,
calling F11 fails fast with an actionable message.  A reviewer can
verify the wiring without us shipping unauthored weights.
"""

from __future__ import annotations

from ..registry import register
from .f7_patch_tst import _load_patch_tst, _PatchTSTBase


@register("f11", description="PriceFM fine-tuned (per arXiv:2508.04875) — patch-TST surrogate when authored weights unavailable")
def _load_f11(seed: int) -> _PatchTSTBase:
    """Load F11 — PriceFM-shaped patch-TST surrogate from local or HF artifacts.

    The directory layout follows the F7/F8 convention so the same
    ``_load_patch_tst`` reconstructor handles it without modification.
    """
    try:
        return _load_patch_tst("f11", seed)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "F11 (PriceFM-shaped surrogate) could not be loaded from local "
            "or HuggingFace artifacts. Expected config.json, model.pt, and "
            "stats.pkl under models/forecaster/f11/seed-<seed>/ locally, or "
            "f11/seed-<seed>/ in Phongsakon/heimdall. If those "
            "artifacts are absent, see the MODEL_CARD fine-tune recipe. "
            f"Underlying error: {e}"
        ) from e
