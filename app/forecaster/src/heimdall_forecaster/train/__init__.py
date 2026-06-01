"""F7 / F8 forecaster training. Per docs/RESEARCH-PROPOSAL.md §4.4.

F7 = patchTST-style transformer encoder, univariate price target.
F8 = same architecture, multivariate input (price + load + DA price).

Both produce quantile forecasts at q ∈ {0.1, 0.5, 0.9} per §4.4 protocol.
"""

from heimdall_forecaster.train.dataset import QuantilePanelDataset, make_windows
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_forecaster.train.trainer import TrainConfig, train_model

__all__ = [
    "PatchTransformerQuantile",
    "QuantilePanelDataset",
    "TrainConfig",
    "make_windows",
    "train_model",
]
