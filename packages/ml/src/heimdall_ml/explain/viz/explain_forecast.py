"""Paper-grade XAI figure: SHAP heatmap + quantile side panel.

Used by the §5 explainability figure of docs/RESEARCH-PROPOSAL.md. We use
``matplotlib`` only — no seaborn — for license-clean reuse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from heimdall_ml.explain.explain_forecast import ForecastExplanation


def render_forecast_explanation(
    explanation: ForecastExplanation,
    *,
    quantile_predictions: NDArray[np.float64] | None = None,
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    out_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """Render the SHAP heatmap with a quantile-prediction side panel.

    Parameters
    ----------
    explanation:
        Output of ``heimdall_ml.explain.explain``.
    quantile_predictions:
        ``(horizon, n_quantiles)`` tensor of model quantile output to plot in
        the side panel. Optional — if ``None``, side panel is skipped.
    out_path:
        Path to save the figure. If ``None``, defaults to
        ``figures/explain_forecast.png`` at repo root.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = Path(__file__).resolve().parents[6] / "figures" / "explain_forecast.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_side = quantile_predictions is not None
    fig = plt.figure(figsize=(10, 4))
    if has_side:
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax2 = None

    vmax = float(np.abs(explanation.values).max() or 1.0)
    im = ax.imshow(
        explanation.values.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
    )
    ax.set_xlabel("Input timestep (oldest -> newest)")
    ax.set_ylabel("Feature")
    ax.set_yticks(np.arange(len(explanation.feature_names)))
    ax.set_yticklabels(explanation.feature_names)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("SHAP value (DKK/MWh)")

    if title is None:
        title = (
            f"Forecast attribution — q{int(quantile_levels[explanation.quantile_idx]*100)} "
            f"@ horizon step {explanation.target_horizon_step}"
        )
    ax.set_title(title)

    if ax2 is not None and quantile_predictions is not None:
        h = np.arange(quantile_predictions.shape[0]) + 1
        ax2.plot(h, quantile_predictions[:, 1], "k-", label=f"q{int(quantile_levels[1]*100)}")
        ax2.fill_between(
            h,
            quantile_predictions[:, 0],
            quantile_predictions[:, -1],
            alpha=0.25,
            label=f"q{int(quantile_levels[0]*100)}-q{int(quantile_levels[-1]*100)}",
        )
        ax2.set_xlabel("Horizon step (15-min)")
        ax2.set_ylabel("Predicted price (DKK/MWh)")
        ax2.set_title("Quantile forecast")
        ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = ["render_forecast_explanation"]
