"""Figure: empirical coverage vs alpha for split-CP and ACI on F7's val set.

Per docs/RESEARCH-PROPOSAL.md §4.6 (Theorem 1a/1b). Sweeps alpha ∈ {0.05, 0.1, 0.2, 0.3}
and plots empirical coverage 1 - alpha_emp. The diagonal y = 1 - alpha is the
target. A model that "delivers" Theorem 1a should sit on or just above the
diagonal (split-CP, finite-sample lower bound); ACI should converge to the
diagonal in the long run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.split_cp import SplitConformal
from heimdall_ml.viz import apply_paper_style, paper_palette
from heimdall_ml.viz.style import COLUMN_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[3]
F7_VAL = REPO_ROOT / "models/forecaster/f7/seed-42/val_preds.npz"
FIG_DIR = REPO_ROOT / "figures"

ALPHAS = np.array([0.05, 0.10, 0.15, 0.20, 0.30])


def _scores_from(val_preds_path: Path) -> np.ndarray:
    z = np.load(val_preds_path)
    preds, targets = z["preds"], z["targets"]
    q50 = preds[:, 0, preds.shape[-1] // 2]
    return np.abs(targets[:, 0] - q50)


def _split_cp_coverage(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    half = n // 2
    cal = scores[:half]
    test = scores[half:]
    cp = SplitConformal.fit(cal, alpha=alpha)
    return float(np.mean(test <= cp.quantile))


def _aci_coverage(scores: np.ndarray, alpha: float, gamma: float = 0.05) -> float:
    aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)
    warm = min(200, scores.size // 4)
    aci.warm_start(scores[:warm])
    cov = 0
    n = 0
    for s in scores[warm:]:
        if aci.predict_in_band(float(s)):
            cov += 1
        aci.update(float(s))
        n += 1
    return cov / max(n, 1)


def main() -> int:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scores = _scores_from(F7_VAL)

    sp_cov = np.array([_split_cp_coverage(scores, a) for a in ALPHAS])
    aci_cov = np.array([_aci_coverage(scores, a) for a in ALPHAS])

    palette = paper_palette(2)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ax.plot(1 - ALPHAS, 1 - ALPHAS, "k--", lw=0.8, label="target (1 − α)")
    ax.plot(1 - ALPHAS, sp_cov, "o-", color=palette[0], label="split-CP (Theorem 1a)")
    ax.plot(1 - ALPHAS, aci_cov, "s-", color=palette[1], label="ACI (Theorem 1b)")
    ax.set_xlabel("Target coverage 1 − α")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="lower right")
    ax.set_title("F7 conformal coverage vs target (val set, seed 42)")
    fig.savefig(FIG_DIR / "conformal_coverage_curve.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'conformal_coverage_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
