"""Heimdall shared ML library.

Contains the conformal calibrators (split-CP, online ACI), reproducibility
infrastructure (seeds, MLflow tracking), evaluation metrics, and XAI/SHAP
hooks. This package contains *no* model weights and *no* training loops — it
is the calibration & evaluation kernel that the apps depend on.

Module layout (per docs/RESEARCH-PROPOSAL.md §6.6 / §10):

- `conformal.split_cp` — Theorem 1a (finite-sample, exchangeability)
- `conformal.aci`      — Theorem 1b (long-run, regime-shift robust)
- `seeds`              — frozen seed list [13, 42, 137, 1729, 31415]
- `tracking`           — MLflow helper used by EVERY experiment
- `eval`               — coverage, Sharpe, KS, etc.
- `explain`            — SHAP wrapper + bid attribution
"""

from heimdall_ml.seeds import FROZEN_SEEDS, seed_everything

__all__ = ["FROZEN_SEEDS", "seed_everything"]
