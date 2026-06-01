"""Toy SHAP smoke test (sklearn LinearRegression).

This used to also carry a stubbed ``explain_forecast``; the real
implementation now lives in ``heimdall_ml.explain.explain_forecast`` and the
stub has been removed (docs/RESEARCH-PROPOSAL.md §1.3 / Day-2 deliverable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShapAttribution:
    """SHAP values for a single prediction.

    ``feature_names`` aligns with the columns of ``values``. ``base_value`` is
    the SHAP base (expected model output); the sum of ``values`` +
    ``base_value`` equals the model's prediction.
    """

    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    base_value: float


def toy_shap_example(seed: int = 13) -> ShapAttribution:
    """Minimal end-to-end SHAP smoke test on a synthetic regression problem.

    Used in tests to assert that SHAP is installed and the contract holds.
    Not part of any production code path.
    """
    import shap
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((128, 4))
    coefs = np.array([1.5, -2.0, 0.5, 0.0])
    y = X @ coefs + 0.1 * rng.standard_normal(128)

    model = LinearRegression().fit(X, y)
    explainer = shap.LinearExplainer(model, X)
    sv = explainer.shap_values(X[:1])
    return ShapAttribution(
        feature_names=("x0", "x1", "x2", "x3"),
        values=np.asarray(sv, dtype=np.float64).reshape(-1),
        base_value=float(explainer.expected_value),
    )
