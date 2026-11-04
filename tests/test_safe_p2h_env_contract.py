"""Static and numerical contracts for the safe-exploration P2H environment."""

from pathlib import Path

import numpy as np

from research.ml.experiments.baselines_focal_agent import (
    DISPATCH_LEVELS,
    P2HSpec,
    dispatch_grid,
    feasible_action_mask,
)


def test_dispatch_is_discrete_and_symmetric():
    levels = dispatch_grid(P2HSpec())
    assert len(levels) == DISPATCH_LEVELS == 21
    assert levels[0] == -levels[-1]
    assert levels[DISPATCH_LEVELS // 2] == 0.0


def test_pre_action_mask_removes_nominal_soc_violations():
    spec = P2HSpec()
    for soc in (0.0, spec.initial_soc_mwh, spec.storage_mwh):
        levels = dispatch_grid(spec)
        mask = feasible_action_mask(soc, spec)
        nominal_next = (
            soc * (1.0 - spec.thermal_loss_mean)
            + spec.cop * levels * 0.25
            - spec.baseline_demand_mw * 0.25
        )
        assert np.any(mask)
        assert np.all((nominal_next[mask] >= 0) & (nominal_next[mask] <= spec.storage_mwh))


def test_observation_does_not_include_current_imbalance():
    source = Path(__file__).parents[1] / "research/ml/experiments/baselines_focal_agent.py"
    text = source.read_text()
    obs_start = text.index("def _obs(self):")
    obs_end = text.index("def reset(self", obs_start)
    observation_code = text[obs_start:obs_end]
    assert "imb[tt]" not in observation_code
    assert "imb[(tt - 1) % n]" in observation_code
