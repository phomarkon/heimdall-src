"""Evaluation metrics. Per docs/RESEARCH-PROPOSAL.md §5.3."""

from heimdall_ml.eval.coverage import (
    conditional_coverage,
    interval_width,
    marginal_coverage,
    pinball_loss,
)
from heimdall_ml.eval.leakage import (
    TEST_END_UTC,
    TEST_START_UTC,
    TRAIN_END_UTC,
    VAL_END_UTC,
    VAL_START_UTC,
    assert_no_test_overlap,
    assert_test_panel_only,
)

__all__ = [
    "TEST_END_UTC",
    "TEST_START_UTC",
    "TRAIN_END_UTC",
    "VAL_END_UTC",
    "VAL_START_UTC",
    "assert_no_test_overlap",
    "assert_test_panel_only",
    "conditional_coverage",
    "interval_width",
    "marginal_coverage",
    "pinball_loss",
]
