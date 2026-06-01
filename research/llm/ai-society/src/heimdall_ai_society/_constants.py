from __future__ import annotations

# _constants.py — Named constants for threshold values extracted from runner.py

# Watch score thresholds
WATCH_SCORE_ACTIVE: float = 0.5
WATCH_SCORE_LOW: float = 0.35

# Price edge thresholds (EUR/MWh)
EDGE_STRONG: float = 25.0
EDGE_MODERATE: float = 15.0

# Interval width thresholds (EUR/MWh)
WIDTH_HIGH: float = 80.0
WIDTH_MODERATE: float = 50.0
WIDTH_LOW: float = 35.0

# Forecast disagreement edge threshold
DISAGREEMENT_EDGE: float = 35.0

# Forecast side signal thresholds
SIDE_SIGNAL_MIN_EDGE: float = 5.0
SIDE_SIGNAL_ADVANTAGE: float = 2.0

# Opportunity hint minimum edge
OPPORTUNITY_HINT_MIN_EDGE: float = 5.0

# Default candidate sizing
DEFAULT_CAP_FRACTION: float = 1.0
DEFAULT_MIN_MWH: float = 0.25
DEFAULT_MAX_CANDIDATES: int = 8

# P2H V2 price regime gate (EUR/MWh)
P2H_UP_ACTIVATION_GATE: float = 70.0

# EV V2 thresholds
EV_CONFIDENCE_GATE: float = 0.9
EV_WORST_CASE_GATE: float = 50.0
EV_LOW_PRICE_GATE: float = 55.0
EV_MIN_WORST_CASE: float = 2.0
EV_MIN_EXPECTED: float = 10.0
EV_MIN_CONFIDENCE: float = 0.7

# Risk filter thresholds
RISK_FILTER_MIN_CONFIDENCE: float = 0.62
RISK_FILTER_MIN_EXPECTED_PROFIT: float = 250.0

# Quota thresholds
QUOTA_BEHIND_MARGIN: float = 0.75
QUOTA_AHEAD_MARGIN: float = 1.25
