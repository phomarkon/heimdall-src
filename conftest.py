"""Repo-root pytest configuration.

Intentionally minimal: it only registers Hypothesis settings profiles so the
property-based suites (``test_property_*.py``) run deterministically and do not
fail on CI timing jitter. It adds NO fixtures and NO autouse hooks, so existing
example-based tests are unaffected.

Profiles:
  - ``ci``  : deadline disabled (shared CI runners are bursty), derandomised so
              a failure reproduces from the printed seed.
  - ``dev`` : more examples, deadline kept, for local invariant hunting.

Select with ``--hypothesis-profile=dev`` (default is ``ci``).
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    deadline=None,
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    deadline=400,
    max_examples=500,
)
settings.load_profile("ci")


@pytest.fixture
def tiny_dk_scenario():
    from packages.pypsa_adapter import build_tiny_dk_network, extract_heimdall_scenario, solve_network

    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")
    return extract_heimdall_scenario(network)
