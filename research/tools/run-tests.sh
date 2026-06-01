#!/usr/bin/env bash
# Run the full Heimdall workspace test suite. Mirrors the pytest config in
# the root pyproject.toml; coverage report is printed for the verifier
# (the only safety-critical service on day 1).
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --no-progress --quiet
uv run pytest -q "$@"
uv run pytest apps/verifier/tests/ --cov=heimdall_verifier --cov-report=term-missing -q
