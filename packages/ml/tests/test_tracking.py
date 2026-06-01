"""MLflow tracking helper smoke test. Uses a tmp_path tracking URI to avoid
polluting the repo `mlruns/` from CI."""

from __future__ import annotations

from pathlib import Path

import mlflow

from heimdall_ml import tracking


def test_tracking_helper_logs_a_toy_run(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    tracking.init(tracking_uri=uri, experiment="heimdall-test")
    with tracking.run("toy", params={"seed": 13, "alpha": 0.1}, tags={"kind": "smoke"}):
        tracking.log_metrics({"empirical_coverage": 0.91}, step=0)
    runs = mlflow.search_runs(experiment_names=["heimdall-test"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.seed"] == "13"
    assert runs.iloc[0]["tags.heimdall.seed"] == "13"
    assert runs.iloc[0]["tags.kind"] == "smoke"


def test_compute_tracker_logs_cpu_fallback(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    tracking.init(tracking_uri=uri, experiment="heimdall-compute-test")
    with tracking.run("toy-compute", params={"seed": 42}):
        with tracking.track_experiment_compute("toy", gpu_indices=None):
            sum(range(1000))
    runs = mlflow.search_runs(experiment_names=["heimdall-compute-test"])
    assert len(runs) == 1
    assert "metrics.heimdall.toy.wall_seconds" in runs.columns
    assert "metrics.heimdall.toy.cpu_seconds" in runs.columns
    assert runs.iloc[0]["metrics.heimdall.toy.gpu_sampled_count"] == 0.0
