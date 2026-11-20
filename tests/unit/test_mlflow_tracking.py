"""Unit tests for MLflow experiment tracking integration."""

import mlflow
from mlflow.entities import RunStatus
import pytest

from app.infrastructure.ml_models.tracking import MLflowTracker


def test_mlflow_tracker_logs_run_details(tmp_path):
    tracking_uri = f"file://{tmp_path}/mlruns"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        run_name="test-run",
    )

    dummy_artifact = tmp_path / "dummy_artifact.txt"
    dummy_artifact.write_text("artifact content", encoding="utf-8")

    with tracker:
        run_id = tracker.run_id
        assert run_id is not None

        tracker.log_training_run(
            params={"param1": "val1", "n_estimators": 100},
            metrics={"test_mae": 2.5, "missing_metric": None},
            tags={"dataset_mode": "real", "candidate_eligible": "true"},
            artifact_paths=[str(dummy_artifact)],
        )

    # Verify run persisted in MLflow
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)
    assert run.data.params["param1"] == "val1"
    assert run.data.params["n_estimators"] == "100"
    assert run.data.metrics["test_mae"] == pytest.approx(2.5)
    assert "missing_metric" not in run.data.metrics
    assert run.data.tags["dataset_mode"] == "real"
    assert run.data.tags["candidate_eligible"] == "true"

    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]
    assert "dummy_artifact.txt" in artifact_names


def test_mlflow_tracker_records_failed_status_on_exception(tmp_path):
    tracking_uri = f"file://{tmp_path}/mlruns"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        run_name="failing-run",
    )

    run_id = None
    with pytest.raises(RuntimeError, match="Simulated training failure"):
        with tracker:
            run_id = tracker.run_id
            raise RuntimeError("Simulated training failure")

    assert run_id is not None
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    # Incomplete / failed runs must NEVER be marked as FINISHED
    assert run.info.status == RunStatus.to_string(RunStatus.FAILED)

