"""Unit tests for retraining trigger rules and safe model lifecycle boundary (AC5, AC6)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.monitoring.drift_detector import (
    FeatureDriftReport,
    SingleFeatureDriftResult,
)
from app.application.monitoring.performance import (
    ModelPerformanceReport,
    SingleVersionPerformance,
)
from app.application.monitoring.trigger_manager import (
    RetrainingTriggerManager,
    TriggerDecision,
)


@pytest.fixture
def clean_drift_report():
    return FeatureDriftReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        baseline_version="v1",
        sample_count=50,
        is_sufficient_samples=True,
        drifted_features_count=0,
        has_significant_drift=False,
        feature_results={},
    )


@pytest.fixture
def clean_perf_report():
    return ModelPerformanceReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        total_evaluated_samples=50,
        version_metrics={},
        overall_mae_minutes=2.5,
        overall_rmse_minutes=3.8,
        has_degradation=False,
    )


@pytest.mark.asyncio
async def test_no_trigger_when_performance_and_drift_are_healthy(
    clean_perf_report, clean_drift_report
):
    """AC5: When all metrics are healthy, no retraining is triggered."""
    manager = RetrainingTriggerManager(cooldown_hours=4.0, enabled_auto_triggers=True)

    decision = await manager.evaluate_triggers(
        performance_report=clean_perf_report,
        drift_report=clean_drift_report,
    )

    assert decision.should_retrain is False
    assert decision.status == "no_trigger_needed"
    assert decision.trigger_type == "none"


@pytest.mark.asyncio
async def test_performance_degradation_triggers_retraining(clean_drift_report):
    """AC5: Performance degradation triggers retraining request."""
    manager = RetrainingTriggerManager(cooldown_hours=4.0, enabled_auto_triggers=True)

    degraded_perf = ModelPerformanceReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        total_evaluated_samples=40,
        version_metrics={
            "v1": SingleVersionPerformance(
                model_version="v1",
                sample_count=40,
                mae_minutes=6.2,
                rmse_minutes=9.1,
                is_sufficient_samples=True,
                is_degraded=True,
                degradation_reasons=["MAE (6.20m) exceeds threshold (5.00m)"],
            )
        },
        overall_mae_minutes=6.2,
        overall_rmse_minutes=9.1,
        has_degradation=True,
    )

    decision = await manager.evaluate_triggers(
        performance_report=degraded_perf,
        drift_report=clean_drift_report,
    )

    assert decision.should_retrain is True
    assert decision.status == "accepted"
    assert decision.trigger_type == "performance_degradation"
    assert "MAE" in decision.reason


@pytest.mark.asyncio
async def test_significant_feature_drift_triggers_retraining(clean_perf_report):
    """AC5: Significant feature drift triggers retraining request."""
    manager = RetrainingTriggerManager(cooldown_hours=4.0, enabled_auto_triggers=True)

    drifted_report = FeatureDriftReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        baseline_version="v1",
        sample_count=60,
        is_sufficient_samples=True,
        drifted_features_count=2,
        has_significant_drift=True,
        feature_results={
            "party_size": SingleFeatureDriftResult(
                feature_name="party_size",
                feature_type="numerical",
                psi_score=0.32,
                drift_level="significant_drift",
                is_drifted=True,
                sample_count=60,
            )
        },
    )

    decision = await manager.evaluate_triggers(
        performance_report=clean_perf_report,
        drift_report=drifted_report,
    )

    assert decision.should_retrain is True
    assert decision.status == "accepted"
    assert decision.trigger_type == "feature_drift"
    assert "party_size" in decision.reason


@pytest.mark.asyncio
async def test_manual_trigger_always_accepted():
    """AC5: Manual operator requests are accepted regardless of drift or metrics."""
    manager = RetrainingTriggerManager(enabled_auto_triggers=False)

    decision = await manager.evaluate_triggers(
        is_manual=True,
        operator_id="operator_jeswin",
        manual_notes="Ad-hoc retraining test",
    )

    assert decision.should_retrain is True
    assert decision.status == "accepted"
    assert decision.trigger_type == "manual"
    assert "operator_jeswin" in decision.reason


@pytest.mark.asyncio
async def test_auto_triggers_disabled_toggle(clean_drift_report):
    """AC5: Disabling automated triggers prevents auto-training while monitoring remains active."""
    manager = RetrainingTriggerManager(enabled_auto_triggers=False)

    degraded_perf = ModelPerformanceReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        total_evaluated_samples=40,
        version_metrics={},
        has_degradation=True,
    )

    decision = await manager.evaluate_triggers(
        performance_report=degraded_perf,
        drift_report=clean_drift_report,
    )

    assert decision.should_retrain is False
    assert decision.status == "rejected_disabled"


@pytest.mark.asyncio
async def test_cooldown_rejects_subsequent_triggers(clean_drift_report):
    """AC5: Cooldown period rejects repeated automatic retraining runs."""
    manager = RetrainingTriggerManager(cooldown_hours=4.0, enabled_auto_triggers=True)
    # Completed 1 hour ago (within 4-hour cooldown)
    manager._last_completed_retraining = datetime.now(timezone.utc) - timedelta(hours=1)

    degraded_perf = ModelPerformanceReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        total_evaluated_samples=40,
        version_metrics={},
        has_degradation=True,
    )

    decision = await manager.evaluate_triggers(
        performance_report=degraded_perf,
        drift_report=clean_drift_report,
    )

    assert decision.should_retrain is False
    assert decision.status == "rejected_cooldown"
    assert "cooldown period" in decision.reason


@pytest.mark.asyncio
async def test_concurrency_lock_prevents_overlapping_runs(clean_drift_report):
    """AC5: Active training execution locks out concurrent trigger requests."""
    manager = RetrainingTriggerManager(enabled_auto_triggers=True)

    degraded_perf = ModelPerformanceReport(
        evaluation_timestamp=datetime.now(timezone.utc),
        total_evaluated_samples=40,
        version_metrics={},
        has_degradation=True,
    )

    # Acquire lock simulating active training
    await manager._lock.acquire()
    try:
        decision = await manager.evaluate_triggers(
            performance_report=degraded_perf,
            drift_report=clean_drift_report,
        )
        assert decision.should_retrain is False
        assert decision.status == "rejected_already_running"
    finally:
        manager._lock.release()


@pytest.mark.asyncio
async def test_lifecycle_safety_retraining_never_mutates_active_model():
    """AC6: Retraining produces candidates only; it NEVER mutates or activates models."""
    manager = RetrainingTriggerManager()

    decision = TriggerDecision(
        trigger_id="trig_test_1",
        should_retrain=True,
        trigger_type="manual",
        status="accepted",
        reason="Test manual execution",
        evaluated_at=datetime.now(timezone.utc),
    )

    mock_snapshot = MagicMock()
    mock_snapshot.model_version = "production_v1_active"

    mock_report = {
        "candidate_model_path": "app/infrastructure/ml_models/artifacts/candidates/eta_model_candidate.json",
        "is_eligible": True,
    }

    with patch(
        "app.application.monitoring.trigger_manager.model_loader.get_snapshot",
        return_value=mock_snapshot,
    ):
        with patch("app.infrastructure.database.mongo_client.MongoManager.get_database", return_value=None):
            with patch("scripts.train_eta_model.run_training", new=AsyncMock(return_value=mock_report)):
                result = await manager.execute_retraining(decision, dataset_mode="synthetic")

                assert result.status == "completed"
                assert result.is_eligible_for_promotion is True
                # Active model remains production_v1_active
                assert mock_snapshot.model_version == "production_v1_active"
