"""Unit tests for monitoring admin endpoints and auth checks (AC7, AC8)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from app.application.monitoring.correlation import CorrelationSummary
from app.application.monitoring.data_quality import (
    DataQualityRejectionBreakdown,
    DataQualityReport,
)
from app.application.monitoring.drift_detector import FeatureDriftReport
from app.application.monitoring.performance import ModelPerformanceReport
from app.application.monitoring.service import MonitoringCycleReport
from app.application.monitoring.trigger_manager import (
    RetrainingExecutionResult,
    TriggerDecision,
)
from app.infrastructure.config.settings import settings
from app.main import app

client = TestClient(app)
ADMIN_KEY = "k" * 32


@pytest.fixture(autouse=True)
def setup_admin_env(monkeypatch):
    """Set configured admin key for tests."""
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)


def test_monitoring_endpoints_require_admin_key():
    """Verify all monitoring admin endpoints reject requests without a valid X-Admin-Key."""
    # 1. Status endpoint
    resp = client.get("/api/v1/eta/admin/monitoring/status")
    assert resp.status_code == 401

    resp_bad = client.get(
        "/api/v1/eta/admin/monitoring/status",
        headers={"X-Admin-Key": "invalid_key"},
    )
    assert resp_bad.status_code == 401

    # 2. Evaluate endpoint
    resp_eval = client.post(
        "/api/v1/eta/admin/monitoring/evaluate",
        json={"lookback_hours": 24},
    )
    assert resp_eval.status_code == 401

    # 3. Retrain trigger endpoint
    resp_trigger = client.post(
        "/api/v1/eta/admin/monitoring/trigger-retrain",
        json={"reason": "test operator trigger"},
    )
    assert resp_trigger.status_code == 401


def test_get_monitoring_status_success():
    """Verify GET /api/v1/eta/admin/monitoring/status returns system state when authorized."""
    resp = client.get(
        "/api/v1/eta/admin/monitoring/status",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "active_model_version" in data
    assert "is_retraining_locked" in data


@pytest.mark.asyncio
async def test_evaluate_endpoint_invokes_cycle():
    """Verify POST /api/v1/eta/admin/monitoring/evaluate triggers the orchestration cycle."""
    dummy_report = MonitoringCycleReport(
        evaluation_id="eval-12345",
        evaluated_at=datetime.now(timezone.utc),
        lookback_hours=12,
        active_model_version="test_v1",
        correlation_summary=CorrelationSummary(
            total_predictions=10,
            total_completed_outcomes=8,
            correlated_count=5,
            uncorrelated_predictions_count=5,
            uncorrelated_outcomes_count=3,
            records=[],
        ),
        data_quality_report=DataQualityReport(
            total_evaluated=5,
            valid_count=5,
            excluded_count=0,
            rejections=DataQualityRejectionBreakdown(),
            is_sufficient_for_evaluation=False,
            sufficiency_message="Insufficient samples",
            valid_records=[],
        ),
        drift_report=FeatureDriftReport(
            evaluation_timestamp=datetime.now(timezone.utc),
            baseline_version="test_v1",
            sample_count=5,
            is_sufficient_samples=False,
            drifted_features_count=0,
            has_significant_drift=False,
            feature_results={},
        ),
        performance_report=ModelPerformanceReport(
            evaluation_timestamp=datetime.now(timezone.utc),
            total_evaluated_samples=5,
            version_metrics={},
            has_degradation=False,
        ),
        trigger_decision=TriggerDecision(
            trigger_id="trig-123",
            should_retrain=False,
            trigger_type="none",
            status="no_trigger_needed",
            reason="Sample count insufficient for trigger",
            evaluated_at=datetime.now(timezone.utc),
        ),
    )

    with patch(
        "app.presentation.admin.monitoring_router.run_monitoring_cycle",
        new=AsyncMock(return_value=dummy_report),
    ) as mock_cycle:
        resp = client.post(
            "/api/v1/eta/admin/monitoring/evaluate",
            headers={"X-Admin-Key": ADMIN_KEY},
            json={
                "lookback_hours": 12,
                "auto_trigger": False,
                "dataset_mode": "synthetic",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["evaluation_id"] == "eval-12345"
        assert data["lookback_hours"] == 12
        assert data["active_model_version"] == "test_v1"
        mock_cycle.assert_awaited_once_with(
            lookback_hours=12,
            min_sample_size=None,
            restaurant_id=None,
            auto_trigger=False,
            dataset_mode="synthetic",
        )


def test_evaluate_endpoint_validation_error():
    """Verify input validation rules on POST /api/v1/eta/admin/monitoring/evaluate."""
    resp = client.post(
        "/api/v1/eta/admin/monitoring/evaluate",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"lookback_hours": 0},  # Below ge=1
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_trigger_manual_retrain_endpoint():
    """Verify POST /api/v1/eta/admin/monitoring/trigger-retrain coordinates manual candidate retraining."""
    now_utc = datetime.now(timezone.utc)
    mock_decision = TriggerDecision(
        trigger_id="manual-trig-999",
        should_retrain=True,
        trigger_type="manual",
        status="accepted",
        reason="Manual operator retraining test",
        evaluated_at=now_utc,
    )
    mock_exec = RetrainingExecutionResult(
        trigger_id="manual-trig-999",
        status="completed",
        candidate_artifact_id="eta_candidate_manual_test.json",
        is_eligible_for_promotion=True,
        completed_at=now_utc,
    )

    with (
        patch(
            "app.presentation.admin.monitoring_router.RetrainingTriggerManager.request_manual_retraining",
            new=AsyncMock(return_value=mock_decision),
        ),
        patch(
            "app.presentation.admin.monitoring_router.RetrainingTriggerManager.execute_retraining",
            new=AsyncMock(return_value=mock_exec),
        ),
    ):
        resp = client.post(
            "/api/v1/eta/admin/monitoring/trigger-retrain",
            headers={"X-Admin-Key": ADMIN_KEY},
            json={
                "reason": "Manual operator retraining test",
                "approver_id": "op_test_123",
                "dataset_mode": "synthetic",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"]["should_retrain"] is True
        assert data["decision"]["trigger_type"] == "manual"
        assert data["execution"]["candidate_artifact_id"] == "eta_candidate_manual_test.json"
        assert data["execution"]["is_eligible_for_promotion"] is True
