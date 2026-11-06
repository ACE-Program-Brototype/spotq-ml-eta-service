"""Unit tests for the monitoring cycle orchestrator (AC7, AC8)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.application.monitoring.correlation import CorrelationSummary
from app.application.monitoring.service import run_monitoring_cycle


@pytest.mark.asyncio
async def test_run_monitoring_cycle_end_to_end_orchestration():
    """Verify run_monitoring_cycle coordinates all monitoring components cleanly."""
    empty_summary = CorrelationSummary(
        total_predictions=0,
        total_completed_outcomes=0,
        correlated_count=0,
        uncorrelated_predictions_count=0,
        uncorrelated_outcomes_count=0,
        records=[],
    )

    with (
        patch(
            "app.application.monitoring.service.correlate_predictions_with_outcomes",
            new=AsyncMock(return_value=empty_summary),
        ),
        patch(
            "app.application.monitoring.service.MongoManager.get_database",
            return_value=None,
        ),
    ):
        report = await run_monitoring_cycle(
            lookback_hours=24,
            auto_trigger=True,
            dataset_mode="synthetic",
        )

        assert report.evaluation_id is not None
        assert report.lookback_hours == 24
        assert report.drift_report is not None
        assert report.drift_report.is_sufficient_samples is False
        assert report.performance_report is not None
        assert report.trigger_decision is not None
        assert report.trigger_decision.should_retrain is False
