"""Unit tests for prediction logging and prediction-to-outcome correlation (AC1)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.monitoring.correlation import (
    correlate_predictions_with_outcomes,
    correlate_records_in_memory,
)
from app.infrastructure.monitoring.prediction_logger import (
    log_prediction,
    log_prediction_background,
)


@pytest.fixture
def sample_prediction():
    return {
        "prediction_id": "pred_001",
        "queue_entry_id": "entry_100",
        "restaurant_id": "restaurant_abc",
        "model_version": "eta_model_candidate",
        "predicted_wait_minutes": 15.0,
        "features": {
            "party_size": 2.0,
            "total_people_ahead_at_join": 3.0,
            "hour_of_day": 12.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "historical_average_wait": 14.5,
        },
        "timestamp": datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
        "fallback_used": False,
    }


@pytest.fixture
def sample_event():
    return {
        "_id": "obj_001",
        "queue_entry_id": "entry_100",
        "restaurant_id": "restaurant_abc",
        "party_size": 2,
        "joined_at": datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
        "completed_at": datetime(2026, 11, 6, 12, 16, tzinfo=timezone.utc),
        "actual_wait_time_minutes": 16.0,
        "is_synthetic": False,
    }


def test_exact_correlation_by_queue_entry_id(sample_prediction, sample_event):
    """AC1: Pair prediction with outcome using exact queue_entry_id match."""
    summary = correlate_records_in_memory([sample_prediction], [sample_event])

    assert summary.total_predictions == 1
    assert summary.total_completed_outcomes == 1
    assert summary.correlated_count == 1
    assert summary.uncorrelated_predictions_count == 0
    assert summary.uncorrelated_outcomes_count == 0

    record = summary.records[0]
    assert record.prediction_id == "pred_001"
    assert record.queue_entry_id == "entry_100"
    assert record.model_version == "eta_model_candidate"
    assert record.predicted_wait_minutes == 15.0
    assert record.actual_wait_minutes == 16.0
    assert record.correlation_method == "exact_queue_entry_id"
    assert record.features["party_size"] == 2.0


def test_heuristic_correlation_when_queue_entry_id_missing(
    sample_prediction, sample_event
):
    """AC1: Fallback heuristic links prediction by restaurant, party size, and timestamp."""
    # Prediction has no queue_entry_id (e.g. browsing prediction before joining queue)
    sample_prediction["queue_entry_id"] = None

    summary = correlate_records_in_memory([sample_prediction], [sample_event])

    assert summary.correlated_count == 1
    record = summary.records[0]
    assert record.prediction_id == "pred_001"
    assert record.queue_entry_id == "entry_100"
    assert record.correlation_method == "heuristic_proximity"
    assert record.predicted_wait_minutes == 15.0
    assert record.actual_wait_minutes == 16.0


def test_heuristic_correlation_rejects_timestamp_outside_tolerance(
    sample_prediction, sample_event
):
    """Predictions more than 10 minutes away from join time are not heuristically linked."""
    sample_prediction["queue_entry_id"] = None
    # 25 minutes prior to join time
    sample_prediction["timestamp"] = sample_event["joined_at"] - timedelta(minutes=25)

    summary = correlate_records_in_memory([sample_prediction], [sample_event])

    assert summary.correlated_count == 0
    assert summary.uncorrelated_predictions_count == 1
    assert summary.uncorrelated_outcomes_count == 1


def test_multiple_predictions_for_same_queue_entry_deduplicated(
    sample_prediction, sample_event
):
    """When multiple predictions exist for one queue entry, select the closest to joined_at."""
    earlier_pred = sample_prediction.copy()
    earlier_pred["prediction_id"] = "pred_earlier"
    earlier_pred["timestamp"] = sample_event["joined_at"] - timedelta(minutes=8)
    earlier_pred["predicted_wait_minutes"] = 18.0

    closer_pred = sample_prediction.copy()
    closer_pred["prediction_id"] = "pred_closer"
    closer_pred["timestamp"] = sample_event["joined_at"] - timedelta(minutes=1)
    closer_pred["predicted_wait_minutes"] = 15.0

    summary = correlate_records_in_memory(
        [earlier_pred, closer_pred], [sample_event]
    )

    assert summary.correlated_count == 1
    assert summary.records[0].prediction_id == "pred_closer"
    assert summary.uncorrelated_predictions_count == 1


@pytest.mark.asyncio
async def test_log_prediction_persists_to_mongo():
    """Verify log_prediction writes to prediction_logs collection."""
    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock()
    mock_db = {"prediction_logs": mock_collection}

    with patch(
        "app.infrastructure.monitoring.prediction_logger.MongoManager.get_database",
        return_value=mock_db,
    ):
        await log_prediction(
            prediction_id="pred_test_123",
            restaurant_id="rest_1",
            model_version="v1",
            predicted_wait_minutes=12.5,
            features={"party_size": 2.0},
            timestamp=datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc),
            queue_entry_id="qe_test_456",
            fallback_used=False,
        )

        mock_collection.insert_one.assert_awaited_once()
        saved = mock_collection.insert_one.await_args.args[0]
        assert saved["prediction_id"] == "pred_test_123"
        assert saved["queue_entry_id"] == "qe_test_456"
        assert saved["predicted_wait_minutes"] == 12.5
        assert saved["features"] == {"party_size": 2.0}


@pytest.mark.asyncio
async def test_log_prediction_handles_mongo_failure_gracefully():
    """Verify prediction logging failure does not raise or crash inference."""
    with patch(
        "app.infrastructure.monitoring.prediction_logger.MongoManager.get_database",
        return_value=None,
    ):
        # Should not raise exception
        await log_prediction(
            prediction_id="pred_test_123",
            restaurant_id="rest_1",
            model_version="v1",
            predicted_wait_minutes=12.5,
            features={"party_size": 2.0},
            timestamp=datetime.now(timezone.utc),
        )


@pytest.mark.asyncio
async def test_log_prediction_background_schedules_task():
    """Verify log_prediction_background returns an asyncio.Task."""
    with patch(
        "app.infrastructure.monitoring.prediction_logger.MongoManager.get_database",
        return_value=None,
    ):
        task = log_prediction_background(
            prediction_id="pred_bg_1",
            restaurant_id="rest_1",
            model_version="v1",
            predicted_wait_minutes=10.0,
            features={},
            timestamp=datetime.now(timezone.utc),
        )
        await task


@pytest.mark.asyncio
async def test_correlate_predictions_with_outcomes_queries_database():
    """Verify correlate_predictions_with_outcomes handles database queries."""
    with patch(
        "app.application.monitoring.correlation.MongoManager.get_database",
        return_value=None,
    ):
        summary = await correlate_predictions_with_outcomes(
            start_time=datetime(2026, 11, 6, 10, 0),
            end_time=datetime(2026, 11, 6, 12, 0),
        )
        assert summary.total_predictions == 0
        assert summary.correlated_count == 0

