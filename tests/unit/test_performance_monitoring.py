"""Unit tests for outcome-based model performance evaluation with version attribution (AC4)."""

from datetime import datetime, timezone

import pytest

from app.application.monitoring.correlation import CorrelatedRecord
from app.application.monitoring.performance import (
    PerformanceEvaluator,
)


def make_record(
    version: str,
    actual: float,
    pred: float,
    qid: str = "entry_1",
) -> CorrelatedRecord:
    return CorrelatedRecord(
        prediction_id=f"pred_{qid}",
        queue_entry_id=qid,
        restaurant_id="rest_1",
        model_version=version,
        prediction_timestamp=datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc),
        outcome_timestamp=datetime(2026, 11, 6, 10, 20, tzinfo=timezone.utc),
        predicted_wait_minutes=pred,
        actual_wait_minutes=actual,
        features={"party_size": 2.0},
        correlation_method="exact_queue_entry_id",
    )


def test_performance_evaluation_exact_metrics():
    """AC4: MAE and RMSE match expected mathematical results."""
    evaluator = PerformanceEvaluator(
        min_sample_size=3, max_acceptable_mae=5.0, max_acceptable_rmse=8.0
    )

    # Actuals: [10, 20, 30], Preds: [12, 18, 35]
    # Abs errors: [2, 2, 5] -> MAE = 3.0
    # Squared errors: [4, 4, 25] = 33 / 3 = 11 -> RMSE = sqrt(11) = 3.32
    records = [
        make_record("v1", actual=10.0, pred=12.0, qid="e1"),
        make_record("v1", actual=20.0, pred=18.0, qid="e2"),
        make_record("v1", actual=30.0, pred=35.0, qid="e3"),
    ]

    report = evaluator.evaluate_performance(records)

    assert report.total_evaluated_samples == 3
    assert report.has_degradation is False

    v1_metrics = report.version_metrics["v1"]
    assert v1_metrics.is_sufficient_samples is True
    assert v1_metrics.is_degraded is False
    assert v1_metrics.mae_minutes == 3.0
    assert v1_metrics.rmse_minutes == pytest.approx(3.32, abs=0.01)
    assert v1_metrics.max_absolute_error_minutes == 5.0
    assert v1_metrics.over_prediction_pct == pytest.approx(66.7, abs=0.1)
    assert v1_metrics.under_prediction_pct == pytest.approx(33.3, abs=0.1)


def test_performance_evaluator_isolates_model_versions():
    """AC4: Predictions are attributed strictly to the originating model version."""
    evaluator = PerformanceEvaluator(min_sample_size=3)

    records = [
        # Version 1 records (excellent performance)
        make_record("v1", actual=10.0, pred=10.0, qid="v1_1"),
        make_record("v1", actual=15.0, pred=15.0, qid="v1_2"),
        make_record("v1", actual=20.0, pred=20.0, qid="v1_3"),
        # Version 2 records (degraded performance)
        make_record("v2", actual=10.0, pred=22.0, qid="v2_1"),
        make_record("v2", actual=15.0, pred=27.0, qid="v2_2"),
        make_record("v2", actual=20.0, pred=32.0, qid="v2_3"),
    ]

    report = evaluator.evaluate_performance(records)

    assert len(report.version_metrics) == 2
    assert report.version_metrics["v1"].mae_minutes == 0.0
    assert report.version_metrics["v1"].is_degraded is False

    assert report.version_metrics["v2"].mae_minutes == 12.0
    assert report.version_metrics["v2"].is_degraded is True
    assert report.has_degradation is True


def test_performance_evaluator_handles_insufficient_samples():
    """AC4: Model versions with fewer than min_sample_size are marked insufficient."""
    evaluator = PerformanceEvaluator(min_sample_size=30)

    # Only 5 records
    records = [make_record("v1", actual=10.0, pred=15.0, qid=f"e_{i}") for i in range(5)]

    report = evaluator.evaluate_performance(records)

    v1_metrics = report.version_metrics["v1"]
    assert v1_metrics.is_sufficient_samples is False
    assert v1_metrics.mae_minutes is None
    assert v1_metrics.rmse_minutes is None
    assert v1_metrics.is_degraded is False
    assert report.has_degradation is False


def test_performance_evaluator_flags_degradation_when_thresholds_exceeded():
    """AC4: Exceeding MAE threshold triggers degradation flag with diagnostic reason."""
    evaluator = PerformanceEvaluator(
        min_sample_size=3, max_acceptable_mae=4.0, max_acceptable_rmse=6.0
    )

    records = [
        make_record("candidate_v2", actual=10.0, pred=16.0, qid="e1"),  # error 6
        make_record("candidate_v2", actual=12.0, pred=19.0, qid="e2"),  # error 7
        make_record("candidate_v2", actual=15.0, pred=21.0, qid="e3"),  # error 6
    ]

    report = evaluator.evaluate_performance(records)

    cand = report.version_metrics["candidate_v2"]
    assert cand.is_degraded is True
    assert report.has_degradation is True
    assert any("MAE" in reason for reason in cand.degradation_reasons)
