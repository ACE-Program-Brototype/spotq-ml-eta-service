"""Unit tests for monitoring data quality validation and exclusion guard (AC2)."""

from datetime import datetime, timezone

import pytest

from app.application.monitoring.correlation import CorrelatedRecord
from app.application.monitoring.data_quality import (
    validate_correlated_records,
    validate_prediction_features,
)


def make_valid_correlated_record(
    qid: str = "entry_1",
    actual_wait: float = 15.0,
    pred_wait: float = 14.0,
    override_features: dict | None = None,
    pred_time: datetime | None = None,
    outcome_time: datetime | None = None,
) -> CorrelatedRecord:
    features = {
        "party_size": 2.0,
        "total_people_ahead": 3.0,
        "hour_of_day": 12.0,
        "day_of_week": 2.0,
        "is_weekend": 0.0,
        "historical_average_wait": 14.5,
    }
    if override_features:
        features.update(override_features)

    p_time = pred_time or datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    o_time = outcome_time or datetime(2026, 11, 6, 12, 15, tzinfo=timezone.utc)

    return CorrelatedRecord(
        prediction_id=f"pred_{qid}",
        queue_entry_id=qid,
        restaurant_id="rest_1",
        model_version="v1",
        prediction_timestamp=p_time,
        outcome_timestamp=o_time,
        predicted_wait_minutes=pred_wait,
        actual_wait_minutes=actual_wait,
        features=features,
        correlation_method="exact_queue_entry_id",
    )


def test_valid_records_are_accepted():
    """AC2: Clean records with complete features pass validation without rejections."""
    records = [make_valid_correlated_record(f"entry_{i}") for i in range(35)]
    report = validate_correlated_records(records, min_sample_size=30)

    assert report.total_evaluated == 35
    assert report.valid_count == 35
    assert report.excluded_count == 0
    assert report.is_sufficient_for_evaluation is True
    assert report.sufficiency_message is None
    assert len(report.valid_records) == 35


def test_missing_features_are_excluded():
    """AC2: Records with missing canonical features are rejected and counted."""
    incomplete_record = make_valid_correlated_record("bad_1")
    del incomplete_record.features["historical_average_wait"]

    valid_record = make_valid_correlated_record("good_1")

    report = validate_correlated_records([incomplete_record, valid_record], min_sample_size=1)

    assert report.total_evaluated == 2
    assert report.valid_count == 1
    assert report.excluded_count == 1
    assert report.rejections.missing_features == 1
    assert report.valid_records[0].queue_entry_id == "good_1"


def test_non_finite_features_are_excluded():
    """AC2: Non-finite values (NaN, Inf) are excluded."""
    nan_record = make_valid_correlated_record("nan_1", override_features={"party_size": float("nan")})
    inf_record = make_valid_correlated_record("inf_1", override_features={"hour_of_day": float("inf")})
    good_record = make_valid_correlated_record("good_1")

    report = validate_correlated_records([nan_record, inf_record, good_record], min_sample_size=1)

    assert report.total_evaluated == 3
    assert report.valid_count == 1
    assert report.excluded_count == 2
    assert report.rejections.non_finite_features == 2


@pytest.mark.parametrize("invalid_wait", [-5.0, 0.0, 120.5, float("inf"), float("nan")])
def test_out_of_bounds_target_wait_is_excluded(invalid_wait):
    """AC2: Targets outside (0, 120] minutes are rejected."""
    bad_record = make_valid_correlated_record("bad_1", actual_wait=invalid_wait)
    report = validate_correlated_records([bad_record], min_sample_size=1)

    assert report.valid_count == 0
    assert report.excluded_count == 1
    assert report.rejections.invalid_target_wait == 1


def test_duplicate_queue_entries_are_deduplicated():
    """AC2: Duplicate outcomes for the same queue entry are counted and excluded."""
    rec1 = make_valid_correlated_record("dup_entry")
    rec2 = make_valid_correlated_record("dup_entry")

    report = validate_correlated_records([rec1, rec2], min_sample_size=1)

    assert report.total_evaluated == 2
    assert report.valid_count == 1
    assert report.excluded_count == 1
    assert report.rejections.duplicate_queue_entry == 1


def test_insufficient_sample_size_flag():
    """AC2: Evaluator flags insufficient samples when valid records are below threshold."""
    records = [make_valid_correlated_record(f"entry_{i}") for i in range(5)]
    report = validate_correlated_records(records, min_sample_size=10)

    assert report.valid_count == 5
    assert report.is_sufficient_for_evaluation is False
    assert "below minimum required (10)" in report.sufficiency_message


def test_validate_prediction_features_for_drift():
    """AC2: Prediction log features are validated and filtered for drift analysis."""
    valid_pred = {
        "features": {
            "party_size": 2.0,
            "total_people_ahead": 3.0,
            "hour_of_day": 12.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "historical_average_wait": 14.5,
        }
    }
    missing_pred = {"features": {"party_size": 2.0}}
    nan_pred = {
        "features": {
            "party_size": 2.0,
            "total_people_ahead": float("nan"),
            "hour_of_day": 12.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "historical_average_wait": 14.5,
        }
    }

    clean_features, report = validate_prediction_features(
        [valid_pred, missing_pred, nan_pred], min_sample_size=1
    )

    assert len(clean_features) == 1
    assert report.total_evaluated == 3
    assert report.valid_count == 1
    assert report.rejections.missing_features == 1
    assert report.rejections.non_finite_features == 1

