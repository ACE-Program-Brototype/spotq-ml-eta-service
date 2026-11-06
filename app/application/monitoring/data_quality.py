"""Data quality validator and exclusion guard for production model monitoring (AC2).

Identifies and excludes invalid, out-of-bounds, duplicate, or non-finite records
so data quality defects cannot corrupt model performance metrics or silently
trigger unsuitable retraining.
"""

import math
from typing import Any

from pydantic import BaseModel, Field

from app.application.monitoring.correlation import CorrelatedRecord
from app.infrastructure.ml_models.features import FEATURE_NAMES

DEFAULT_MIN_SAMPLE_SIZE = 30
MIN_WAIT_MINUTES = 0.0
MAX_WAIT_MINUTES = 120.0


class DataQualityRejectionBreakdown(BaseModel):
    """Counts of records excluded categorized by rejection reason."""

    missing_features: int = 0
    non_finite_features: int = 0
    invalid_target_wait: int = 0
    invalid_predicted_wait: int = 0
    duplicate_queue_entry: int = 0
    invalid_timestamps: int = 0


class DataQualityReport(BaseModel):
    """Result of data quality validation for a monitoring evaluation batch."""

    total_evaluated: int
    valid_count: int
    excluded_count: int
    rejections: DataQualityRejectionBreakdown
    is_sufficient_for_evaluation: bool
    sufficiency_message: str | None = None
    valid_records: list[CorrelatedRecord] = Field(default_factory=list)


def validate_correlated_records(
    records: list[CorrelatedRecord],
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> DataQualityReport:
    """Filter correlated records against data quality rules, returning clean records and metrics."""
    rejections = DataQualityRejectionBreakdown()
    valid_records: list[CorrelatedRecord] = []
    seen_queue_entry_ids: set[str] = set()

    for record in records:
        # 1. Duplicate queue_entry_id check
        qid = str(record.queue_entry_id).strip()
        if qid in seen_queue_entry_ids:
            rejections.duplicate_queue_entry += 1
            continue

        # 2. Check canonical features existence
        features = record.features or {}
        missing = [f for f in FEATURE_NAMES if f not in features or features[f] is None]
        if missing:
            rejections.missing_features += 1
            continue

        # 3. Check feature finiteness
        try:
            if not all(math.isfinite(float(features[f])) for f in FEATURE_NAMES):
                rejections.non_finite_features += 1
                continue
        except (ValueError, TypeError):
            rejections.non_finite_features += 1
            continue

        # 4. Check actual wait time target bounds (0, 120]
        actual_wait = record.actual_wait_minutes
        if (
            not isinstance(actual_wait, (int, float))
            or not math.isfinite(actual_wait)
            or actual_wait <= MIN_WAIT_MINUTES
            or actual_wait > MAX_WAIT_MINUTES
        ):
            rejections.invalid_target_wait += 1
            continue

        # 5. Check predicted wait bounds
        pred_wait = record.predicted_wait_minutes
        if (
            not isinstance(pred_wait, (int, float))
            or not math.isfinite(pred_wait)
            or pred_wait <= MIN_WAIT_MINUTES
            or pred_wait > MAX_WAIT_MINUTES
        ):
            rejections.invalid_predicted_wait += 1
            continue

        # 6. Check timestamps consistency
        if record.outcome_timestamp <= record.prediction_timestamp:
            rejections.invalid_timestamps += 1
            continue

        # Record is valid
        seen_queue_entry_ids.add(qid)
        valid_records.append(record)

    total_evaluated = len(records)
    valid_count = len(valid_records)
    excluded_count = total_evaluated - valid_count

    is_sufficient = valid_count >= min_sample_size
    sufficiency_message = (
        None
        if is_sufficient
        else f"Valid sample count ({valid_count}) is below minimum required ({min_sample_size})"
    )

    return DataQualityReport(
        total_evaluated=total_evaluated,
        valid_count=valid_count,
        excluded_count=excluded_count,
        rejections=rejections,
        is_sufficient_for_evaluation=is_sufficient,
        sufficiency_message=sufficiency_message,
        valid_records=valid_records,
    )


def validate_prediction_features(
    predictions: list[dict[str, Any]],
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> tuple[list[dict[str, float]], DataQualityReport]:
    """Validate canonical features across prediction logs for drift monitoring."""
    rejections = DataQualityRejectionBreakdown()
    valid_features_list: list[dict[str, float]] = []

    for pred in predictions:
        features = pred.get("features", {})
        missing = [f for f in FEATURE_NAMES if f not in features or features[f] is None]
        if missing:
            rejections.missing_features += 1
            continue

        try:
            cleaned_row = {f: float(features[f]) for f in FEATURE_NAMES}
            if not all(math.isfinite(v) for v in cleaned_row.values()):
                rejections.non_finite_features += 1
                continue
        except (ValueError, TypeError):
            rejections.non_finite_features += 1
            continue

        valid_features_list.append(cleaned_row)

    total_evaluated = len(predictions)
    valid_count = len(valid_features_list)
    excluded_count = total_evaluated - valid_count

    is_sufficient = valid_count >= min_sample_size
    sufficiency_message = (
        None
        if is_sufficient
        else f"Valid sample count ({valid_count}) is below minimum required ({min_sample_size})"
    )

    report = DataQualityReport(
        total_evaluated=total_evaluated,
        valid_count=valid_count,
        excluded_count=excluded_count,
        rejections=rejections,
        is_sufficient_for_evaluation=is_sufficient,
        sufficiency_message=sufficiency_message,
        valid_records=[],
    )
    return valid_features_list, report

