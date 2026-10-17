"""Auditable validation and cleaning for SpotQ ETA training records."""

from collections import Counter
from datetime import datetime, timezone
import math
from typing import Any, Iterable

REQUIRED_FIELDS = (
    "queue_entry_id",
    "restaurant_id",
    "party_size",
    "joined_at",
    "completed_at",
    "actual_wait_time_minutes",
    "total_people_ahead_at_join",
)

MAX_PARTY_SIZE = 100
MAX_PEOPLE_AHEAD = 1000
MIN_TRAINING_WAIT_MINUTES = 0.0
MAX_TRAINING_WAIT_MINUTES = 120.0
DURATION_TOLERANCE_MINUTES = 0.1


def _parse_aware_timestamp(value: Any) -> datetime:
    """Parse an ISO timestamp and require timezone information."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("invalid_timestamp")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone_missing")

    return parsed


def _is_integer(value: Any) -> bool:
    """Return whether a value is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def clean_training_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate records and return accepted training records plus an audit report.

    A record is accepted only if all required fields are valid, timestamps are
    consistent, the observed duration agrees with the target, and the target
    is strictly between zero and 120 minutes inclusive of the upper bound.

    The input records are never mutated.
    """
    accepted: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    input_count = 0

    for original in records:
        input_count += 1

        if not isinstance(original, dict):
            rejection_reasons["record_not_object"] += 1
            continue

        record = dict(original)

        missing = [
            field
            for field in REQUIRED_FIELDS
            if field not in record or record[field] is None
        ]
        if missing:
            rejection_reasons["missing_required_fields"] += 1
            continue

        event_id = record["queue_entry_id"]
        restaurant_id = record["restaurant_id"]

        if not isinstance(event_id, str) or not event_id.strip():
            rejection_reasons["invalid_queue_entry_id"] += 1
            continue

        if not isinstance(restaurant_id, str) or not restaurant_id.strip():
            rejection_reasons["invalid_restaurant_id"] += 1
            continue

        # Validate all fields before reserving the ID. A corrupt record should
        # not prevent a later valid record with the same ID from being used.
        party_size = record["party_size"]
        people_ahead = record["total_people_ahead_at_join"]

        if (
            not _is_integer(party_size)
            or not 1 <= party_size <= MAX_PARTY_SIZE
        ):
            rejection_reasons["invalid_party_size"] += 1
            continue

        if (
            not _is_integer(people_ahead)
            or not 0 <= people_ahead <= MAX_PEOPLE_AHEAD
        ):
            rejection_reasons["invalid_people_ahead"] += 1
            continue

        target = record["actual_wait_time_minutes"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            rejection_reasons["invalid_wait_time_type"] += 1
            continue

        target = float(target)
        if not math.isfinite(target):
            rejection_reasons["non_finite_wait_time"] += 1
            continue

        try:
            joined_at = _parse_aware_timestamp(record["joined_at"])
            completed_at = _parse_aware_timestamp(record["completed_at"])
        except (ValueError, TypeError, OverflowError):
            rejection_reasons["invalid_timestamp"] += 1
            continue

        if joined_at >= completed_at:
            rejection_reasons["invalid_timestamp_order"] += 1
            continue

        # Compare instants in UTC. Different timezone offsets are valid.
        # Apply training-target bounds before comparing target and timestamps.
        # This ensures out-of-policy targets receive their intended audit reason.
        if target <= MIN_TRAINING_WAIT_MINUTES:
            rejection_reasons["non_positive_wait_time"] += 1
            continue

        if target > MAX_TRAINING_WAIT_MINUTES:
            rejection_reasons["wait_time_above_limit"] += 1
            continue

        observed_duration = (
            completed_at.astimezone(timezone.utc)
            - joined_at.astimezone(timezone.utc)
        ).total_seconds() / 60.0

        if abs(observed_duration - target) > DURATION_TOLERANCE_MINUTES:
            rejection_reasons["duration_target_mismatch"] += 1
            continue

        if event_id in seen_ids:
            rejection_reasons["duplicate_queue_entry_id"] += 1
            continue

        seen_ids.add(event_id)

        # Normalize validated values in the returned copy.
        record["actual_wait_time_minutes"] = target
        record["joined_at"] = joined_at.isoformat()
        record["completed_at"] = completed_at.isoformat()
        accepted.append(record)

    rejected_count = sum(rejection_reasons.values())

    report = {
        "input_records": input_count,
        "accepted_records": len(accepted),
        "rejected_records": rejected_count,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "training_wait_bounds_minutes": {
            "exclusive_minimum": MIN_TRAINING_WAIT_MINUTES,
            "inclusive_maximum": MAX_TRAINING_WAIT_MINUTES,
        },
    }

    return accepted, report
