"""Canonical, leakage-safe ETA feature engineering with local timezones."""

from collections import defaultdict
from datetime import datetime, timezone
import heapq
import math
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


FEATURE_NAMES = (
    "party_size",
    "total_people_ahead",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "historical_average_wait",
)

FEATURE_COUNT = len(FEATURE_NAMES)


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse a timestamp and normalize the instant to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("Timestamp must be a datetime or ISO string")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include timezone information")

    return parsed.astimezone(timezone.utc)


def _get_timezone(timezone_name: str) -> ZoneInfo:
    """Resolve an explicit IANA timezone or fail clearly."""
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("Restaurant timezone must be explicitly configured")

    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Invalid restaurant timezone: {timezone_name!r}"
        ) from exc


def _validate_cold_start_average(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Invalid cold_start_average")

    return float(value)


def build_feature_row(
    record: dict[str, Any],
    historical_average_wait: float,
    *,
    restaurant_timezone: str,
) -> dict[str, int | float]:
    """Build one canonical feature row using restaurant-local calendar time."""
    zone = _get_timezone(restaurant_timezone)
    local_joined_at = parse_timestamp(record["joined_at"]).astimezone(zone)

    party_size = record["party_size"]
    people_ahead = record["total_people_ahead_at_join"]

    if (
        isinstance(party_size, bool)
        or not isinstance(party_size, int)
        or not 1 <= party_size <= 100
    ):
        raise ValueError("Invalid party_size")

    if (
        isinstance(people_ahead, bool)
        or not isinstance(people_ahead, int)
        or not 0 <= people_ahead <= 1000
    ):
        raise ValueError("Invalid total_people_ahead_at_join")

    if (
        isinstance(historical_average_wait, bool)
        or not isinstance(historical_average_wait, (int, float))
        or not math.isfinite(historical_average_wait)
        or historical_average_wait < 0
    ):
        raise ValueError("Invalid historical_average_wait")

    day_of_week = local_joined_at.weekday()

    return {
        "party_size": party_size,
        "total_people_ahead": people_ahead,
        "hour_of_day": local_joined_at.hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "historical_average_wait": float(historical_average_wait),
    }


def build_training_features(
    records: Iterable[dict[str, Any]],
    cold_start_average: float,
    *,
    restaurant_timezones: Mapping[str, str],
) -> tuple[list[dict[str, int | float]], list[float]]:
    """Build features using only waits completed before each join time.

    All event comparisons use UTC instants. Calendar features and historical
    buckets use each restaurant's explicitly configured IANA timezone.
    """
    fallback = _validate_cold_start_average(cold_start_average)
    ordered = sorted(
        records,
        key=lambda record: parse_timestamp(record["joined_at"]),
    )

    pending: list[
        tuple[datetime, int, tuple[str, int, int], float]
    ] = []

    history: dict[tuple[str, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0]
    )

    features: list[dict[str, int | float]] = []
    targets: list[float] = []

    for index, record in enumerate(ordered):
        joined_at = parse_timestamp(record["joined_at"])
        completed_at = parse_timestamp(record["completed_at"])

        if completed_at <= joined_at:
            raise ValueError("Completion must be after join time")

        restaurant_id = record.get("restaurant_id")
        if not isinstance(restaurant_id, str) or not restaurant_id.strip():
            raise ValueError("Invalid restaurant_id")

        if restaurant_id not in restaurant_timezones:
            raise ValueError(
                f"Missing timezone configuration for restaurant {restaurant_id!r}"
            )

        zone = _get_timezone(restaurant_timezones[restaurant_id])
        local_joined_at = joined_at.astimezone(zone)

        # Only release events whose completion instant strictly precedes
        # this join instant. Simultaneous completion is not considered history.
        while pending and pending[0][0] < joined_at:
            _, _, bucket, wait = heapq.heappop(pending)
            history[bucket][0] += wait
            history[bucket][1] += 1

        bucket = (
            restaurant_id,
            local_joined_at.weekday(),
            local_joined_at.hour,
        )

        total, count = history[bucket]
        historical_average = total / count if count else fallback

        row = build_feature_row(
            record,
            historical_average,
            restaurant_timezone=restaurant_timezones[restaurant_id],
        )

        target_value = record["actual_wait_time_minutes"]
        if (
            isinstance(target_value, bool)
            or not isinstance(target_value, (int, float))
        ):
            raise ValueError("Invalid training target")

        target = float(target_value)
        if not math.isfinite(target) or not 0 < target <= 120:
            raise ValueError("Invalid training target")

        features.append(row)
        targets.append(target)

        heapq.heappush(
            pending,
            (completed_at, index, bucket, target),
        )

    return features, targets
