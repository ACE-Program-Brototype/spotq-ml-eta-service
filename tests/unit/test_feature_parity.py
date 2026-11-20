"""Automated tests verifying feature parity between training and serving contracts."""

from datetime import datetime, timedelta, timezone
import pytest

from app.infrastructure.config.settings import settings
from app.infrastructure.ml_models.features import (
    FEATURE_NAMES,
    build_feature_row,
    build_training_features,
)


def make_raw_record(
    event_id: str,
    restaurant_id: str,
    party_size: int,
    people_ahead: int,
    joined_at: datetime,
    wait_minutes: float,
) -> dict:
    return {
        "queue_entry_id": event_id,
        "restaurant_id": restaurant_id,
        "party_size": party_size,
        "total_people_ahead_at_join": people_ahead,
        "joined_at": joined_at.isoformat(),
        "completed_at": (joined_at + timedelta(minutes=wait_minutes)).isoformat(),
        "actual_wait_time_minutes": wait_minutes,
    }


def test_feature_parity_schema_and_ordering():
    """Verify both pipelines generate keys in exact FEATURE_NAMES tuple order."""
    record = make_raw_record(
        "parity-1",
        "restaurant-nyc",
        4,
        10,
        datetime(2026, 10, 15, 18, 30, tzinfo=timezone.utc),
        20.0,
    )
    tz = "America/New_York"
    hist_avg = 18.5

    # Serving feature assembly
    serving_features = build_feature_row(
        record,
        historical_average_wait=hist_avg,
        restaurant_timezone=tz,
    )

    # Training feature assembly
    training_features, targets = build_training_features(
        [record],
        cold_start_average=hist_avg,
        restaurant_timezones={"restaurant-nyc": tz},
    )

    assert tuple(serving_features.keys()) == FEATURE_NAMES
    assert tuple(training_features[0].keys()) == FEATURE_NAMES
    assert len(serving_features) == 6
    assert len(training_features[0]) == 6


def test_feature_value_parity_across_timezones():
    """Verify local hour, day of week, and weekend flag match across timezones."""
    # 2026-10-18 is Sunday. 20:00 UTC = Sunday 16:00 in NY, but Monday 01:30 in Kolkata.
    sunday_utc = datetime(2026, 10, 18, 20, 0, tzinfo=timezone.utc)

    # Kolkata test
    record_kolkata = make_raw_record(
        "kolkata-1", "rest-kolkata", 2, 5, sunday_utc, 15.0
    )
    serving_kolkata = build_feature_row(
        record_kolkata, 22.0, restaurant_timezone="Asia/Kolkata"
    )
    training_kolkata, _ = build_training_features(
        [record_kolkata],
        cold_start_average=22.0,
        restaurant_timezones={"rest-kolkata": "Asia/Kolkata"},
    )

    assert serving_kolkata == training_kolkata[0]
    assert serving_kolkata["hour_of_day"] == 1
    assert serving_kolkata["day_of_week"] == 0  # Monday in local time
    assert serving_kolkata["is_weekend"] == 0

    # NY test
    record_ny = make_raw_record("ny-1", "rest-ny", 2, 5, sunday_utc, 15.0)
    serving_ny = build_feature_row(
        record_ny, 22.0, restaurant_timezone="America/New_York"
    )
    training_ny, _ = build_training_features(
        [record_ny],
        cold_start_average=22.0,
        restaurant_timezones={"rest-ny": "America/New_York"},
    )

    assert serving_ny == training_ny[0]
    assert serving_ny["hour_of_day"] == 16
    assert serving_ny["day_of_week"] == 6  # Sunday in local time
    assert serving_ny["is_weekend"] == 1


def test_cold_start_fallback_parity():
    """Verify both training and serving use identical cold-start values when history is missing."""
    record = make_raw_record(
        "cold-1",
        "rest-new",
        3,
        2,
        datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc),
        12.0,
    )
    fallback = settings.COLD_START_AVERAGE_WAIT_MINUTES

    serving = build_feature_row(
        record,
        historical_average_wait=fallback,
        restaurant_timezone="UTC",
    )
    training, _ = build_training_features(
        [record],
        cold_start_average=fallback,
        restaurant_timezones={"rest-new": "UTC"},
    )

    assert serving["historical_average_wait"] == pytest.approx(fallback)
    assert training[0]["historical_average_wait"] == pytest.approx(fallback)
    assert serving == training[0]
