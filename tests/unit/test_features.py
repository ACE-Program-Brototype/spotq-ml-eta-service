from datetime import datetime, timedelta, timezone

import pytest

from app.infrastructure.ml_models.features import (
    FEATURE_NAMES,
    build_feature_row,
    build_training_features,
)


def make_record(
    event_id: str,
    joined_at: datetime,
    wait: float,
    restaurant_id: str = "restaurant-001",
):
    return {
        "queue_entry_id": event_id,
        "restaurant_id": restaurant_id,
        "party_size": 2,
        "total_people_ahead_at_join": 5,
        "joined_at": joined_at.isoformat(),
        "completed_at": (joined_at + timedelta(minutes=wait)).isoformat(),
        "actual_wait_time_minutes": wait,
    }


def test_canonical_feature_order():
    record = make_record(
        "event-1", datetime(2026, 10, 5, 12, tzinfo=timezone.utc), 10
    )

    row = build_feature_row(
        record, 15.0, restaurant_timezone="UTC"
    )

    assert tuple(row) == FEATURE_NAMES
    assert list(row.values()) == [2, 5, 12, 0, 0, 15.0]


def test_weekend_feature_uses_restaurant_local_day():
    # Sunday 20:00 UTC is Monday 01:30 in India.
    record = make_record(
        "event-1", datetime(2026, 10, 4, 20, tzinfo=timezone.utc), 10
    )

    row = build_feature_row(
        record, 15.0, restaurant_timezone="Asia/Kolkata"
    )

    assert row["hour_of_day"] == 1
    assert row["day_of_week"] == 0
    assert row["is_weekend"] == 0


def test_historical_average_uses_completed_records_only():
    monday = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    records = [
        make_record("first", monday, 20),
        make_record("second", monday + timedelta(minutes=5), 40),
        make_record("third", monday + timedelta(minutes=50), 10),
    ]

    rows, targets = build_training_features(
        records,
        12.0,
        restaurant_timezones={"restaurant-001": "UTC"},
    )

    assert [row["historical_average_wait"] for row in rows] == [
        12.0, 12.0, 30.0
    ]
    assert targets == [20.0, 40.0, 10.0]


def test_restaurant_histories_are_isolated():
    monday = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    records = [
        make_record("first", monday, 20, "restaurant-A"),
        make_record(
            "second",
            monday + timedelta(minutes=30),
            10,
            "restaurant-B",
        ),
    ]

    rows, _ = build_training_features(
        records,
        8.0,
        restaurant_timezones={
            "restaurant-A": "UTC",
            "restaurant-B": "UTC",
        },
    )

    assert [row["historical_average_wait"] for row in rows] == [8.0, 8.0]


def test_invalid_historical_average_rejected():
    record = make_record(
        "event-1", datetime(2026, 10, 5, 12, tzinfo=timezone.utc), 10
    )

    with pytest.raises(ValueError):
        build_feature_row(
            record, float("nan"), restaurant_timezone="UTC"
        )


def test_missing_restaurant_timezone_rejected():
    record = make_record(
        "event-1", datetime(2026, 10, 5, 12, tzinfo=timezone.utc), 10
    )

    with pytest.raises(ValueError, match="Missing timezone"):
        build_training_features(
            [record], 12.0, restaurant_timezones={}
        )


def test_invalid_timezone_rejected():
    record = make_record(
        "event-1", datetime(2026, 10, 5, 12, tzinfo=timezone.utc), 10
    )

    with pytest.raises(ValueError, match="Invalid restaurant timezone"):
        build_feature_row(
            record, 12.0, restaurant_timezone="Not/A_Real_Zone"
        )


def test_local_hour_is_used_for_historical_buckets():
    # Both events are Sunday evening in India, in the same local hour.
    first_join = datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc)
    second_join = datetime(2026, 10, 4, 18, 20, tzinfo=timezone.utc)

    records = [
        make_record("first", first_join, 10),
        make_record("second", second_join, 20),
    ]

    rows, _ = build_training_features(
        records,
        8.0,
        restaurant_timezones={"restaurant-001": "Asia/Kolkata"},
    )

    assert rows[0]["hour_of_day"] == 23
    assert rows[1]["hour_of_day"] == 23
    assert rows[1]["historical_average_wait"] == 10.0
