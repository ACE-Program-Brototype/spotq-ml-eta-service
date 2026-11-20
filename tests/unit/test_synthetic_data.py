from datetime import datetime, timedelta, timezone

from scripts.seed_training_data import generate_synthetic_records


def test_generator_returns_requested_number_of_records():
    records = generate_synthetic_records(count=2000, seed=42)

    assert len(records) == 2000
    assert len({record["queue_entry_id"] for record in records}) == 2000


def test_generator_is_reproducible():
    first = generate_synthetic_records(count=25, seed=7)
    second = generate_synthetic_records(count=25, seed=7)

    assert first == second


def test_generator_uses_multiple_restaurants():
    records = generate_synthetic_records(
        count=100,
        seed=42,
        restaurant_count=5,
    )

    assert len({record["restaurant_id"] for record in records}) == 5
    assert all(record["is_synthetic"] is True for record in records)


def test_required_queue_event_schema_and_wait_duration():
    records = generate_synthetic_records(count=100, seed=42)

    required_fields = {
        "queue_entry_id",
        "restaurant_id",
        "party_size",
        "joined_at",
        "completed_at",
        "actual_wait_time_minutes",
        "total_people_ahead_at_join",
        "is_synthetic",
        "synthetic_seed",
        "synthetic_dataset_version",
    }

    for record in records:
        assert required_fields.issubset(record)
        assert record["party_size"] >= 1
        assert record["total_people_ahead_at_join"] >= 0
        assert 1.0 <= record["actual_wait_time_minutes"] <= 120.0

        joined_at = datetime.fromisoformat(record["joined_at"])
        completed_at = datetime.fromisoformat(record["completed_at"])

        assert joined_at.tzinfo is not None
        assert completed_at.tzinfo is not None
        assert completed_at > joined_at

        duration = (completed_at - joined_at).total_seconds() / 60
        assert abs(duration - record["actual_wait_time_minutes"]) < 0.001


def test_generator_rejects_invalid_arguments():
    import pytest

    with pytest.raises(ValueError):
        generate_synthetic_records(count=0)

    with pytest.raises(ValueError):
        generate_synthetic_records(restaurant_count=0)

    with pytest.raises(ValueError):
        generate_synthetic_records(days=0)


def test_generated_timestamps_are_inside_requested_window():
    end_at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    records = generate_synthetic_records(
        count=500,
        seed=42,
        days=30,
        end_at=end_at,
    )
    start_at = end_at - timedelta(days=30)

    for record in records:
        joined_at = datetime.fromisoformat(record["joined_at"])
        assert start_at <= joined_at < end_at


def test_dataset_configuration_changes_stable_ids():
    baseline = generate_synthetic_records(
        count=20, seed=42, restaurant_count=5, days=90
    )
    different_window = generate_synthetic_records(
        count=20, seed=42, restaurant_count=5, days=30
    )
    different_restaurant_count = generate_synthetic_records(
        count=20, seed=42, restaurant_count=4, days=90
    )

    baseline_ids = {record["queue_entry_id"] for record in baseline}
    window_ids = {record["queue_entry_id"] for record in different_window}
    restaurant_ids = {
        record["queue_entry_id"] for record in different_restaurant_count
    }

    assert baseline_ids.isdisjoint(window_ids)
    assert baseline_ids.isdisjoint(restaurant_ids)
