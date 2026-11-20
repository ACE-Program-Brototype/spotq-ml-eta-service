from datetime import datetime, timedelta, timezone

import pytest

from app.infrastructure.ml_models.dataset import (
    chronological_train_test_split,
    make_purged_time_series_cv,
)


def make_records(count=36, interval_minutes=60, wait_minutes=75):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    records = []

    for index in range(count):
        joined_at = start + timedelta(minutes=index * interval_minutes)
        completed_at = joined_at + timedelta(minutes=wait_minutes)

        records.append(
            {
                "queue_entry_id": f"event-{index:03d}",
                "restaurant_id": "restaurant-001",
                "joined_at": joined_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "actual_wait_time_minutes": float(wait_minutes),
            }
        )

    return records


def test_train_test_split_is_chronological_and_purged():
    split = chronological_train_test_split(
        make_records(), test_fraction=0.20
    )

    assert split.train_records
    assert split.test_records
    assert split.cutoff == min(
        datetime.fromisoformat(record["joined_at"])
        for record in split.test_records
    )

    assert all(
        datetime.fromisoformat(record["joined_at"]) < split.cutoff
        and datetime.fromisoformat(record["completed_at"]) < split.cutoff
        for record in split.train_records
    )

    assert all(
        datetime.fromisoformat(record["joined_at"]) >= split.cutoff
        for record in split.test_records
    )

    assert all(
        datetime.fromisoformat(record["joined_at"]) < split.cutoff
        and datetime.fromisoformat(record["completed_at"]) >= split.cutoff
        for record in split.embargo_records
    )


def test_split_preserves_all_records_as_train_test_or_embargo():
    records = make_records()
    split = chronological_train_test_split(records)

    all_ids = {
        record["queue_entry_id"]
        for record in (
            list(split.train_records)
            + list(split.test_records)
            + list(split.embargo_records)
        )
    }

    assert all_ids == {record["queue_entry_id"] for record in records}


def test_cv_folds_are_purged_and_chronological():
    cv = make_purged_time_series_cv(make_records(), n_splits=3)

    assert len(cv.folds) == 3

    previous_validation_end = None

    for train_indices, validation_indices in cv.folds:
        train_records = [cv.records[i] for i in train_indices]
        validation_records = [cv.records[i] for i in validation_indices]

        validation_start = min(
            datetime.fromisoformat(record["joined_at"])
            for record in validation_records
        )

        assert train_indices
        assert validation_indices

        assert all(
            datetime.fromisoformat(record["joined_at"]) < validation_start
            and datetime.fromisoformat(record["completed_at"]) < validation_start
            for record in train_records
        )

        assert all(
            datetime.fromisoformat(record["joined_at"]) >= validation_start
            for record in validation_records
        )

        if previous_validation_end is not None:
            assert validation_start > previous_validation_end

        previous_validation_end = max(
            datetime.fromisoformat(record["joined_at"])
            for record in validation_records
        )


def test_cv_does_not_split_records_with_identical_join_times():
    records = make_records(count=20, wait_minutes=5)
    tied_time = datetime.fromisoformat(records[-1]["joined_at"])

    for index in range(17, 20):
        records[index]["joined_at"] = tied_time.isoformat()
        records[index]["completed_at"] = (
            tied_time + timedelta(minutes=5)
        ).isoformat()

    cv = make_purged_time_series_cv(records, n_splits=3)

    for _, validation_indices in cv.folds:
        validation_times = {
            cv.records[i]["joined_at"] for i in validation_indices
        }

        for record in cv.records:
            if record["joined_at"] in validation_times:
                assert record["queue_entry_id"] in {
                    cv.records[i]["queue_entry_id"]
                    for i in validation_indices
                }


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.5, float("nan")])
def test_invalid_test_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        chronological_train_test_split(
            make_records(), test_fraction=fraction
        )


def test_insufficient_unique_timestamps_rejected():
    records = make_records(count=6)

    for record in records:
        record["joined_at"] = "2026-09-01T12:00:00+00:00"
        record["completed_at"] = "2026-09-01T12:05:00+00:00"

    with pytest.raises(ValueError, match="unique join timestamps"):
        make_purged_time_series_cv(records, n_splits=3)
