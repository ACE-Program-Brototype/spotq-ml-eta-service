from copy import deepcopy

import pytest

from app.infrastructure.ml_models.data_cleaning import clean_training_records


def valid_record(**overrides):
    record = {
        "queue_entry_id": "event-001",
        "restaurant_id": "mock-restaurant-001",
        "party_size": 4,
        "joined_at": "2026-09-22T12:00:00Z",
        "completed_at": "2026-09-22T12:25:00Z",
        "actual_wait_time_minutes": 25.0,
        "total_people_ahead_at_join": 10,
        "is_synthetic": True,
    }
    record.update(overrides)
    return record


def test_valid_record_is_accepted():
    accepted, report = clean_training_records([valid_record()])

    assert len(accepted) == 1
    assert report["accepted_records"] == 1
    assert report["rejected_records"] == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"party_size": 0}, "invalid_party_size"),
        ({"party_size": True}, "invalid_party_size"),
        ({"total_people_ahead_at_join": -1}, "invalid_people_ahead"),
        ({"actual_wait_time_minutes": 0}, "non_positive_wait_time"),
        ({"actual_wait_time_minutes": -2}, "non_positive_wait_time"),
        ({"joined_at": "2026-09-22T12:00:00"}, "invalid_timestamp"),
        ({"completed_at": "not-a-timestamp"}, "invalid_timestamp"),
        ({"queue_entry_id": ""}, "invalid_queue_entry_id"),
        ({"restaurant_id": ""}, "invalid_restaurant_id"),
    ],
)
def test_invalid_records_are_rejected_with_reason(overrides, reason):
    record = valid_record(**overrides)
    accepted, report = clean_training_records([record])

    assert accepted == []
    assert report["rejected_records"] == 1
    assert report["rejection_reasons"][reason] == 1


def test_missing_required_field_is_rejected():
    record = valid_record()
    del record["party_size"]

    accepted, report = clean_training_records([record])

    assert accepted == []
    assert report["rejection_reasons"]["missing_required_fields"] == 1


def test_completed_before_join_is_rejected():
    record = valid_record(
        completed_at="2026-09-22T11:59:00Z",
        actual_wait_time_minutes=1.0,
    )

    accepted, report = clean_training_records([record])

    assert accepted == []
    assert report["rejection_reasons"]["invalid_timestamp_order"] == 1


def test_duplicate_event_ids_are_deduplicated():
    records = [
        valid_record(),
        valid_record(actual_wait_time_minutes=25.0),
    ]

    accepted, report = clean_training_records(records)

    assert len(accepted) == 1
    assert report["rejection_reasons"]["duplicate_queue_entry_id"] == 1


def test_corrupt_first_duplicate_does_not_block_valid_record():
    records = [
        valid_record(party_size=0),
        valid_record(),
    ]

    accepted, report = clean_training_records(records)

    assert len(accepted) == 1
    assert report["rejection_reasons"]["invalid_party_size"] == 1


def test_cleaner_does_not_mutate_input():
    records = [valid_record()]
    original = deepcopy(records)

    clean_training_records(records)

    assert records == original


def test_mixed_dataset_report_is_auditable():
    records = [
        valid_record(queue_entry_id="event-001"),
        valid_record(queue_entry_id="event-002", actual_wait_time_minutes=0),
        valid_record(queue_entry_id="event-003", party_size=0),
        valid_record(
            queue_entry_id="event-004",
            completed_at="2026-09-22T14:01:00Z",
            actual_wait_time_minutes=121,
        ),
    ]

    accepted, report = clean_training_records(records)

    assert len(accepted) == 1
    assert report["input_records"] == 4
    assert report["accepted_records"] == 1
    assert report["rejected_records"] == 3
    assert report["rejection_reasons"]["non_positive_wait_time"] == 1
    assert report["rejection_reasons"]["invalid_party_size"] == 1
    assert report["rejection_reasons"]["wait_time_above_limit"] == 1
    assert (
        report["accepted_records"] + report["rejected_records"]
        == report["input_records"]
    )



def test_wait_above_120_minutes_is_rejected_by_training_policy():
    record = valid_record(
        joined_at="2026-09-22T12:00:00Z",
        completed_at="2026-09-22T14:01:00Z",
        actual_wait_time_minutes=121.0,
    )

    accepted, report = clean_training_records([record])

    assert accepted == []
    assert report["rejection_reasons"]["wait_time_above_limit"] == 1



def test_wait_above_120_minutes_is_rejected_by_training_policy():
    record = valid_record(
        joined_at="2026-09-22T12:00:00Z",
        completed_at="2026-09-22T14:01:00Z",
        actual_wait_time_minutes=121.0,
    )

    accepted, report = clean_training_records([record])

    assert accepted == []
    assert report["rejection_reasons"]["wait_time_above_limit"] == 1
