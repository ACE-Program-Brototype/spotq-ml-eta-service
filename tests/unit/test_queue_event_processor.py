"""Unit tests for durable queue-event persistence and idempotency."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

from app.application import queue_event_processor as processor


@pytest.fixture
def event_data():
    """Return a valid completed queue event."""
    return {
        "queue_entry_id": "entry-1",
        "restaurant_id": "restaurant-1",
        "party_size": 2,
        "joined_at": "2026-10-18T12:00:00Z",
        "completed_at": "2026-10-18T12:10:00Z",
        "actual_wait_time_minutes": 10.0,
        "total_people_ahead_at_join": 3,
    }


@pytest.fixture
def mongo_collection(monkeypatch):
    """Mock the MongoDB database and queue-events collection."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    collection.find_one = AsyncMock()

    database = MagicMock()
    database.__getitem__.return_value = collection

    monkeypatch.setattr(
        processor.MongoManager,
        "get_database",
        staticmethod(lambda: database),
    )

    return collection


@pytest.mark.asyncio
async def test_new_event_is_persisted_with_utc_timestamps(
    event_data, mongo_collection
):
    """Persist a new event with normalized timestamps and metadata."""
    result = await processor.process_queue_event(event_data)

    assert result == {
        "status": "persisted",
        "queue_entry_id": "entry-1",
    }

    mongo_collection.insert_one.assert_awaited_once()
    saved = mongo_collection.insert_one.await_args.args[0]

    assert saved["queue_entry_id"] == "entry-1"
    assert saved["joined_at"] == datetime(
        2026, 10, 18, 12, 0, tzinfo=timezone.utc
    )
    assert saved["completed_at"] == datetime(
        2026, 10, 18, 12, 10, tzinfo=timezone.utc
    )
    assert saved["is_synthetic"] is False
    assert isinstance(saved["received_at"], datetime)
    assert saved["received_at"].tzinfo is not None
    mongo_collection.find_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_identical_duplicate_is_idempotent(
    event_data, mongo_collection
):
    """Accept a duplicate event without inserting another record."""
    mongo_collection.insert_one.side_effect = DuplicateKeyError(
        "duplicate queue_entry_id"
    )

    existing = {
        "queue_entry_id": "entry-1",
        "restaurant_id": "restaurant-1",
        "party_size": 2,
        "joined_at": datetime(2026, 10, 18, 12, 0),
        "completed_at": datetime(2026, 10, 18, 12, 10),
        "actual_wait_time_minutes": 10.0,
        "total_people_ahead_at_join": 3,
    }
    mongo_collection.find_one.return_value = existing

    result = await processor.process_queue_event(event_data)

    assert result == {
        "status": "duplicate",
        "queue_entry_id": "entry-1",
    }
    mongo_collection.insert_one.assert_awaited_once()
    mongo_collection.find_one.assert_awaited_once_with(
        {"queue_entry_id": "entry-1"}
    )


@pytest.mark.asyncio
async def test_conflicting_duplicate_is_rejected(
    event_data, mongo_collection
):
    """Reject reuse of an event ID with different event data."""
    mongo_collection.insert_one.side_effect = DuplicateKeyError(
        "duplicate queue_entry_id"
    )

    existing = {
        "queue_entry_id": "entry-1",
        "restaurant_id": "restaurant-1",
        "party_size": 2,
        "joined_at": datetime(2026, 10, 18, 12, 0),
        "completed_at": datetime(2026, 10, 18, 12, 10),
        "actual_wait_time_minutes": 10.0,
        "total_people_ahead_at_join": 3,
    }
    mongo_collection.find_one.return_value = existing

    event_data["party_size"] = 4

    with pytest.raises(
        processor.QueueEventConflictError,
        match="Conflicting event already exists",
    ):
        await processor.process_queue_event(event_data)

    mongo_collection.insert_one.assert_awaited_once()
    mongo_collection.find_one.assert_awaited_once_with(
        {"queue_entry_id": "entry-1"}
    )


@pytest.mark.asyncio
async def test_transient_database_error_propagates(
    event_data, mongo_collection
):
    """Propagate transient MongoDB failures so BullMQ can retry."""
    mongo_collection.insert_one.side_effect = ConnectionError(
        "MongoDB temporarily unavailable"
    )

    with pytest.raises(
        ConnectionError, match="MongoDB temporarily unavailable"
    ):
        await processor.process_queue_event(event_data)

    mongo_collection.insert_one.assert_awaited_once()
    mongo_collection.find_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_wait_duration_is_rejected(
    event_data, mongo_collection
):
    """Reject events whose duration disagrees with their timestamps."""
    event_data["actual_wait_time_minutes"] = 15.0

    with pytest.raises(
        processor.QueueEventValidationError,
        match="wait time",
    ):
        await processor.process_queue_event(event_data)

    mongo_collection.insert_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_unique_index_is_created(monkeypatch):
    """Ensure the event ID index is unique before consumption."""
    collection = MagicMock()
    collection.create_index = AsyncMock()

    database = MagicMock()
    database.__getitem__.return_value = collection

    monkeypatch.setattr(
        processor.MongoManager,
        "get_database",
        staticmethod(lambda: database),
    )

    await processor.ensure_queue_event_indexes()

    collection.create_index.assert_awaited_once_with(
        [("queue_entry_id", processor.ASCENDING)],
        unique=True,
        name="uq_queue_entry_id",
    )


@pytest.mark.asyncio
async def test_index_creation_fails_if_mongodb_unavailable(monkeypatch):
    """Prevent consumer startup when MongoDB is unavailable."""
    monkeypatch.setattr(
        processor.MongoManager,
        "get_database",
        staticmethod(lambda: None),
    )

    with pytest.raises(RuntimeError, match="MongoDB is unavailable"):
        await processor.ensure_queue_event_indexes()
