"""Tests for BullMQ event processing and failure classification."""

from types import SimpleNamespace

import pytest
from bullmq import UnrecoverableError

from app.application.queue_event_processor import (
    QueueEventConflictError,
    QueueEventValidationError,
)
from app.infrastructure.queue import bullmq_worker


@pytest.fixture
def job():
    """Provide a minimal BullMQ-like job for processor tests."""
    return SimpleNamespace(
        id="test-job-1",
        data={
            "queue_entry_id": "entry-1",
            "restaurant_id": "restaurant-1",
            "party_size": 2,
            "joined_at": "2026-10-18T12:00:00Z",
            "completed_at": "2026-10-18T12:10:00Z",
            "actual_wait_time_minutes": 10.0,
            "total_people_ahead_at_join": 3,
        },
    )


@pytest.mark.asyncio
async def test_process_job_returns_success(job, monkeypatch):
    """Return the persistence result when an event is processed."""
    async def fake_process(data):
        assert data == job.data
        return {"status": "persisted", "queue_entry_id": "entry-1"}

    monkeypatch.setattr(
        bullmq_worker, "process_queue_event", fake_process
    )

    result = await bullmq_worker.process_bullmq_job(job, "token")

    assert result == {
        "status": "persisted",
        "queue_entry_id": "entry-1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        QueueEventValidationError("invalid event"),
        QueueEventConflictError("conflicting event"),
    ],
)
async def test_permanent_errors_become_unrecoverable(job, monkeypatch, error):
    """Mark invalid or conflicting events as permanently failed."""
    async def fake_process(data):
        raise error

    monkeypatch.setattr(
        bullmq_worker, "process_queue_event", fake_process
    )

    with pytest.raises(UnrecoverableError):
        await bullmq_worker.process_bullmq_job(job, "token")


@pytest.mark.asyncio
async def test_transient_error_propagates_for_retry(job, monkeypatch):
    """Propagate unexpected processing errors so BullMQ can retry."""
    async def fake_process(data):
        raise ConnectionError("MongoDB unavailable")

    monkeypatch.setattr(
        bullmq_worker, "process_queue_event", fake_process
    )

    with pytest.raises(ConnectionError, match="MongoDB unavailable"):
        await bullmq_worker.process_bullmq_job(job, "token")


@pytest.mark.asyncio
async def test_non_object_job_data_is_permanently_rejected(monkeypatch):
    """Reject malformed job payloads without retrying them."""
    job = SimpleNamespace(id="bad-job", data=["not", "an", "object"])

    with pytest.raises(UnrecoverableError):
        await bullmq_worker.process_bullmq_job(job, "token")
