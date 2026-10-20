
"""Validate and durably persist completed queue events."""

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.application.schemas.queue_event import ETAIngestRequest


COLLECTION_NAME = "queue_events"
WAIT_TIME_TOLERANCE_MINUTES = 0.51
COMPARABLE_FIELDS = (
    "restaurant_id",
    "party_size",
    "joined_at",
    "completed_at",
    "actual_wait_time_minutes",
    "total_people_ahead_at_join",
)


class QueueEventValidationError(ValueError):
    """Raised when a queue event fails schema or business validation."""


class QueueEventConflictError(ValueError):
    """Raised when an existing event ID has conflicting event data."""


def _parse_utc_datetime(value: str) -> datetime:
    """Parse a timezone-aware ISO 8601 timestamp and normalize it to UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QueueEventValidationError(
            "Timestamp must be a valid ISO 8601 value"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QueueEventValidationError(
            "Timestamp must include timezone information"
        )

    return parsed.astimezone(timezone.utc)


def _stored_datetime_to_utc(value: Any) -> datetime | None:
    """Normalize a MongoDB datetime to UTC for reliable comparisons."""
    if not isinstance(value, datetime):
        return None

    if value.tzinfo is None:
        # PyMongo commonly returns naive datetimes representing UTC.
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _matches_existing_event(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> bool:
    """Return whether an existing record represents the same event."""
    for field in COMPARABLE_FIELDS:
        existing_value = existing.get(field)
        incoming_value = incoming.get(field)

        if field in ("joined_at", "completed_at"):
            existing_value = _stored_datetime_to_utc(existing_value)
            incoming_value = _stored_datetime_to_utc(incoming_value)

        if existing_value != incoming_value:
            return False

    return True


async def ensure_queue_event_indexes() -> None:
    """Create the unique event-ID index before starting consumption."""
    db = MongoManager.get_database()
    if db is None:
        raise RuntimeError("MongoDB is unavailable")

    await db[COLLECTION_NAME].create_index(
        [("queue_entry_id", ASCENDING)],
        unique=True,
        name="uq_queue_entry_id",
    )


async def process_queue_event(data: dict[str, Any]) -> dict[str, str]:
    """Validate and persist one event with idempotent duplicate handling.

    Database failures propagate so the queue worker can retry them.
    Identical duplicates return success; conflicting duplicates raise
    QueueEventConflictError and are not silently overwritten.
    """
    try:
        payload = ETAIngestRequest.model_validate(data)
    except ValidationError as exc:
        raise QueueEventValidationError(
            "Queue event failed schema validation"
        ) from exc

    joined_at = _parse_utc_datetime(payload.joined_at)
    completed_at = _parse_utc_datetime(payload.completed_at)

    elapsed_minutes = (completed_at - joined_at).total_seconds() / 60
    if abs(elapsed_minutes - payload.actual_wait_time_minutes) > (
        WAIT_TIME_TOLERANCE_MINUTES
    ):
        raise QueueEventValidationError(
            "Reported wait time does not match the event timestamps"
        )

    db = MongoManager.get_database()
    if db is None:
        raise RuntimeError("MongoDB is unavailable")

    event = payload.model_dump()
    event["joined_at"] = joined_at
    event["completed_at"] = completed_at
    event["is_synthetic"] = False
    event["received_at"] = datetime.now(timezone.utc)

    collection = db[COLLECTION_NAME]

    try:
        await collection.insert_one(event)
    except DuplicateKeyError as exc:
        existing = await collection.find_one(
            {"queue_entry_id": payload.queue_entry_id}
        )
        if existing is None:
            # The duplicate key may relate to another constraint.
            raise

        if not _matches_existing_event(existing, event):
            logger.error(
                "Conflicting duplicate queue event",
                queue_entry_id=payload.queue_entry_id,
            )
            raise QueueEventConflictError(
                "Conflicting event already exists for this queue_entry_id"
            ) from exc

        logger.info(
            "Duplicate queue event ignored",
            queue_entry_id=payload.queue_entry_id,
        )
        return {
            "status": "duplicate",
            "queue_entry_id": payload.queue_entry_id,
        }

    logger.info(
        "Queue event persisted",
        queue_entry_id=payload.queue_entry_id,
    )
    return {
        "status": "persisted",
        "queue_entry_id": payload.queue_entry_id,
    }
