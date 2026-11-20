"""Enqueue one test event into the isolated BullMQ integration queue."""

import asyncio
from uuid import uuid4

from bullmq import Queue

from app.infrastructure.config.settings import settings


async def main() -> None:
    connection = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
    }
    if settings.REDIS_PASSWORD:
        connection["password"] = settings.REDIS_PASSWORD
        if settings.REDIS_USERNAME:
            connection["username"] = settings.REDIS_USERNAME

    queue = Queue(
        settings.BULLMQ_QUEUE_NAME,
        {
            "connection": connection,
            "prefix": settings.BULLMQ_PREFIX,
        },
    )

    event_id = str(uuid4())
    data = {
        "queue_entry_id": event_id,
        "restaurant_id": "integration-test-restaurant",
        "party_size": 2,
        "joined_at": "2026-10-18T10:00:00Z",
        "completed_at": "2026-10-18T10:10:00Z",
        "actual_wait_time_minutes": 10.0,
        "total_people_ahead_at_join": 3,
    }

    try:
        job = await queue.add(
            "completed-queue-event",
            data,
            {"attempts": 3, "backoff": {"type": "fixed", "delay": 1000}},
        )
        print(f"Queue: {settings.BULLMQ_QUEUE_NAME}")
        print(f"Job ID: {job.id}")
        print(f"Event ID: {event_id}")
    finally:
        await queue.close()


if __name__ == "__main__":
    asyncio.run(main())
