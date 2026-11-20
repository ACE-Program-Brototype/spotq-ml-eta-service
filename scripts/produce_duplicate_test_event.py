"""Re-enqueue the same event to verify idempotent processing."""

import asyncio

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

    data = {
        "queue_entry_id": "e79ba266-9d1a-4b09-a847-13a235e00012",
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
        print("Duplicate event enqueued")
        print("Job ID:", job.id)
        print("Event ID:", data["queue_entry_id"])
    finally:
        await queue.close()


if __name__ == "__main__":
    asyncio.run(main())
