"""BullMQ consumer for completed SpotQ queue events."""

from typing import Any

from bullmq import UnrecoverableError, Worker

from app.application.queue_event_processor import (
    QueueEventConflictError,
    QueueEventValidationError,
    ensure_queue_event_indexes,
    process_queue_event,
)
from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger




def build_redis_connection() -> dict[str, Any]:
    """Build BullMQ Redis connection options from service settings."""
    connection: dict[str, Any] = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
    }

    if settings.REDIS_PASSWORD:
        connection["password"] = settings.REDIS_PASSWORD
        if settings.REDIS_USERNAME:
            connection["username"] = settings.REDIS_USERNAME

    return connection




async def process_bullmq_job(job: Any, token: str) -> dict[str, str]:
    """Process one BullMQ job and classify permanent versus transient errors."""
    job_id = str(getattr(job, "id", "unknown"))

    try:
        data = job.data
        if not isinstance(data, dict):
            raise QueueEventValidationError(
                "BullMQ job data must be a JSON object"
            )

        result = await process_queue_event(data)
        logger.info(
            "BullMQ queue event processed",
            job_id=job_id,
            result=result["status"],
        )
        return result

    except (QueueEventValidationError, QueueEventConflictError) as exc:
        logger.error(
            "BullMQ job rejected permanently",
            job_id=job_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise UnrecoverableError(str(exc)) from exc

    except Exception:
        logger.exception(
            "BullMQ job processing failed; allowing retry",
            job_id=job_id,
        )
        raise


async def create_queue_worker() -> Worker:
    """Create the worker after MongoDB indexes are ready."""
    await ensure_queue_event_indexes()

    worker = Worker(
        settings.BULLMQ_QUEUE_NAME,
        process_bullmq_job,
        {
            "connection": build_redis_connection(),
            "prefix": settings.BULLMQ_PREFIX,
            "concurrency": settings.BULLMQ_CONCURRENCY,
            "autorun": False,
        },
    )

    logger.info(
        "BullMQ consumer initialized",
        queue=settings.BULLMQ_QUEUE_NAME,
        concurrency=settings.BULLMQ_CONCURRENCY,
    )
    return worker
