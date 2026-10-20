
"""Run the SpotQ BullMQ consumer as a standalone process."""

import asyncio

from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger, setup_logging
from app.infrastructure.queue.bullmq_worker import create_queue_worker


async def main() -> None:
    """Initialize MongoDB, run the consumer, and clean up resources."""
    setup_logging()
    worker = None

    try:
        await mongo_manager.connect()
        worker = await create_queue_worker()

        logger.info("BullMQ consumer starting")
        await worker.run()
    finally:
        try:
            if worker is not None:
                logger.info("Closing BullMQ consumer")
                await worker.close()
        finally:
            await mongo_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
