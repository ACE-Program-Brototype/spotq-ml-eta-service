"""Run the SpotQ RabbitMQ consumer as a standalone process (SCRUM-1257)."""

import asyncio
import signal

from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger, setup_logging
from app.infrastructure.queue.rabbitmq_consumer import RabbitMQConsumer


async def main() -> None:
    """Initialize MongoDB, run the RabbitMQ consumer, and clean up resources."""
    setup_logging()
    consumer = None

    stop_event = asyncio.Event()

    def _handle_stop_signal(*_args: object) -> None:
        logger.info("Termination signal received; stopping consumer...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop_signal)
        except NotImplementedError:
            # Signal handling on Windows/non-POSIX event loops
            pass

    try:
        await mongo_manager.connect()
        consumer = RabbitMQConsumer()
        await consumer.start()
        logger.info("RabbitMQ consumer running. Press Ctrl+C to exit.")

        # Wait until stop event is signaled
        await stop_event.wait()
    finally:
        try:
            if consumer is not None:
                logger.info("Closing RabbitMQ consumer...")
                await consumer.close()
        finally:
            await mongo_manager.disconnect()
            logger.info("MongoDB disconnected. Worker exited.")


if __name__ == "__main__":
    asyncio.run(main())
