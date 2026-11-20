"""RabbitMQ consumer for completed SpotQ queue events (SCRUM-1257).

Replaces legacy BullMQ with an enterprise-grade async RabbitMQ consumer
supporting AMQP 0-9-1, reliable acknowledgements (ACK/NACK/DLQ),
idempotent persistence, and automated connection recovery.
"""

import asyncio
import json
from time import perf_counter
from typing import Optional

import aio_pika
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)
from aiormq.exceptions import AMQPConnectionError, ConnectionClosed

from app.application.queue_event_processor import (
    QueueEventConflictError,
    QueueEventValidationError,
    ensure_queue_event_indexes,
    process_queue_event,
)
from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.observability.metrics import (
    RABBITMQ_HEALTH_GAUGE,
    RABBITMQ_MESSAGE_PROCESSING_DURATION,
    RABBITMQ_MESSAGES_FAILED_TOTAL,
    RABBITMQ_MESSAGES_PROCESSED_TOTAL,
    RABBITMQ_MESSAGES_RECEIVED_TOTAL,
    RABBITMQ_MESSAGES_REJECTED_TOTAL,
)


class RabbitMQConsumer:
    """Asynchronous RabbitMQ consumer for SpotQ queue event streaming."""

    def __init__(
        self,
        connection_url: Optional[str] = None,
        queue_name: Optional[str] = None,
        exchange_name: Optional[str] = None,
        dlx_exchange_name: Optional[str] = None,
        routing_key: Optional[str] = None,
        prefetch_count: Optional[int] = None,
        reconnect_interval: Optional[float] = None,
    ) -> None:
        self.connection_url = connection_url or settings.RABBITMQ_URL
        self.queue_name = queue_name or settings.RABBITMQ_QUEUE_NAME
        self.exchange_name = exchange_name or settings.RABBITMQ_EXCHANGE_NAME
        self.dlx_exchange_name = (
            dlx_exchange_name or settings.RABBITMQ_DLX_EXCHANGE_NAME
        )
        self.routing_key = routing_key or settings.RABBITMQ_ROUTING_KEY
        self.prefetch_count = prefetch_count or settings.RABBITMQ_PREFETCH_COUNT
        self.reconnect_interval = (
            reconnect_interval or settings.RABBITMQ_RECONNECT_INTERVAL_SECONDS
        )

        self.dlq_queue_name = f"{self.queue_name}.dlq"

        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractRobustChannel] = None
        self._queue: Optional[AbstractRobustQueue] = None
        self._consumer_tag: Optional[str] = None
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Return True if connection and channel are open and active."""
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )

    def _get_connection_urls_to_try(self) -> list[str]:
        """Return candidate connection URLs, including localhost fallback in development."""
        urls = [self.connection_url]
        if (
            "rabbitmq" in self.connection_url
            and settings.ENVIRONMENT != EnvironmentType.PRODUCTION
        ):
            fallback_url = self.connection_url.replace("@rabbitmq:", "@localhost:").replace("@rabbitmq/", "@localhost/")
            if fallback_url not in urls:
                urls.append(fallback_url)
        return urls

    async def connect(self) -> None:
        """Connect to RabbitMQ broker and declare topology (Exchanges, Queues, DLQ)."""
        if self._connection and not self._connection.is_closed:
            try:
                await self._connection.close()
            except Exception:
                pass

        urls_to_try = self._get_connection_urls_to_try()
        last_error = None

        for url in urls_to_try:
            try:
                self._connection = await aio_pika.connect_robust(url)
                self._channel = await self._connection.channel()
                await self._channel.set_qos(prefetch_count=self.prefetch_count)
                RABBITMQ_HEALTH_GAUGE.set(1.0)
                logger.info(
                    "Connected to RabbitMQ broker",
                    url=url.split("@")[-1],  # Sanitize credentials from log
                    queue=self.queue_name,
                    prefetch_count=self.prefetch_count,
                )
                break
            except (AMQPConnectionError, ConnectionClosed, OSError) as exc:
                last_error = exc
                RABBITMQ_HEALTH_GAUGE.set(0.0)
                if self._connection and not self._connection.is_closed:
                    await self._connection.close()

        if not self.is_connected:
            raise ConnectionError(
                f"Failed to connect to RabbitMQ broker: {last_error}"
            ) from last_error

        # Setup Topology: DLX & DLQ for poison-pill handling
        dlx = await self._channel.declare_exchange(
            self.dlx_exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dlq = await self._channel.declare_queue(
            self.dlq_queue_name,
            durable=True,
        )
        await dlq.bind(dlx, routing_key=self.dlq_queue_name)

        # Setup Main Exchange & Queue with DLX configuration
        main_exchange = await self._channel.declare_exchange(
            self.exchange_name,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

        queue_arguments = {
            "x-dead-letter-exchange": self.dlx_exchange_name,
            "x-dead-letter-routing-key": self.dlq_queue_name,
        }
        self._queue = await self._channel.declare_queue(
            self.queue_name,
            durable=True,
            arguments=queue_arguments,
        )
        await self._queue.bind(main_exchange, routing_key=self.routing_key)
        # Also bind wildcard if appropriate
        await self._queue.bind(main_exchange, routing_key="queue.event.#")

        logger.info(
            "RabbitMQ topology declared successfully",
            queue=self.queue_name,
            exchange=self.exchange_name,
            routing_key=self.routing_key,
            dlq=self.dlq_queue_name,
        )

    async def _handle_message(self, message: AbstractIncomingMessage) -> None:
        """Process incoming message with strict ACK, NACK, and DLQ rejection semantics."""
        RABBITMQ_MESSAGES_RECEIVED_TOTAL.inc()
        started = perf_counter()
        correlation_id = str(message.correlation_id or message.message_id or "unknown")

        try:
            body = message.body.decode("utf-8")
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Poison pill: malformed payload cannot be processed; reject to DLQ without requeuing
            logger.error(
                "RabbitMQ message rejected permanently (malformed JSON)",
                correlation_id=correlation_id,
                error=str(exc),
            )
            RABBITMQ_MESSAGES_REJECTED_TOTAL.inc()
            await message.reject(requeue=False)
            return

        try:
            result = await process_queue_event(payload)
            # Reliable Acknowledgement: ACK strictly AFTER database persistence
            await message.ack()
            RABBITMQ_MESSAGES_PROCESSED_TOTAL.inc()
            logger.info(
                "RabbitMQ queue event processed successfully",
                correlation_id=correlation_id,
                queue_entry_id=payload.get("queue_entry_id"),
                status=result.get("status"),
            )
        except (QueueEventValidationError, QueueEventConflictError) as exc:
            # Permanent business/schema error: reject to DLQ without requeuing
            logger.error(
                "RabbitMQ message rejected permanently (validation/conflict)",
                correlation_id=correlation_id,
                queue_entry_id=payload.get("queue_entry_id"),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            RABBITMQ_MESSAGES_REJECTED_TOTAL.inc()
            await message.reject(requeue=False)

        except Exception as exc:
            # Transient error (e.g. database disconnect): NACK with requeue for retry
            logger.exception(
                "RabbitMQ message processing failed temporarily; requeuing for retry",
                correlation_id=correlation_id,
                queue_entry_id=payload.get("queue_entry_id"),
                error=str(exc),
            )
            RABBITMQ_MESSAGES_FAILED_TOTAL.inc()
            await message.nack(requeue=True)

        finally:
            duration = perf_counter() - started
            RABBITMQ_MESSAGE_PROCESSING_DURATION.observe(duration)

    async def start(self) -> None:
        """Start listening for queue messages."""
        if not self.is_connected:
            await self.connect()

        if self._queue is None:
            raise RuntimeError("RabbitMQ queue has not been declared")

        self._consumer_tag = await self._queue.consume(self._handle_message)
        self._is_running = True
        logger.info(
            "RabbitMQ consumer started listening",
            queue=self.queue_name,
            consumer_tag=self._consumer_tag,
        )

    async def run(self) -> None:
        """Continuous consumer loop with automatic reconnection on broker disconnect."""
        self._is_running = True

        while self._is_running:
            try:
                await ensure_queue_event_indexes()
                await self.connect()
                await self.start()

                # Keep worker running until cancellation or connection loss
                while self._is_running and self.is_connected:
                    await asyncio.sleep(1.0)

            except (AMQPConnectionError, ConnectionClosed, ConnectionError, OSError) as exc:
                RABBITMQ_HEALTH_GAUGE.set(0.0)
                logger.warning(
                    "RabbitMQ connection lost; attempting reconnection in background",
                    error=str(exc),
                    retry_seconds=self.reconnect_interval,
                )
            except asyncio.CancelledError:
                logger.info("RabbitMQ consumer task cancelled")
                break
            except Exception as exc:
                logger.error(
                    "Unexpected error in RabbitMQ consumer loop",
                    error=str(exc),
                )

            if self._is_running:
                await asyncio.sleep(self.reconnect_interval)

        await self.close()

    async def close(self) -> None:
        """Gracefully stop consumer and close AMQP channels and connection."""
        self._is_running = False
        RABBITMQ_HEALTH_GAUGE.set(0.0)

        if self._channel and not self._channel.is_closed:
            try:
                if self._consumer_tag and self._queue:
                    await self._queue.cancel(self._consumer_tag)
                await self._channel.close()
            except Exception as exc:
                logger.warning("Error closing RabbitMQ channel", error=str(exc))

        if self._connection and not self._connection.is_closed:
            try:
                await self._connection.close()
            except Exception as exc:
                logger.warning("Error closing RabbitMQ connection", error=str(exc))

        logger.info("RabbitMQ consumer gracefully closed")


async def create_rabbitmq_consumer(
    connection_url: Optional[str] = None,
) -> RabbitMQConsumer:
    """Factory creating and initializing RabbitMQ consumer instance."""
    await ensure_queue_event_indexes()
    consumer = RabbitMQConsumer(connection_url=connection_url)
    return consumer
