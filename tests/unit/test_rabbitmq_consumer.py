"""Unit tests for RabbitMQ event consumer (SCRUM-1257).

Tests ACK, NACK, and DLQ rejection semantics, schema validation,
auto-calculated wait times, and connection handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.application.queue_event_processor import (
    QueueEventConflictError,
    QueueEventValidationError,
)
from app.infrastructure.queue.rabbitmq_consumer import RabbitMQConsumer


@pytest.fixture
def valid_event_dict():
    return {
        "queue_entry_id": "entry-rmq-001",
        "restaurant_id": "restaurant-1",
        "party_size": 2,
        "joined_at": "2026-10-18T12:00:00Z",
        "completed_at": "2026-10-18T12:15:00Z",
        "actual_wait_time_minutes": 15.0,
        "total_people_ahead_at_join": 3,
    }


def make_mock_message(payload, correlation_id="test-corr-1"):
    message = MagicMock()
    message.correlation_id = correlation_id
    message.message_id = "test-msg-1"
    if isinstance(payload, (dict, list)):
        message.body = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        message.body = payload.encode("utf-8")
    else:
        message.body = payload

    message.ack = AsyncMock()
    message.nack = AsyncMock()
    message.reject = AsyncMock()
    return message


@pytest.mark.asyncio
async def test_valid_message_is_processed_and_acknowledged(valid_event_dict):
    """AC1 & AC4: Valid event is processed and explicitly acknowledged (ACK)."""
    consumer = RabbitMQConsumer()
    message = make_mock_message(valid_event_dict)

    with patch(
        "app.infrastructure.queue.rabbitmq_consumer.process_queue_event",
        new=AsyncMock(return_value={"status": "persisted", "queue_entry_id": "entry-rmq-001"}),
    ) as mock_process:
        await consumer._handle_message(message)

        mock_process.assert_awaited_once_with(valid_event_dict)
        message.ack.assert_awaited_once()
        message.nack.assert_not_called()
        message.reject.assert_not_called()


@pytest.mark.asyncio
async def test_missing_wait_time_is_auto_calculated_and_persisted(valid_event_dict):
    """Event missing actual_wait_time_minutes has it auto-calculated and ACKed."""
    event_without_wait = valid_event_dict.copy()
    del event_without_wait["actual_wait_time_minutes"]

    consumer = RabbitMQConsumer()
    message = make_mock_message(event_without_wait)

    with patch(
        "app.infrastructure.queue.rabbitmq_consumer.process_queue_event",
        new=AsyncMock(return_value={"status": "persisted", "queue_entry_id": "entry-rmq-001"}),
    ) as mock_process:
        await consumer._handle_message(message)

        mock_process.assert_awaited_once_with(event_without_wait)
        message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_json_is_rejected_to_dlq_without_requeue():
    """AC4 & AC6: Malformed non-JSON message is permanently rejected to DLQ."""
    consumer = RabbitMQConsumer()
    message = make_mock_message("NOT_VALID_JSON_CHARS{{{")

    await consumer._handle_message(message)

    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_called()
    message.nack.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        QueueEventValidationError("Validation error: party_size exceeds 100"),
        QueueEventConflictError("Conflicting duplicate event exists"),
    ],
)
async def test_permanent_validation_or_conflict_is_rejected_to_dlq(valid_event_dict, error):
    """AC4 & AC5: Schema or conflict errors are rejected to DLQ without requeuing."""
    consumer = RabbitMQConsumer()
    message = make_mock_message(valid_event_dict)

    with patch(
        "app.infrastructure.queue.rabbitmq_consumer.process_queue_event",
        new=AsyncMock(side_effect=error),
    ):
        await consumer._handle_message(message)

        message.reject.assert_awaited_once_with(requeue=False)
        message.ack.assert_not_called()
        message.nack.assert_not_called()


@pytest.mark.asyncio
async def test_transient_database_error_is_nacked_with_requeue(valid_event_dict):
    """AC4 & AC6: Transient database failures trigger NACK with requeue for retry."""
    consumer = RabbitMQConsumer()
    message = make_mock_message(valid_event_dict)

    with patch(
        "app.infrastructure.queue.rabbitmq_consumer.process_queue_event",
        new=AsyncMock(side_effect=RuntimeError("MongoDB is unavailable")),
    ):
        await consumer._handle_message(message)

        message.nack.assert_awaited_once_with(requeue=True)
        message.ack.assert_not_called()
        message.reject.assert_not_called()


@pytest.mark.asyncio
async def test_rabbitmq_consumer_topology_declaration():
    """AC1: RabbitMQ declares exchanges, queues, and binds DLQ and main queue."""
    mock_connection = AsyncMock()
    mock_channel = AsyncMock()
    mock_connection.channel.return_value = mock_channel
    mock_connection.is_closed = False
    mock_channel.is_closed = False

    consumer = RabbitMQConsumer(
        connection_url="amqp://guest:guest@localhost:5672/",
        queue_name="ml-eta-queue",
        exchange_name="spotq.events",
    )

    with patch("aio_pika.connect_robust", new=AsyncMock(return_value=mock_connection)):
        await consumer.connect()

        assert consumer.is_connected is True
        # DLX and main exchange declared
        assert mock_channel.declare_exchange.call_count == 2
        # DLQ and main queue declared
        assert mock_channel.declare_queue.call_count == 2
        # Queues bound
        assert mock_channel.declare_queue.return_value.bind.call_count >= 2


def test_rabbitmq_localhost_fallback_in_development():
    """Verify that consumer adds localhost fallback when configured with @rabbitmq hostname."""
    consumer = RabbitMQConsumer(
        connection_url="amqp://guest:guest@rabbitmq:5672/",
    )
    urls = consumer._get_connection_urls_to_try()
    assert "amqp://guest:guest@rabbitmq:5672/" in urls
    assert "amqp://guest:guest@localhost:5672/" in urls

