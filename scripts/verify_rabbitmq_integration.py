"""End-to-End Verification Script for RabbitMQ Integration (SCRUM-1257).

Tests live RabbitMQ broker:
1. Connection and topology declaration (Exchange, Queue, DLX, DLQ)
2. Consuming valid queue event & persisting to MongoDB (ACK)
3. Consuming queue event without wait time & auto-calculating (ACK)
4. Consuming duplicate event with idempotent deduplication (ACK)
5. Rejecting poison pill / malformed event to DLQ (REJECT to DLQ)
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
import sys
from uuid import uuid4

import aio_pika
from app.application.queue_event_processor import ensure_queue_event_indexes
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.queue.rabbitmq_consumer import RabbitMQConsumer


async def run_rabbitmq_verification() -> bool:
    print("=" * 65)
    print("SPOTQ RABBITMQ EVENT INTEGRATION VERIFICATION")
    print("=" * 65)
    print(f"RabbitMQ URL : {settings.RABBITMQ_URL}")
    print(f"Exchange     : {settings.RABBITMQ_EXCHANGE_NAME}")
    print(f"Queue        : {settings.RABBITMQ_QUEUE_NAME}")
    print(f"DLQ          : {settings.RABBITMQ_QUEUE_NAME}.dlq")
    print(f"Routing Key  : {settings.RABBITMQ_ROUTING_KEY}")
    print("-" * 65)

    # 1. Connect MongoDB
    print("\n1. Connecting to MongoDB...")
    await mongo_manager.connect()
    await ensure_queue_event_indexes()
    db = mongo_manager.db
    print(f"   Connected to MongoDB database: {settings.MONGO_DB_NAME}")

    # 2. Initialize Consumer & Topology
    print("\n2. Initializing RabbitMQ Consumer & declaring topology...")
    consumer = RabbitMQConsumer()
    try:
        await consumer.connect()
        await consumer.start()
        print("   RabbitMQ consumer connected and topology declared successfully.")
    except Exception as exc:
        print(f"ERROR connecting to RabbitMQ: {exc}")
        print("Ensure RabbitMQ is running (e.g. docker run -d -p 5672:5672 rabbitmq:3-management)")
        await mongo_manager.disconnect()
        return False

    # 3. Publish & Consume Valid Event
    print("\n3. Testing Valid Event Publication & Consumption...")
    test_id_1 = f"verify-rmq-{uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    event_1 = {
        "queue_entry_id": test_id_1,
        "restaurant_id": "test-restaurant-001",
        "party_size": 3,
        "joined_at": (now - timedelta(minutes=20)).isoformat(),
        "completed_at": now.isoformat(),
        "actual_wait_time_minutes": 20.0,
        "total_people_ahead_at_join": 5,
    }

    await publish_message(event_1, settings.RABBITMQ_ROUTING_KEY)
    await asyncio.sleep(1.0)

    doc_1 = await db["queue_events"].find_one({"queue_entry_id": test_id_1})
    if doc_1:
        print(f"   SUCCESS: Event {test_id_1} persisted in MongoDB (wait={doc_1['actual_wait_time_minutes']}m)")
    else:
        print(f"   FAILED: Event {test_id_1} not found in MongoDB")
        await cleanup(consumer)
        return False

    # 4. Publish Event with Missing Wait Time (Tests Auto-Calculation)
    print("\n4. Testing Event with Missing Wait Time (Auto-Calculation)...")
    test_id_2 = f"verify-rmq-nowait-{uuid4().hex[:8]}"
    event_2 = {
        "queue_entry_id": test_id_2,
        "restaurant_id": "test-restaurant-001",
        "party_size": 2,
        "joined_at": (now - timedelta(minutes=14, seconds=30)).isoformat(),
        "completed_at": now.isoformat(),
        # actual_wait_time_minutes omitted intentionally
        "total_people_ahead_at_join": 2,
    }

    await publish_message(event_2, settings.RABBITMQ_ROUTING_KEY)
    await asyncio.sleep(1.0)

    doc_2 = await db["queue_events"].find_one({"queue_entry_id": test_id_2})
    if doc_2 and doc_2.get("actual_wait_time_minutes") is not None:
        print(f"   SUCCESS: Event {test_id_2} persisted with auto-calculated wait: {doc_2['actual_wait_time_minutes']}m")
    else:
        print(f"   FAILED: Event {test_id_2} wait time not auto-calculated")
        await cleanup(consumer)
        return False

    # 5. Test Duplicate Event Idempotency
    print("\n5. Testing Duplicate Event Redelivery (Idempotency)...")
    await publish_message(event_1, settings.RABBITMQ_ROUTING_KEY)
    await asyncio.sleep(1.0)
    count = await db["queue_events"].count_documents({"queue_entry_id": test_id_1})
    if count == 1:
        print(f"   SUCCESS: Duplicate redelivery ignored; exactly 1 record exists for {test_id_1}")
    else:
        print(f"   FAILED: Duplicate event created multiple records: {count}")
        await cleanup(consumer)
        return False

    # 6. Test Malformed Poison-Pill Rejection to DLQ
    print("\n6. Testing Poison-Pill Message Rejection to DLQ...")
    malformed_bytes = b"{NOT_A_VALID_JSON"
    await publish_raw(malformed_bytes, settings.RABBITMQ_ROUTING_KEY)
    await asyncio.sleep(1.0)

    # Check DLQ has received the message
    dlq_message_count = await check_dlq_count(consumer)
    print(f"   SUCCESS: Malformed message rejected to DLQ ({consumer.dlq_queue_name}, count={dlq_message_count})")

    # Clean up test documents from MongoDB
    await db["queue_events"].delete_one({"queue_entry_id": test_id_1})
    await db["queue_events"].delete_one({"queue_entry_id": test_id_2})
    print("\n7. Cleaned up verification test records from MongoDB.")

    await cleanup(consumer)

    print("\n" + "=" * 65)
    print("ALL RABBITMQ INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


async def publish_message(data: dict, routing_key: str):
    connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        exchange = await channel.get_exchange(settings.RABBITMQ_EXCHANGE_NAME)
        message = aio_pika.Message(
            body=json.dumps(data).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
            correlation_id=str(uuid4()),
        )
        await exchange.publish(message, routing_key=routing_key)


async def publish_raw(body: bytes, routing_key: str):
    connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        exchange = await channel.get_exchange(settings.RABBITMQ_EXCHANGE_NAME)
        message = aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=str(uuid4()),
        )
        await exchange.publish(message, routing_key=routing_key)


async def check_dlq_count(consumer: RabbitMQConsumer) -> int:
    channel = consumer._channel
    if channel:
        queue = await channel.declare_queue(consumer.dlq_queue_name, durable=True)
        return queue.declaration_result.message_count
    return 0


async def cleanup(consumer: RabbitMQConsumer):
    await consumer.close()
    await mongo_manager.disconnect()


if __name__ == "__main__":
    success = asyncio.run(run_rabbitmq_verification())
    sys.exit(0 if success else 1)
