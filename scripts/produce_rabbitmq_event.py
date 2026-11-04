"""Produce test events to RabbitMQ exchange for verification (SCRUM-1257).

Usage:
    # Publish standard valid event:
    uv run python scripts/produce_rabbitmq_event.py

    # Publish event without wait time (tests auto-calculation):
    uv run python scripts/produce_rabbitmq_event.py --omit-wait-time

    # Publish malformed poison-pill event (tests DLQ routing):
    uv run python scripts/produce_rabbitmq_event.py --malformed
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import aio_pika
from app.infrastructure.config.settings import settings


async def publish_test_event(args: argparse.Namespace) -> None:
    event_id = args.entry_id or f"test-rmq-{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    joined_at = (now - timedelta(minutes=15)).isoformat()
    completed_at = now.isoformat()

    if args.malformed:
        payload_bytes = b"{INVALID_JSON_BODY_FOR_DLQ_TEST"
    else:
        data = {
            "queue_entry_id": event_id,
            "restaurant_id": args.restaurant_id,
            "party_size": args.party_size,
            "joined_at": joined_at,
            "completed_at": completed_at,
            "total_people_ahead_at_join": 4,
        }
        if not args.omit_wait_time:
            data["actual_wait_time_minutes"] = 15.0

        payload_bytes = json.dumps(data).encode("utf-8")

    print(f"Connecting to RabbitMQ: {settings.RABBITMQ_URL}...")
    url = settings.RABBITMQ_URL
    try:
        connection = await aio_pika.connect_robust(url)
    except Exception:
        if "@rabbitmq" in url:
            url = url.replace("@rabbitmq:", "@localhost:").replace("@rabbitmq/", "@localhost/")
            connection = await aio_pika.connect_robust(url)
        else:
            raise

    async with connection:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            settings.RABBITMQ_EXCHANGE_NAME,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

        message = aio_pika.Message(
            body=payload_bytes,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
            correlation_id=str(uuid4()),
        )

        await exchange.publish(
            message,
            routing_key=settings.RABBITMQ_ROUTING_KEY,
        )

        print("=" * 60)
        print("RABBITMQ EVENT PUBLISHED")
        print("=" * 60)
        print(f"Exchange    : {settings.RABBITMQ_EXCHANGE_NAME}")
        print(f"Routing Key : {settings.RABBITMQ_ROUTING_KEY}")
        print(f"Event ID    : {event_id}")
        print(f"Payload     : {payload_bytes.decode('utf-8', errors='replace')}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish test RabbitMQ queue events")
    parser.add_argument("--entry-id", default=None, help="Custom queue_entry_id")
    parser.add_argument("--restaurant-id", default="mock-restaurant-001")
    parser.add_argument("--party-size", type=int, default=2)
    parser.add_argument("--omit-wait-time", action="store_true", help="Omit actual_wait_time_minutes to test auto-calculation")
    parser.add_argument("--malformed", action="store_true", help="Send malformed bytes to test DLQ rejection")
    asyncio.run(publish_test_event(parser.parse_args()))

