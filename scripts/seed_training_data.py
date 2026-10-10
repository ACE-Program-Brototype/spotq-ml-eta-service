"""Generate reproducible synthetic SpotQ queue records and seed MongoDB."""

import argparse
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from random import Random
from typing import Any

from pymongo import UpdateOne

from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger

DEFAULT_SEED = 42
DEFAULT_END_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
DEFAULT_COLLECTION = "spotq_production"
RESTAURANT_PREFIX = "mock-restaurant"

# Friday=4, Saturday=5, Sunday=6. Monday=0.
BUSY_DAYS = {4, 5, 6}

RESTAURANT_PROFILES = (
    {"service_factor": 0.78, "capacity": 18},
    {"service_factor": 0.92, "capacity": 24},
    {"service_factor": 1.05, "capacity": 30},
    {"service_factor": 1.18, "capacity": 22},
    {"service_factor": 0.86, "capacity": 28},
)


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and require an explicit timezone."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("end_at must include a timezone, e.g. 2026-10-01T00:00:00Z")
    return parsed.astimezone(timezone.utc)


def _choose_hour(rng: Random) -> int:
    """Sample hours with lunch and dinner demand peaks."""
    weights = []

    for hour in range(24):
        weight = 1.0

        if 12 <= hour < 14:
            weight = 4.5
        elif 19 <= hour < 22:
            weight = 5.0
        elif 11 <= hour < 12 or 14 <= hour < 15:
            weight = 2.0
        elif 18 <= hour < 19 or 22 <= hour < 23:
            weight = 2.2
        elif 0 <= hour < 8:
            weight = 0.15

        weights.append(weight)

    return rng.choices(range(24), weights=weights, k=1)[0]


def generate_synthetic_records(
    count: int = 3000,
    seed: int = DEFAULT_SEED,
    restaurant_count: int = 5,
    days: int = 90,
    end_at: datetime = DEFAULT_END_AT,
) -> list[dict[str, Any]]:
    """Generate deterministic queue events for pipeline development and testing."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if restaurant_count < 1:
        raise ValueError("restaurant_count must be at least 1")
    if days < 1:
        raise ValueError("days must be at least 1")
    if end_at.tzinfo is None:
        raise ValueError("end_at must be timezone-aware")

    rng = Random(seed)
    end_at = end_at.astimezone(timezone.utc)
    start_at = end_at - timedelta(days=days)
    records: list[dict[str, Any]] = []

    for index in range(count):
        restaurant_index = index % restaurant_count
        restaurant_id = f"{RESTAURANT_PREFIX}-{restaurant_index + 1:03d}"
        profile = RESTAURANT_PROFILES[
            restaurant_index % len(RESTAURANT_PROFILES)
        ]

        # Select a date, then a demand-weighted hour and random minute.
        day_offset = rng.randrange(days)
        selected_date = (start_at + timedelta(days=day_offset)).date()
        hour = _choose_hour(rng)
        minute = rng.randrange(60)
        second = rng.randrange(60)

        joined_at = datetime(
            selected_date.year,
            selected_date.month,
            selected_date.day,
            hour,
            minute,
            second,
            tzinfo=timezone.utc,
        )

        # Ensure every generated join time falls inside the requested window.
        if joined_at < start_at:
            joined_at = start_at
        if joined_at >= end_at:
            joined_at = end_at - timedelta(seconds=1)

        is_peak_hour = hour in {12, 13, 19, 20, 21}
        is_busy_day = joined_at.weekday() in BUSY_DAYS

        demand_multiplier = (
            (1.35 if is_peak_hour else 1.0)
            * (1.25 if is_busy_day else 1.0)
        )

        party_size = rng.choices(
            population=[1, 2, 3, 4, 5, 6, 7, 8],
            weights=[12, 28, 15, 25, 7, 6, 4, 3],
            k=1,
        )[0]

        # Queue workload increases nonlinearly under high demand.
        capacity = profile["capacity"]
        base_ahead = rng.randint(0, capacity)
        total_people_ahead = min(
            100,
            int(base_ahead * demand_multiplier + rng.uniform(0, 4)),
        )

        workload = total_people_ahead + party_size
        congestion = max(0.0, workload - capacity * 0.55)

        wait_minutes = (
            2.5
            + profile["service_factor"] * 0.72 * total_people_ahead
            + 0.85 * party_size
            + 0.025 * congestion**2
            + (4.0 if is_peak_hour else 0.0)
            + (2.5 if is_busy_day else 0.0)
            + rng.gauss(0, 2.5)
        )

        # Keep synthetic target values within the initial training bounds.
        wait_minutes = round(min(120.0, max(1.0, wait_minutes)), 2)
        completed_at = joined_at + timedelta(minutes=wait_minutes)

        # Include seed and time window so the same run is idempotent, while
        # intentionally different datasets get distinct stable event IDs.
        stable_key = f"v1|{seed}|{end_at.isoformat()}|{days}|{restaurant_count}|{index}"
        queue_entry_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"spotq-synthetic:{stable_key}")
        )

        records.append(
            {
                "queue_entry_id": queue_entry_id,
                "restaurant_id": restaurant_id,
                "party_size": party_size,
                "joined_at": joined_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "actual_wait_time_minutes": wait_minutes,
                "total_people_ahead_at_join": total_people_ahead,
                "is_synthetic": True,
                "synthetic_seed": seed,
                "synthetic_dataset_version": "v1",
            }
        )

    return records


async def seed_mongodb(
    records: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 500,
) -> None:
    """Idempotently upsert generated records into MongoDB in batches."""
    if not records:
        raise ValueError("Cannot seed an empty dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty")

    await mongo_manager.connect()

    try:
        database = mongo_manager.db
        if database is None:
            raise RuntimeError("MongoDB connection succeeded without a database")

        collection = database[collection_name]
        await collection.create_index("queue_entry_id", unique=True)

        inserted_or_matched = 0
        upserted = 0

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            operations = [
                UpdateOne(
                    {"queue_entry_id": record["queue_entry_id"]},
                    {"$setOnInsert": record},
                    upsert=True,
                )
                for record in batch
            ]

            result = await collection.bulk_write(operations, ordered=False)
            inserted_or_matched += result.matched_count
            upserted += result.upserted_count

        logger.info(
            "Synthetic ETA dataset seeding completed",
            collection=collection_name,
            requested_records=len(records),
            inserted_records=upserted,
            existing_records=inserted_or_matched,
            seed=records[0]["synthetic_seed"],
        )
    finally:
        await mongo_manager.disconnect()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and idempotently seed synthetic SpotQ ETA records."
    )
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--restaurants", type=int, default=5)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--end-at",
        default=DEFAULT_END_AT.isoformat(),
        help="Timezone-aware ISO timestamp; fixed default makes runs reproducible.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION_NAME", DEFAULT_COLLECTION),
    )
    args = parser.parse_args()

    end_at = _parse_datetime(args.end_at)
    records = generate_synthetic_records(
        count=args.count,
        seed=args.seed,
        restaurant_count=args.restaurants,
        days=args.days,
        end_at=end_at,
    )

    logger.info(
        "Generated synthetic ETA records",
        record_count=len(records),
        restaurant_count=args.restaurants,
        seed=args.seed,
        end_at=end_at.isoformat(),
    )

    await seed_mongodb(
        records,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    asyncio.run(main())
