"""Pre-warm Redis with restaurant-specific historical ETA averages."""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.infrastructure.cache.redis_client import redis_manager
from app.infrastructure.database.mongo_client import mongo_manager

DEFAULT_COLLECTION = "spotq_production"
DEFAULT_TIMEZONES = Path("config/restaurant_timezones.json")
DEFAULT_REPORT = Path(
    "app/infrastructure/ml_models/artifacts/candidates/redis_prewarm_report.json"
)


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("Invalid timestamp type")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include timezone information")

    return parsed.astimezone(timezone.utc)


def aggregate_records(records, restaurant_timezones):
    buckets = defaultdict(
        lambda: {"total": 0.0, "count": 0, "synthetic": 0, "live": 0}
    )
    rejected = 0

    for record in records:
        try:
            restaurant_id = record["restaurant_id"]
            joined_at = parse_timestamp(record["joined_at"])
            completed_at = parse_timestamp(record["completed_at"])
            wait = float(record["actual_wait_time_minutes"])

            if not 0 < wait <= 120 or completed_at <= joined_at:
                rejected += 1
                continue

            if restaurant_id not in restaurant_timezones:
                raise ValueError(f"Missing timezone for {restaurant_id}")

            local_time = joined_at.astimezone(
                ZoneInfo(restaurant_timezones[restaurant_id])
            )
            key = (restaurant_id, local_time.weekday(), local_time.hour)
            bucket = buckets[key]
            bucket["total"] += wait
            bucket["count"] += 1
            bucket["synthetic" if record.get("is_synthetic", False) else "live"] += 1

        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1

    return dict(buckets), rejected


def resolve_historical_average(
    buckets,
    restaurant_id,
    weekday,
    hour,
    cold_start_minutes=20.0,
):
    """Resolve exact, restaurant/day, restaurant, global, then cold-start."""
    def weighted_average(selected):
        total = sum(item["total"] for item in selected)
        count = sum(item["count"] for item in selected)
        return total / count if count else None

    exact = buckets.get((restaurant_id, weekday, hour))
    if exact and exact["count"]:
        return exact["total"] / exact["count"], "exact"

    same_day = [
        value for (rid, day, _), value in buckets.items()
        if rid == restaurant_id and day == weekday
    ]
    average = weighted_average(same_day)
    if average is not None:
        return average, "restaurant_weekday"

    same_restaurant = [
        value for (rid, _, _), value in buckets.items()
        if rid == restaurant_id
    ]
    average = weighted_average(same_restaurant)
    if average is not None:
        return average, "restaurant"

    average = weighted_average(list(buckets.values()))
    if average is not None:
        return average, "global"

    return float(cold_start_minutes), "cold_start"


async def connect_redis_with_retries(max_attempts=3):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            await redis_manager.connect()
            return
        except Exception as exc:
            last_error = exc
            print(f"Redis attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                await asyncio.sleep(attempt)

    raise RuntimeError(
        f"Redis connection failed after {max_attempts} attempts"
    ) from last_error


async def run(collection_name, timezone_path, report_path):
    with timezone_path.open(encoding="utf-8") as handle:
        restaurant_timezones = json.load(handle)

    if not restaurant_timezones:
        raise ValueError("Restaurant timezone configuration is empty")

    await mongo_manager.connect()
    try:
        if mongo_manager.db is None:
            raise RuntimeError("MongoDB database is unavailable")

        records = await mongo_manager.db[collection_name].find(
            {
                "joined_at": {"$exists": True},
                "completed_at": {"$exists": True},
                "actual_wait_time_minutes": {"$gt": 0, "$lte": 120},
            }
        ).to_list(length=None)
    finally:
        await mongo_manager.disconnect()

    if not records:
        raise ValueError("Insufficient dataset records")

    buckets, rejected = aggregate_records(records, restaurant_timezones)
    if not buckets:
        raise ValueError("Insufficient valid historical records")

    await connect_redis_with_retries()
    written = 0
    synthetic_only_buckets = 0

    try:
        client = redis_manager.client
        if client is None:
            raise RuntimeError("Redis client is unavailable")

        async def write_bucket(key, bucket):
            nonlocal written, synthetic_only_buckets

            average = bucket["total"] / bucket["count"]
            await client.set(key, repr(float(average)))

            if bucket["synthetic"] and bucket["live"]:
                source = "mixed"
            elif bucket["synthetic"]:
                source = "synthetic"
                synthetic_only_buckets += 1
            else:
                source = "live"

            await client.set(f"{key}:source", source)
            await client.set(f"{key}:sample_count", str(bucket["count"]))
            written += 1

        # Exact restaurant / weekday / hour buckets.
        for (restaurant_id, weekday, hour), bucket in buckets.items():
            key = f"eta:history:{restaurant_id}:{weekday}:{hour}"
            await write_bucket(key, bucket)

        # Fallback aggregates retain weighted totals and sample counts.
        fallback_buckets = {}

        def accumulate(key, bucket):
            target = fallback_buckets.setdefault(
                key, {"total": 0.0, "count": 0, "synthetic": 0, "live": 0}
            )
            for field in ("total", "count", "synthetic", "live"):
                target[field] += bucket[field]

        for (restaurant_id, weekday, _hour), bucket in buckets.items():
            accumulate(
                f"eta:history:{restaurant_id}:weekday:{weekday}", bucket
            )
            accumulate(f"eta:history:{restaurant_id}:all", bucket)
            accumulate("eta:history:global", bucket)

        for key, bucket in fallback_buckets.items():
            await write_bucket(key, bucket)
    finally:
        await redis_manager.disconnect()

    report = {
        "status": "completed",
        "collection": collection_name,
        "input_records": len(records),
        "rejected_records": rejected,
        "buckets_written": written,
        "synthetic_only_buckets": synthetic_only_buckets,
        "redis_key_format": "eta:history:{restaurant_id}:{day_of_week}:{hour_of_day}",
        "fallback_hierarchy": [
            "exact restaurant/weekday/hour",
            "same restaurant/weekday average",
            "same restaurant across available buckets",
            "global historical average",
            "20-minute cold-start default",
        ],
        "synthetic_data_warning": (
            "Synthetic aggregates are for development and integration testing, "
            "not validated live restaurant history."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION_NAME", DEFAULT_COLLECTION),
    )
    parser.add_argument("--timezones", type=Path, default=DEFAULT_TIMEZONES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    await run(args.collection, args.timezones, args.report)


if __name__ == "__main__":
    asyncio.run(main())
