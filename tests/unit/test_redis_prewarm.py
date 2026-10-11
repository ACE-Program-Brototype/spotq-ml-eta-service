import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from scripts import prewarm_redis_history as prewarm


def make_record(
    event_id="e1",
    restaurant_id="mock-restaurant-001",
    joined_at=None,
    wait=20.0,
    synthetic=True,
):
    joined_at = joined_at or datetime(
        2026, 10, 5, 12, tzinfo=timezone.utc
    )
    return {
        "queue_entry_id": event_id,
        "restaurant_id": restaurant_id,
        "joined_at": joined_at.isoformat(),
        "completed_at": (joined_at + timedelta(minutes=wait)).isoformat(),
        "actual_wait_time_minutes": wait,
        "is_synthetic": synthetic,
    }


def test_aggregation_uses_restaurant_local_weekday_and_hour():
    records = [
        make_record(wait=10),
        make_record(
            event_id="e2",
            joined_at=datetime(2026, 10, 5, 12, 20, tzinfo=timezone.utc),
            wait=30,
        ),
    ]

    buckets, rejected = prewarm.aggregate_records(
        records, {"mock-restaurant-001": "Asia/Kolkata"}
    )

    # 12:00 UTC is 17:30 in India: Monday, hour 17.
    bucket = buckets[("mock-restaurant-001", 0, 17)]
    assert bucket["total"] == 40
    assert bucket["count"] == 2
    assert bucket["synthetic"] == 2
    assert rejected == 0


def test_invalid_wait_and_timestamp_order_are_rejected():
    records = [
        make_record(event_id="bad-wait", wait=121),
        {
            **make_record(event_id="bad-order"),
            "completed_at": "2026-10-05T11:00:00Z",
        },
    ]

    buckets, rejected = prewarm.aggregate_records(
        records, {"mock-restaurant-001": "UTC"}
    )

    assert buckets == {}
    assert rejected == 2


def test_missing_timezone_is_rejected():
    buckets, rejected = prewarm.aggregate_records(
        [make_record()], {}
    )

    assert buckets == {}
    assert rejected == 1


def test_fallback_resolver_uses_hierarchy():
    buckets = {
        ("r1", 0, 10): {
            "total": 20.0, "count": 2, "synthetic": 2, "live": 0
        },
        ("r1", 0, 11): {
            "total": 40.0, "count": 2, "synthetic": 2, "live": 0
        },
        ("r1", 1, 10): {
            "total": 60.0, "count": 2, "synthetic": 2, "live": 0
        },
        ("r2", 0, 10): {
            "total": 100.0, "count": 2, "synthetic": 2, "live": 0
        },
    }

    assert prewarm.resolve_historical_average(
        buckets, "r1", 0, 10
    ) == (10.0, "exact")

    assert prewarm.resolve_historical_average(
        buckets, "r1", 0, 12
    ) == (15.0, "restaurant_weekday")

    assert prewarm.resolve_historical_average(
        buckets, "r1", 2, 12
    ) == (20.0, "restaurant")

    assert prewarm.resolve_historical_average(
        buckets, "unknown", 2, 12
    ) == (27.5, "global")

    assert prewarm.resolve_historical_average(
        {}, "unknown", 2, 12
    ) == (20.0, "cold_start")


@pytest.mark.asyncio
async def test_redis_connect_retries_then_succeeds(monkeypatch):
    attempts = 0

    async def fake_connect():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporarily unavailable")

    async def no_sleep(_):
        return None

    monkeypatch.setattr(prewarm.redis_manager, "connect", fake_connect)
    monkeypatch.setattr(prewarm.asyncio, "sleep", no_sleep)

    await prewarm.connect_redis_with_retries()

    assert attempts == 3


@pytest.mark.asyncio
async def test_redis_connect_fails_after_three_attempts(monkeypatch):
    attempts = 0

    async def fake_connect():
        nonlocal attempts
        attempts += 1
        raise ConnectionError("unavailable")

    async def no_sleep(_):
        return None

    monkeypatch.setattr(prewarm.redis_manager, "connect", fake_connect)
    monkeypatch.setattr(prewarm.asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await prewarm.connect_redis_with_retries()

    assert attempts == 3


@pytest.mark.asyncio
async def test_run_writes_exact_and_fallback_redis_keys(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    import json

    joined = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)

    records = [
        make_record(
            event_id="one",
            restaurant_id="mock-restaurant-001",
            joined_at=joined,
            wait=10,
            synthetic=True,
        ),
        make_record(
            event_id="two",
            restaurant_id="mock-restaurant-001",
            joined_at=joined + timedelta(minutes=5),
            wait=30,
            synthetic=True,
        ),
        make_record(
            event_id="three",
            restaurant_id="mock-restaurant-002",
            joined_at=joined,
            wait=50,
            synthetic=False,
        ),
    ]

    class FakeCursor:
        async def to_list(self, length=None):
            return records

    class FakeCollection:
        def find(self, query):
            return FakeCursor()

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    class FakeMongo:
        db = FakeDB()

        async def connect(self):
            return None

        async def disconnect(self):
            return None

    class FakeRedis:
        def __init__(self):
            self.values = {}

        async def set(self, key, value):
            self.values[key] = value

    fake_redis = FakeRedis()

    class FakeRedisManager:
        client = fake_redis

        async def connect(self):
            return None

        async def disconnect(self):
            return None

    monkeypatch.setattr(prewarm, "mongo_manager", FakeMongo())
    monkeypatch.setattr(prewarm, "redis_manager", FakeRedisManager())

    timezone_path = tmp_path / "timezones.json"
    timezone_path.write_text(
        json.dumps({
            "mock-restaurant-001": "Asia/Kolkata",
            "mock-restaurant-002": "Asia/Kolkata",
        }),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    report = await prewarm.run(
        "spotq_production", timezone_path, report_path
    )

    exact_key = "eta:history:mock-restaurant-001:0:17"
    weekday_key = "eta:history:mock-restaurant-001:weekday:0"
    restaurant_key = "eta:history:mock-restaurant-001:all"
    global_key = "eta:history:global"

    assert float(fake_redis.values[exact_key]) == 20.0
    assert float(fake_redis.values[weekday_key]) == pytest.approx(20.0)
    assert float(fake_redis.values[restaurant_key]) == pytest.approx(20.0)
    assert float(fake_redis.values[global_key]) == pytest.approx(30.0)

    assert fake_redis.values[f"{exact_key}:source"] == "synthetic"
    assert fake_redis.values[f"{exact_key}:sample_count"] == "2"
    assert report["status"] == "completed"
    assert report["buckets_written"] > 0
    assert report_path.exists()


@pytest.mark.asyncio
async def test_run_rejects_empty_dataset_before_connecting_redis(
    tmp_path, monkeypatch
):
    import json

    class FakeCursor:
        async def to_list(self, length=None):
            return []

    class FakeCollection:
        def find(self, query):
            return FakeCursor()

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    class FakeMongo:
        db = FakeDB()

        async def connect(self):
            return None

        async def disconnect(self):
            return None

    async def unexpected_redis_connect():
        pytest.fail("Redis must not be connected for an empty dataset")

    monkeypatch.setattr(prewarm, "mongo_manager", FakeMongo())
    monkeypatch.setattr(
        prewarm.redis_manager, "connect", unexpected_redis_connect
    )

    timezone_path = tmp_path / "timezones.json"
    timezone_path.write_text(
        json.dumps({"mock-restaurant-001": "Asia/Kolkata"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Insufficient dataset"):
        await prewarm.run(
            "spotq_production", timezone_path, tmp_path / "report.json"
        )


@pytest.mark.asyncio
async def test_redis_history_reader_uses_fallback_hierarchy():
    from app.infrastructure.ml_models.redis_history import (
        get_historical_average,
    )

    class FakeRedis:
        def __init__(self, values):
            self.values = values
            self.requested = []

        async def get(self, key):
            self.requested.append(key)
            return self.values.get(key)

    redis = FakeRedis({
        "eta:history:r1:0:12": "18.5",
        "eta:history:r1:weekday:0": "21",
        "eta:history:r1:all": "24",
        "eta:history:global": "28",
    })

    assert await get_historical_average(
        "r1", 0, 12, client=redis
    ) == (18.5, "exact")

    assert redis.requested == ["eta:history:r1:0:12"]

    redis.values.pop("eta:history:r1:0:12")
    assert await get_historical_average(
        "r1", 0, 12, client=redis
    ) == (21.0, "restaurant_weekday")

    redis.values.pop("eta:history:r1:weekday:0")
    assert await get_historical_average(
        "r1", 0, 12, client=redis
    ) == (24.0, "restaurant")

    redis.values.pop("eta:history:r1:all")
    assert await get_historical_average(
        "r1", 0, 12, client=redis
    ) == (28.0, "global")

    redis.values.clear()
    assert await get_historical_average(
        "r1", 0, 12, client=redis
    ) == (20.0, "cold_start")


@pytest.mark.asyncio
async def test_redis_history_reader_skips_invalid_values():
    from app.infrastructure.ml_models.redis_history import (
        get_historical_average,
    )

    class FakeRedis:
        async def get(self, key):
            if key == "eta:history:r1:0:12":
                return "nan"
            if key == "eta:history:r1:weekday:0":
                return "not-a-number"
            if key == "eta:history:r1:all":
                return "-5"
            if key == "eta:history:global":
                return "16.25"
            return None

    assert await get_historical_average(
        "r1", 0, 12, client=FakeRedis()
    ) == (16.25, "global")


@pytest.mark.asyncio
async def test_redis_history_reader_uses_cold_start_without_client():
    from app.infrastructure.ml_models.redis_history import (
        get_historical_average,
    )

    assert await get_historical_average(
        "r1", 0, 12, client=None, cold_start_minutes=17
    ) == (17.0, "cold_start")
