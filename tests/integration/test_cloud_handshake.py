"""Integration tests for live cloud services (skipped by default in CI)."""

import pytest
import motor.motor_asyncio
import redis.asyncio as aioredis
from app.infrastructure.config.settings import settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_cloud_connection():
    """Verify live Redis Cloud connectivity and round-trip."""
    client = aioredis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        username=settings.REDIS_USERNAME,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=3.0,
    )
    try:
        pong = await client.ping()
        assert pong is True
        
        await client.set("spotq:test:key", "active", ex=5)
        val = await client.get("spotq:test:key")
        assert val == "active"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mongodb_atlas_connection():
    """Verify live MongoDB Atlas connectivity and CRUD handshake."""
    client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=3000
    )
    db = client[settings.MONGO_DB_NAME]
    
    # Ping database
    await db.command("ping")
    
    # Simple write/read test
    collection = db["integration_smoke_test"]
    result = await collection.insert_one({"test": True})
    doc = await collection.find_one({"_id": result.inserted_id})
    await collection.delete_one({"_id": result.inserted_id})
    
    client.close()
    assert doc is not None
    assert doc["test"] is True