"""Integration tests for live cloud services (skipped by default in CI)."""

import uuid
import pytest
import motor.motor_asyncio
import redis.asyncio as aioredis
from app.infrastructure.config.settings import settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_cloud_connection():
    """Verify live Redis Cloud connectivity and round-trip using isolated keys."""
    client = aioredis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        username=settings.REDIS_USERNAME,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=3.0,
    )
    test_key = f"spotq:test:key:{uuid.uuid4()}"
    try:
        pong = await client.ping()
        assert pong is True
        
        await client.set(test_key, "active", ex=5)
        val = await client.get(test_key)
        assert val == "active"
    finally:
        await client.delete(test_key)
        await client.aclose()


@pytest.mark.asyncio
async def test_mongodb_atlas_connection():
    """Verify live MongoDB Atlas connectivity and CRUD handshake on dedicated test collection."""
    client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=3000
    )
    db = client[settings.MONGO_DB_NAME]
    collection = db[f"integration_smoke_test_{uuid.uuid4()}"]
    
    try:
        await db.command("ping")
        
        result = await collection.insert_one({"test": True})
        doc = await collection.find_one({"_id": result.inserted_id})
        
        assert doc is not None
        assert doc["test"] is True
    finally:
        await collection.drop()
        client.close()