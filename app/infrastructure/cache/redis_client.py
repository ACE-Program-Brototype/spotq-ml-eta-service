"""Async Redis client manager for low-latency feature store access."""

import time
import redis.asyncio as aioredis
from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger


class RedisManager:
    """Manages connection pooling and health checks for Redis Feature Store."""

    def __init__(self):
        self.client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self.client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=1.0,
            )
            await self.client.ping()
            logger.info("Connected to Redis Feature Store")
        except Exception as exc:
            logger.warning("Unable to connect to Redis Feature Store", error=str(exc))

    async def disconnect(self) -> None:
        """Close Redis connection pool gracefully."""
        if self.client:
            await self.client.aclose()
            logger.info("Closed Redis connection pool")

    async def ping(self) -> tuple[bool, float]:
        """Ping Redis and return connection status with round-trip latency in milliseconds."""
        if not self.client:
            return False, 0.0
        start_time = time.perf_counter()
        try:
            await self.client.ping()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return True, round(latency_ms, 2)
        except Exception as exc:
            logger.warning("Redis ping failed", error=str(exc))
            return False, 0.0


redis_manager = RedisManager()