
"""Async Redis client manager for low-latency feature store access."""

import time

import redis.asyncio as aioredis

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger


class RedisManager:
    """Manage the Redis connection pool and health checks."""

    def __init__(self) -> None:
        self.client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Initialize the Redis connection pool and verify connectivity."""
        hosts_to_try = [settings.REDIS_HOST]
        if (
            settings.REDIS_HOST not in {"localhost", "127.0.0.1"}
            and settings.ENVIRONMENT != EnvironmentType.PRODUCTION
        ):
            hosts_to_try.append("localhost")

        last_exc: Exception | None = None
        for host in hosts_to_try:
            try:
                redis_options = {
                    "host": host,
                    "port": settings.REDIS_PORT,
                    "decode_responses": True,
                }

                if settings.REDIS_PASSWORD:
                    redis_options["password"] = settings.REDIS_PASSWORD
                    if settings.REDIS_USERNAME:
                        redis_options["username"] = settings.REDIS_USERNAME

                self.client = aioredis.Redis(**redis_options)
                await self.client.ping()

                logger.info(
                    "Connected to Redis feature store",
                    host=host,
                    port=settings.REDIS_PORT,
                )
                return
            except Exception as exc:
                last_exc = exc
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None

        logger.error(
            "Unable to connect to Redis feature store",
            error=str(last_exc),
        )
        if last_exc:
            raise last_exc

    async def disconnect(self) -> None:
        """Close the Redis connection pool gracefully."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            logger.info("Closed Redis connection pool")

    async def ping(self) -> tuple[bool, float]:
        """Return Redis connectivity and round-trip latency in milliseconds."""
        if self.client is None:
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


def get_redis_manager() -> RedisManager:
    """Provide the shared RedisManager instance."""
    return redis_manager
