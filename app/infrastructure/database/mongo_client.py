"""Async MongoDB client manager for queue ingestion buffer."""

import time
from motor.motor_asyncio import AsyncIOMotorClient
from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger


class MongoManager:
    """Manages connection lifecycle and health diagnostics for MongoDB Ingestion Buffer."""

    def __init__(self):
        self.client: AsyncIOMotorClient | None = None
        self.db = None

    async def connect(self) -> None:
        """Initialize MongoDB client."""
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGO_URI, serverSelectionTimeoutMS=2000
            )
            self.db = self.client[settings.MONGO_DB_NAME]
            await self.client.admin.command("ping")
            logger.info(
                "Connected to MongoDB Ingestion Buffer",
                database=settings.MONGO_DB_NAME,
            )
        except Exception as exc:
            logger.warning("Unable to connect to MongoDB Ingestion Buffer", error=str(exc))

    async def disconnect(self) -> None:
        """Close MongoDB connection gracefully."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection pool")

    async def ping(self) -> tuple[bool, float]:
        """Ping MongoDB and return status with round-trip latency in milliseconds."""
        if not self.client:
            return False, 0.0
        start_time = time.perf_counter()
        try:
            await self.client.admin.command("ping")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return True, round(latency_ms, 2)
        except Exception as exc:
            logger.warning("MongoDB ping failed", error=str(exc))
            return False, 0.0


mongo_manager = MongoManager()