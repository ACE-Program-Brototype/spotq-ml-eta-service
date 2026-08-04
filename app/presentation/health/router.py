"""Health check diagnostics router for spotq-eta-service."""

from typing import Any, Dict
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient

from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger

router = APIRouter(tags=["Health"])


async def check_redis() -> bool:
    """Verify connectivity to Redis Feature Store."""
    try:
        client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            socket_timeout=1.0,
        )
        await client.ping()
        await client.aclose()
        return True
    except Exception as exc:
        logger.warning("Redis health check failed", error=str(exc))
        return False


async def check_mongodb() -> bool:
    """Verify connectivity to MongoDB Ingestion Buffer."""
    try:
        client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.MONGO_URI, serverSelectionTimeoutMS=1000
        )
        await client.admin.command("ping")
        client.close()
        return True
    except Exception as exc:
        logger.warning("MongoDB health check failed", error=str(exc))
        return False


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Comprehensive service health diagnostics endpoint."""
    redis_healthy = await check_redis()
    mongo_healthy = await check_mongodb()

    # Overall system status
    is_healthy = redis_healthy and mongo_healthy
    status_code = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    payload: Dict[str, Any] = {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "spotq-eta-service",
        "environment": settings.ENVIRONMENT,
        "dependencies": {
            "redis_feature_store": "connected" if redis_healthy else "disconnected",
            "mongodb_buffer": "connected" if mongo_healthy else "disconnected",
        },
    }

    return JSONResponse(status_code=status_code, content=payload)