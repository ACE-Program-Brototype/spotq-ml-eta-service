"""Health check diagnostics router for spotq-eta-service."""

from typing import Any, Dict
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.infrastructure.cache.redis_client import redis_manager
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.observability.metrics import (
    MODEL_LOADED_GAUGE,
    MONGO_HEALTH_GAUGE,
    REDIS_HEALTH_GAUGE,
)

router = APIRouter(tags=["Health"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Comprehensive service health diagnostics endpoint."""
    redis_healthy, redis_latency = await redis_manager.ping()
    mongo_healthy, mongo_latency = await mongo_manager.ping()
    model_healthy = model_loader.is_healthy()

    REDIS_HEALTH_GAUGE.set(1 if redis_healthy else 0)
    MONGO_HEALTH_GAUGE.set(1 if mongo_healthy else 0)
    MODEL_LOADED_GAUGE.set(1 if model_healthy else 0)

    # Note: Service reports 200 during local scaffold testing; in strict mode requires all healthy
    is_healthy = redis_healthy and mongo_healthy and model_healthy
    status_code = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    payload: Dict[str, Any] = {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "spotq-eta-service",
        "environment": settings.ENVIRONMENT,
        "dependencies": {
            "redis_feature_store": {
                "status": "connected" if redis_healthy else "disconnected",
                "latency_ms": redis_latency,
            },
            "mongodb_buffer": {
                "status": "connected" if mongo_healthy else "disconnected",
                "latency_ms": mongo_latency,
            },
            "xgboost_model_artifact": {
                "status": "loaded" if model_healthy else "missing",
                "path": settings.MODEL_PATH,
            },
        },
    }

    return JSONResponse(status_code=status_code, content=payload)