"""Health diagnostics for the SpotQ ETA microservice."""

from typing import Any

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
    """Report dependency health and the active model's loading state.

    Returns:
        A JSON response with dependency diagnostics. Returns HTTP 200 only
        when Redis, MongoDB, and a schema-compatible model are all healthy;
        otherwise, returns HTTP 503.
    """
    redis_healthy, redis_latency = await redis_manager.ping()
    mongo_healthy, mongo_latency = await mongo_manager.ping()
    model_healthy = model_loader.is_healthy()

    REDIS_HEALTH_GAUGE.set(1 if redis_healthy else 0)
    MONGO_HEALTH_GAUGE.set(1 if mongo_healthy else 0)
    MODEL_LOADED_GAUGE.set(1 if model_healthy else 0)

    is_healthy = redis_healthy and mongo_healthy and model_healthy
    response_status = (
        status.HTTP_200_OK
        if is_healthy
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    model_details: dict[str, Any] = {
        "status": model_loader.status,
        "path": settings.MODEL_PATH,
    }
    if model_loader.failure_reason is not None:
        model_details["reason"] = model_loader.failure_reason

    payload: dict[str, Any] = {
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
            "xgboost_model_artifact": model_details,
        },
    }

    return JSONResponse(status_code=response_status, content=payload)
