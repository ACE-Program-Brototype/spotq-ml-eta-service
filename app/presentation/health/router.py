"""Liveness, readiness, and dependency diagnostics for SpotQ ETA."""

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


@router.get("/livez")
async def liveness_check() -> dict[str, str]:
    """Report whether the API process is alive."""
    return {"status": "alive", "service": "spotq-eta-service"}


@router.get("/readyz")
async def readiness_check() -> JSONResponse:
    """Check required dependencies; Redis may degrade because fallback exists."""
    mongo_ok, _ = await mongo_manager.ping()
    redis_ok, _ = await redis_manager.ping()
    model_snapshot = model_loader.get_snapshot()
    model_ok = (
        model_snapshot.booster is not None
        and (
            model_snapshot.status == "loaded"
            or (
                model_snapshot.status == "loaded_synthetic"
                and settings.ENVIRONMENT == "development"
                and settings.ALLOW_SYNTHETIC_MODEL
            )
        )
    )

    MONGO_HEALTH_GAUGE.set(int(mongo_ok))
    REDIS_HEALTH_GAUGE.set(int(redis_ok))
    MODEL_LOADED_GAUGE.set(int(model_ok))

    ready = mongo_ok and model_ok
    return JSONResponse(
        status_code=200 if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ready else "not_ready",
            "model": "loaded" if model_ok else "unavailable",
            "mongodb": "connected" if mongo_ok else "disconnected",
            "redis": "connected" if redis_ok else "degraded_fallback_available",
        },
    )


@router.get("/health")
@router.get("/healthz")
async def health_check() -> JSONResponse:
    """Report detailed dependency status without exposing internal paths."""
    redis_ok, redis_latency = await redis_manager.ping()
    mongo_ok, mongo_latency = await mongo_manager.ping()
    model_snapshot = model_loader.get_snapshot()
    model_ok = (
        model_snapshot.booster is not None
        and (
            model_snapshot.status == "loaded"
            or (
                model_snapshot.status == "loaded_synthetic"
                and settings.ENVIRONMENT == "development"
                and settings.ALLOW_SYNTHETIC_MODEL
            )
        )
    )

    REDIS_HEALTH_GAUGE.set(int(redis_ok))
    MONGO_HEALTH_GAUGE.set(int(mongo_ok))
    MODEL_LOADED_GAUGE.set(int(model_ok))

    healthy = redis_ok and mongo_ok and model_ok
    model_details: dict[str, Any] = {
        "status": model_snapshot.status,
        "version": model_snapshot.model_version,
        "artifact_id": model_snapshot.artifact_id,
        "loaded_at": model_snapshot.loaded_at,
    }

    if model_snapshot.failure_reason:
        model_details["reason"] = (
            "Model failed validation or is unavailable"
        )

    payload = {
        "status": "healthy" if healthy else "unhealthy",
        "service": "spotq-eta-service",
        "environment": settings.ENVIRONMENT,
        "dependencies": {
            "redis_feature_store": {
                "status": "connected" if redis_ok else "disconnected",
                "latency_ms": redis_latency,
            },
            "mongodb_buffer": {
                "status": "connected" if mongo_ok else "disconnected",
                "latency_ms": mongo_latency,
            },
            "xgboost_model_artifact": model_details,
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=payload)
