"""Application entry point for spotq-eta-service."""

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.infrastructure.cache.redis_client import redis_manager
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger, setup_logging
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.observability.metrics import (
    MODEL_LOADED_GAUGE,
    MONGO_HEALTH_GAUGE,
    REDIS_HEALTH_GAUGE,
)
from app.presentation.health.router import router as health_router
from app.presentation.predict.router import router as predict_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI application."""
    logger.info(
        "Service starting",
        environment=settings.ENVIRONMENT,
        port=settings.PORT,
    )

    try:
        await redis_manager.connect()
        REDIS_HEALTH_GAUGE.set(1.0)
    except Exception as exc:
        logger.error("Redis connection failed", error=str(exc))
        REDIS_HEALTH_GAUGE.set(0.0)

    try:
        await mongo_manager.connect()
        MONGO_HEALTH_GAUGE.set(1.0)
    except Exception as exc:
        logger.error("MongoDB connection failed", error=str(exc))
        MONGO_HEALTH_GAUGE.set(0.0)

    model_loaded = model_loader.load_model()
    MODEL_LOADED_GAUGE.set(1.0 if model_loaded else 0.0)

    yield

    logger.info("Service shutting down")
    REDIS_HEALTH_GAUGE.set(0.0)
    MONGO_HEALTH_GAUGE.set(0.0)
    MODEL_LOADED_GAUGE.set(0.0)

    await redis_manager.disconnect()
    await mongo_manager.disconnect()


app = FastAPI(
    title="SpotQ ETA Microservice",
    description="Sub-50ms Wait-Time Prediction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning basic service status."""
    return {
        "service": "spotq-eta-service",
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }