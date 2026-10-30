"""Application entry point for spotq-eta-service."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
from app.infrastructure.queue.bullmq_worker import create_queue_worker
from app.presentation.health.router import router as health_router
from app.presentation.admin.model_router import router as model_admin_router
from app.presentation.predict.router import api_v1_router as eta_v1_router

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

    if not model_loaded:
        logger.critical("Required ETA model failed startup validation")
        try:
            await redis_manager.disconnect()
        except Exception as exc:
            logger.warning("Redis cleanup failed after startup error", error_type=type(exc).__name__)
        try:
            await mongo_manager.disconnect()
        except Exception as exc:
            logger.warning("MongoDB cleanup failed after startup error", error_type=type(exc).__name__)
        REDIS_HEALTH_GAUGE.set(0.0)
        MONGO_HEALTH_GAUGE.set(0.0)
        MODEL_LOADED_GAUGE.set(0.0)
        raise RuntimeError(
            "SpotQ ETA service cannot start: no valid ETA model is available"
        )

    worker = None
    worker_task = None
    try:
        worker = await create_queue_worker()
        worker_task = asyncio.create_task(worker.run())
        logger.info("BullMQ queue consumer started in background")
    except Exception as exc:
        logger.warning("BullMQ worker startup deferred or failed", error=str(exc))

    try:
        yield
    finally:
        logger.info("Service shutting down")
        if worker is not None:
            try:
                await worker.close()
            except Exception as exc:
                logger.warning("BullMQ worker cleanup failed", error=str(exc))
        if worker_task is not None:
            worker_task.cancel()

        REDIS_HEALTH_GAUGE.set(0.0)
        MONGO_HEALTH_GAUGE.set(0.0)
        MODEL_LOADED_GAUGE.set(0.0)
        await redis_manager.disconnect()
        await mongo_manager.disconnect()




async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return structured HTTP 400 responses for invalid request parameters."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": "Request parameters or body are invalid",
            "details": [
                {
                    "location": list(error.get("loc", ())),
                    "message": error.get("msg", "Invalid value"),
                }
                for error in exc.errors()
            ],
        },
    )


app = FastAPI(
    title="SpotQ ETA Microservice",
    description="Sub-50ms Wait-Time Prediction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
app.add_exception_handler(RequestValidationError, validation_error_handler)

app.include_router(health_router)
app.include_router(eta_v1_router)
app.include_router(model_admin_router)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning basic service status."""
    return {
        "service": "spotq-eta-service",
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }