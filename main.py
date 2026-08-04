"""Application entry point for spotq-eta-service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger, setup_logging
from app.presentation.health.router import router as health_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI application."""
    logger.info(
        "Service starting",
        environment=settings.ENVIRONMENT,
        port=settings.PORT,
    )
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="SpotQ ETA Microservice",
    description="Sub-50ms Wait-Time Prediction Engine",
    version="1.0.0",
    lifespan=lifespan,
)

"""Application entry point for spotq-eta-service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.infrastructure.cache.redis_client import redis_manager
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger, setup_logging
from app.infrastructure.ml_models.loader import model_loader
from app.presentation.health.router import router as health_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI application."""
    logger.info(
        "Service starting",
        environment=settings.ENVIRONMENT,
        port=settings.PORT,
    )
    await redis_manager.connect()
    await mongo_manager.connect()
    model_loader.load_model()

    yield

    logger.info("Service shutting down")
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


@app.get("/")
def root():
    return {
        "service": "spotq-eta-service",
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }