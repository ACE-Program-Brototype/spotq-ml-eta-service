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

# Expose Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Register routers
app.include_router(health_router)


@app.get("/")
def root():
    return {
        "service": "spotq-eta-service",
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }