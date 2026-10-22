"""Versioned ETA prediction and legacy queue-event ingestion endpoints."""
import asyncio
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import xgboost as xgb
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES, build_feature_row
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.ml_models.redis_history import get_historical_average
from app.infrastructure.observability.metrics import (
    ETA_CLAMPED_TOTAL,
    ETA_FALLBACK_TOTAL,
    ETA_INFERENCE_FAILURE_TOTAL,
    ETA_INFERENCE_LATENCY,
    ETA_REQUEST_LATENCY,
)
from app.application.schemas.queue_event import ETAIngestRequest

api_v1_router = APIRouter(prefix="/api/v1/eta", tags=["ETA"])

class ETAPredictResponse(BaseModel):
    """Response contract for a real-time ETA prediction."""

    restaurant_id: str
    estimated_wait_minutes: float
    confidence_score: float | None = None
    timestamp: str
    fallback_used: bool = False


class ETAIngestResponse(BaseModel):
    """Response from the existing queue-event ingestion endpoint."""

    status: str
    message: str


@lru_cache(maxsize=1)
def _load_restaurant_timezones() -> dict[str, str]:
    """Load the trusted restaurant-to-timezone mapping once per process."""
    path = Path("config/restaurant_timezones.json")
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Timezone configuration must be a JSON object")
        return {
            restaurant_id: timezone_name
            for restaurant_id, timezone_name in data.items()
            if isinstance(restaurant_id, str)
            and isinstance(timezone_name, str)
            and timezone_name.strip()
        }
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Could not load restaurant timezone configuration", error=str(exc))
        return {}


def _restaurant_timezone(restaurant_id: str) -> str:
    """Resolve a configured timezone or the explicit service default."""
    configured = _load_restaurant_timezones().get(restaurant_id)
    timezone_name = configured or settings.DEFAULT_RESTAURANT_TIMEZONE
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.error(
            "Invalid timezone configuration; using UTC",
            restaurant_id=restaurant_id,
            timezone=timezone_name,
        )
        return "UTC"

    if configured is None:
        logger.warning(
            "Restaurant timezone missing; using configured default",
            restaurant_id=restaurant_id,
            timezone=timezone_name,
        )
    return timezone_name


def _predict_sync(booster: xgb.Booster, values: list[float]) -> float:
    """Run one blocking XGBoost inference away from the event loop."""
    started = perf_counter()
    try:
        matrix = xgb.DMatrix(
            np.asarray([values], dtype=np.float32),
            feature_names=list(FEATURE_NAMES),
        )
        predictions = booster.predict(matrix)
        if len(predictions) != 1:
            raise ValueError("Model must return exactly one prediction")
        prediction = float(predictions[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return prediction
    finally:
        ETA_INFERENCE_LATENCY.observe(perf_counter() - started)


async def predict_eta(
    restaurant_id: str = Query(..., min_length=1),
    total_people_ahead: int = Query(..., ge=0, le=1000),
    party_size: int = Query(..., ge=1, le=100),
) -> ETAPredictResponse:
    """Predict wait using the canonical six-feature contract and local calendar time."""
    request_started = perf_counter()
    try:
        if not restaurant_id.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_restaurant_id",
                    "message": "restaurant_id cannot be blank",
                },
            )

        model_snapshot = model_loader.get_snapshot()
        booster = model_snapshot.booster

        if (
            booster is None
            or model_snapshot.status not in ("loaded", "loaded_synthetic")
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "eta_model_unavailable",
                    "message": "A compatible ETA model is not loaded",
                },
            )

        timestamp = datetime.now(timezone.utc)
        restaurant_timezone = _restaurant_timezone(restaurant_id)
        local_time = timestamp.astimezone(ZoneInfo(restaurant_timezone))

        historical_average, history_source = await get_historical_average(
            restaurant_id,
            local_time.weekday(),
            local_time.hour,
            cold_start_minutes=settings.COLD_START_AVERAGE_WAIT_MINUTES,
            timeout_seconds=settings.REDIS_LOOKUP_TIMEOUT_SECONDS,
        )

        feature_row = build_feature_row(
            {
                "party_size": party_size,
                "total_people_ahead_at_join": total_people_ahead,
                "joined_at": timestamp.isoformat(),
            },
            historical_average,
            restaurant_timezone=restaurant_timezone,
        )
        if tuple(feature_row) != FEATURE_NAMES:
            raise RuntimeError("Generated feature order does not match model contract")

        raw_prediction = await asyncio.to_thread(
            _predict_sync,
            booster,
            [float(feature_row[name]) for name in FEATURE_NAMES],
        )

        minimum = settings.ETA_MIN_WAIT_MINUTES
        maximum = settings.ETA_MAX_WAIT_MINUTES
        if minimum > maximum:
            raise RuntimeError("ETA minimum guardrail exceeds maximum guardrail")

        estimate = min(maximum, max(minimum, raw_prediction))
        if estimate != raw_prediction:
            ETA_CLAMPED_TOTAL.inc()
            logger.warning(
                "ETA prediction clamped to configured guardrails",
                restaurant_id=restaurant_id,
                raw_prediction=raw_prediction,
                estimate=estimate,
            )

        fallback_used = history_source != "exact"
        if fallback_used:
            ETA_FALLBACK_TOTAL.inc()
            logger.info(
                "ETA prediction used fallback history",
                restaurant_id=restaurant_id,
                history_source=history_source,
            )

        return ETAPredictResponse(
            restaurant_id=restaurant_id,
            estimated_wait_minutes=float(estimate),
            confidence_score=None,
            timestamp=timestamp.isoformat(),
            fallback_used=fallback_used,
        )
    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        ETA_INFERENCE_FAILURE_TOTAL.inc()
        logger.error(
            "ETA prediction rejected invalid model or feature data",
            restaurant_id=restaurant_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "eta_prediction_invalid_output",
                "message": "ETA prediction could not be produced",
            },
        ) from exc
    except Exception as exc:
        ETA_INFERENCE_FAILURE_TOTAL.inc()
        logger.error(
            "ETA prediction failed",
            restaurant_id=restaurant_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "eta_prediction_failed",
                "message": "ETA prediction failed",
            },
        ) from exc
    finally:
        ETA_REQUEST_LATENCY.observe(perf_counter() - request_started)


api_v1_router.add_api_route(
    "/predict", predict_eta, methods=["GET"], response_model=ETAPredictResponse
)


@api_v1_router.post(
    "/ingest",
    response_model=ETAIngestResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_queue_event(payload: ETAIngestRequest) -> ETAIngestResponse:
    """Persist a completed queue event to the existing MongoDB ingestion buffer."""
    db = MongoManager.get_database()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable for ingestion",
        )

    try:
        event = payload.model_dump()
        event["is_synthetic"] = False

        await db.queue_events.insert_one(event)
        return ETAIngestResponse(
            status="success",
            message="Queue event durably persisted for background processing",
        )
    except Exception as exc:
        logger.error("Failed to persist queue event", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist queue event",
        ) from exc
