"""Asynchronous non-blocking logger for production ETA predictions.

Preserves exact prediction-time feature values, model version, and timestamps
for outcome correlation and drift monitoring (AC1).
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING

from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger

PREDICTION_LOGS_COLLECTION = "prediction_logs"


async def ensure_prediction_log_indexes() -> None:
    """Create indexes for prediction logs to optimize correlation queries."""
    db = MongoManager.get_database()
    if db is None:
        return

    try:
        collection = db[PREDICTION_LOGS_COLLECTION]
        await collection.create_index(
            [("prediction_id", ASCENDING)],
            unique=True,
            name="uq_prediction_id",
        )
        await collection.create_index(
            [("queue_entry_id", ASCENDING)],
            sparse=True,
            name="idx_queue_entry_id",
        )
        await collection.create_index(
            [("timestamp", ASCENDING)],
            name="idx_prediction_timestamp",
        )
        await collection.create_index(
            [("restaurant_id", ASCENDING), ("timestamp", ASCENDING)],
            name="idx_restaurant_timestamp",
        )
    except Exception as exc:
        logger.warning(
            "Could not ensure prediction log indexes",
            error=str(exc),
        )


async def log_prediction(
    prediction_id: str,
    restaurant_id: str,
    model_version: str,
    predicted_wait_minutes: float,
    features: dict[str, float],
    timestamp: datetime,
    queue_entry_id: str | None = None,
    fallback_used: bool = False,
) -> None:
    """Durably log a single prediction record to MongoDB."""
    db = MongoManager.get_database()
    if db is None:
        logger.warning("MongoDB unavailable; prediction logging skipped")
        return

    record: dict[str, Any] = {
        "prediction_id": prediction_id,
        "queue_entry_id": queue_entry_id,
        "restaurant_id": restaurant_id,
        "model_version": model_version,
        "predicted_wait_minutes": float(predicted_wait_minutes),
        "features": {k: float(v) for k, v in features.items()},
        "timestamp": timestamp.astimezone(timezone.utc),
        "fallback_used": fallback_used,
        "logged_at": datetime.now(timezone.utc),
    }

    try:
        await db[PREDICTION_LOGS_COLLECTION].insert_one(record)
    except Exception as exc:
        logger.warning(
            "Failed to persist prediction log",
            prediction_id=prediction_id,
            error=str(exc),
        )


def log_prediction_background(
    prediction_id: str,
    restaurant_id: str,
    model_version: str,
    predicted_wait_minutes: float,
    features: dict[str, float],
    timestamp: datetime,
    queue_entry_id: str | None = None,
    fallback_used: bool = False,
) -> asyncio.Task:
    """Schedule prediction logging as a background task to avoid inference latency."""
    return asyncio.create_task(
        log_prediction(
            prediction_id=prediction_id,
            restaurant_id=restaurant_id,
            model_version=model_version,
            predicted_wait_minutes=predicted_wait_minutes,
            features=features,
            timestamp=timestamp,
            queue_entry_id=queue_entry_id,
            fallback_used=fallback_used,
        )
    )
