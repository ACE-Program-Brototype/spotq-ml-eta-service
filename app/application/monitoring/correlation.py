"""Correlates production ETA predictions with eventual completed queue outcomes (AC1).

Links predictions to completed ground truth using queue_entry_id (exact correlation)
or restaurant/timestamp proximity (heuristic fallback), preserving prediction-time
features and model version attribution.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.infrastructure.monitoring.prediction_logger import PREDICTION_LOGS_COLLECTION

QUEUE_EVENTS_COLLECTION = "queue_events"
MAX_PREDICTION_WINDOW_HOURS = 2
HEURISTIC_TIME_TOLERANCE_SECONDS = 600  # 10 minutes


class CorrelatedRecord(BaseModel):
    """A prediction paired with its eventual completed wait outcome."""

    prediction_id: str
    queue_entry_id: str
    restaurant_id: str
    model_version: str
    prediction_timestamp: datetime
    outcome_timestamp: datetime
    predicted_wait_minutes: float
    actual_wait_minutes: float
    features: dict[str, float]
    correlation_method: str = Field(
        ..., description="'exact_queue_entry_id' or 'heuristic_proximity'"
    )


class CorrelationSummary(BaseModel):
    """Summary of prediction-to-outcome correlation evaluation."""

    total_predictions: int
    total_completed_outcomes: int
    correlated_count: int
    uncorrelated_predictions_count: int
    uncorrelated_outcomes_count: int
    records: list[CorrelatedRecord]


def _ensure_utc(dt: Any) -> datetime | None:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def correlate_predictions_with_outcomes(
    start_time: datetime,
    end_time: datetime,
    restaurant_id: str | None = None,
) -> CorrelationSummary:
    """Query MongoDB and link predictions with completed queue events."""
    db = MongoManager.get_database()
    if db is None:
        logger.warning("MongoDB unavailable; cannot correlate predictions")
        return CorrelationSummary(
            total_predictions=0,
            total_completed_outcomes=0,
            correlated_count=0,
            uncorrelated_predictions_count=0,
            uncorrelated_outcomes_count=0,
            records=[],
        )

    start_utc = _ensure_utc(start_time)
    end_utc = _ensure_utc(end_time)

    # 1. Fetch completed queue events in window
    event_filter: dict[str, Any] = {
        "completed_at": {"$gte": start_utc, "$lte": end_utc},
        "is_synthetic": False,
    }
    if restaurant_id:
        event_filter["restaurant_id"] = restaurant_id

    events: list[dict[str, Any]] = []
    async for doc in db[QUEUE_EVENTS_COLLECTION].find(event_filter):
        events.append(doc)

    # 2. Fetch predictions in extended window (predictions happen before completion)
    pred_start_utc = start_utc - timedelta(hours=MAX_PREDICTION_WINDOW_HOURS)
    pred_filter: dict[str, Any] = {
        "timestamp": {"$gte": pred_start_utc, "$lte": end_utc},
    }
    if restaurant_id:
        pred_filter["restaurant_id"] = restaurant_id

    predictions: list[dict[str, Any]] = []
    async for doc in db[PREDICTION_LOGS_COLLECTION].find(pred_filter):
        predictions.append(doc)

    return correlate_records_in_memory(predictions, events)


def correlate_records_in_memory(
    predictions: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> CorrelationSummary:
    """Correlate predictions and events with exact and heuristic matching."""
    # Index predictions by queue_entry_id for O(1) exact lookup
    predictions_by_qid: dict[str, list[dict[str, Any]]] = {}
    unlinked_predictions: list[dict[str, Any]] = []

    for pred in predictions:
        qid = pred.get("queue_entry_id")
        if qid and str(qid).strip():
            predictions_by_qid.setdefault(str(qid).strip(), []).append(pred)
        else:
            unlinked_predictions.append(pred)

    matched_prediction_ids: set[str] = set()
    matched_event_ids: set[str] = set()
    correlated_records: list[CorrelatedRecord] = []

    # Pass 1: Exact matching on queue_entry_id
    for event in events:
        eqid = str(event.get("queue_entry_id", "")).strip()
        if not eqid:
            continue

        matching_preds = predictions_by_qid.get(eqid, [])
        if matching_preds:
            # If multiple predictions exist for the same queue_entry_id, pick the one closest to joined_at
            joined_at = _ensure_utc(event.get("joined_at"))
            best_pred = min(
                matching_preds,
                key=lambda p: abs(
                    (_ensure_utc(p.get("timestamp")) - joined_at).total_seconds()
                )
                if joined_at and _ensure_utc(p.get("timestamp"))
                else 0,
            )

            pid = str(best_pred.get("prediction_id"))
            if pid not in matched_prediction_ids:
                matched_prediction_ids.add(pid)
                matched_event_ids.add(str(event.get("_id", eqid)))

                actual_wait = float(
                    event.get("actual_wait_time_minutes")
                    or (
                        _ensure_utc(event.get("completed_at"))
                        - _ensure_utc(event.get("joined_at"))
                    ).total_seconds()
                    / 60.0
                )

                correlated_records.append(
                    CorrelatedRecord(
                        prediction_id=pid,
                        queue_entry_id=eqid,
                        restaurant_id=str(best_pred.get("restaurant_id")),
                        model_version=str(best_pred.get("model_version", "unknown")),
                        prediction_timestamp=_ensure_utc(best_pred.get("timestamp")),
                        outcome_timestamp=_ensure_utc(event.get("completed_at")),
                        predicted_wait_minutes=float(
                            best_pred.get("predicted_wait_minutes")
                        ),
                        actual_wait_minutes=actual_wait,
                        features=best_pred.get("features", {}),
                        correlation_method="exact_queue_entry_id",
                    )
                )

    # Pass 2: Fallback heuristic matching for unlinked predictions
    for event in events:
        ev_id = str(event.get("_id", event.get("queue_entry_id", "")))
        if ev_id in matched_event_ids:
            continue

        joined_at = _ensure_utc(event.get("joined_at"))
        ev_rest = str(event.get("restaurant_id"))
        ev_party = event.get("party_size")

        for pred in unlinked_predictions:
            pid = str(pred.get("prediction_id"))
            if pid in matched_prediction_ids:
                continue

            if (
                str(pred.get("restaurant_id")) == ev_rest
                and pred.get("features", {}).get("party_size") == ev_party
            ):
                pred_time = _ensure_utc(pred.get("timestamp"))
                if (
                    joined_at
                    and pred_time
                    and abs((pred_time - joined_at).total_seconds())
                    <= HEURISTIC_TIME_TOLERANCE_SECONDS
                ):
                    matched_prediction_ids.add(pid)
                    matched_event_ids.add(ev_id)

                    actual_wait = float(
                        event.get("actual_wait_time_minutes")
                        or (
                            _ensure_utc(event.get("completed_at")) - joined_at
                        ).total_seconds()
                        / 60.0
                    )

                    correlated_records.append(
                        CorrelatedRecord(
                            prediction_id=pid,
                            queue_entry_id=str(event.get("queue_entry_id", "")),
                            restaurant_id=ev_rest,
                            model_version=str(pred.get("model_version", "unknown")),
                            prediction_timestamp=pred_time,
                            outcome_timestamp=_ensure_utc(event.get("completed_at")),
                            predicted_wait_minutes=float(
                                pred.get("predicted_wait_minutes")
                            ),
                            actual_wait_minutes=actual_wait,
                            features=pred.get("features", {}),
                            correlation_method="heuristic_proximity",
                        )
                    )
                    break

    return CorrelationSummary(
        total_predictions=len(predictions),
        total_completed_outcomes=len(events),
        correlated_count=len(correlated_records),
        uncorrelated_predictions_count=len(predictions) - len(matched_prediction_ids),
        uncorrelated_outcomes_count=len(events) - len(matched_event_ids),
        records=correlated_records,
    )
