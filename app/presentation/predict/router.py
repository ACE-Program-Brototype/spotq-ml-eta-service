from datetime import datetime, timezone
import numpy as np
import xgboost as xgb
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.ml_models.loader import model_loader


router = APIRouter(prefix="/eta", tags=["ETA Predictions"])


class ETAPredictResponse(BaseModel):
    restaurant_id: str
    estimated_wait_minutes: float
    confidence_score: float
    timestamp: str


class ETAIngestRequest(BaseModel):
    queue_entry_id: str = Field(..., description="ID of the completed queue entry")
    restaurant_id: str = Field(..., description="ID of the restaurant")
    party_size: int = Field(..., description="Size of the seated group")
    joined_at: str = Field(..., description="Timestamp of joining the queue (ISO format)")
    completed_at: str = Field(..., description="Timestamp of being seated (ISO format)")
    actual_wait_time_minutes: float = Field(..., description="The exact duration waited")
    total_people_ahead_at_join: int = Field(
        ..., description="Sum of party sizes ahead of them when they joined"
    )


class ETAIngestResponse(BaseModel):
    status: str
    message: str


@router.get("/predict", response_model=ETAPredictResponse, status_code=status.HTTP_200_OK)
async def predict_eta(
    restaurant_id: str = Query(..., description="Unique identifier of the restaurant"),
    total_people_ahead: int = Query(
        ..., description="Sum of party sizes for active queue entries ahead"
    ),
    party_size: int = Query(..., description="Size of the group requesting a table"),
) -> ETAPredictResponse:
    """Evaluate restaurant queue volume against the XGBoost model for a live wait-time estimate."""
    if model_loader.booster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ETA machine learning model artifact is not loaded",
        )

    try:
        features = np.array(
            [[float(hash(restaurant_id) % 1000), float(total_people_ahead), float(party_size)]],
            dtype=np.float32,
        )
        dmatrix = xgb.DMatrix(features)
        prediction_array = model_loader.booster.predict(dmatrix)
        estimated_minutes = float(prediction_array[0])

        return ETAPredictResponse(
            restaurant_id=restaurant_id,
            estimated_wait_minutes=max(0.0, estimated_minutes),
            confidence_score=0.95,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction inference failed: {str(exc)}",
        )


@router.post("/ingest", response_model=ETAIngestResponse, status_code=status.HTTP_200_OK)
async def ingest_queue_event(payload: ETAIngestRequest) -> ETAIngestResponse:
    """Accept a completed queue entry and durably persist it to the MongoDB buffer."""
    db = MongoManager.get_database()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable for ingestion",
        )

    try:
        await db.queue_events.insert_one(payload.model_dump())
        return ETAIngestResponse(
            status="success",
            message="Queue event durably persisted for background processing",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist queue event: {str(exc)}",
        )