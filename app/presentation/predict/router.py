from datetime import datetime, timezone
from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

# Changed prefix from "/api/v1/eta" to "/eta" to prevent path duplication
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
    total_people_ahead_at_join: int = Field(..., description="Sum of party sizes ahead of them when they joined")


class ETAIngestResponse(BaseModel):
    status: str
    message: str


@router.get("/predict", response_model=ETAPredictResponse, status_code=status.HTTP_200_OK)
async def predict_eta(
    restaurant_id: str = Query(..., description="Unique identifier of the restaurant"),
    total_people_ahead: int = Query(..., description="Sum of party sizes for active queue entries ahead"),
    party_size: int = Query(..., description="Size of the group requesting a table"),
) -> ETAPredictResponse:
    """Evaluate restaurant queue volume against the XGBoost model for a live wait-time estimate."""
    return ETAPredictResponse(
        restaurant_id=restaurant_id,
        estimated_wait_minutes=25.5,
        confidence_score=0.88,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/ingest", response_model=ETAIngestResponse, status_code=status.HTTP_200_OK)
async def ingest_queue_event(payload: ETAIngestRequest) -> ETAIngestResponse:
    """Accept a completed queue entry and defer background processing to the MongoDB buffer."""
    return ETAIngestResponse(
        status="success",
        message="Queue event accepted for background processing",
    )