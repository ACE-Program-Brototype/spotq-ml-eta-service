from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import xgboost as xgb

from app.infrastructure.ml_models.loader import model_loader


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Array of numerical features for wait-time prediction")


class PredictResponse(BaseModel):
    eta_seconds: float


router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_eta(payload: PredictRequest) -> PredictResponse:
    """Generate real-time wait-time prediction using the XGBoost booster."""
    if not model_loader.booster:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model booster is not loaded",
        )

    try:
        dmatrix = xgb.DMatrix([payload.features])
        predictions = model_loader.booster.predict(dmatrix)
        return PredictResponse(eta_seconds=float(predictions[0]))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failure: {str(exc)}",
        )