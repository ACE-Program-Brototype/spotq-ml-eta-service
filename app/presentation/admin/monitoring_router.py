"""Authenticated model monitoring, drift evaluation, and retraining trigger endpoints (AC7, AC8)."""

from typing import Any, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from app.application.monitoring.service import (
    MONITORING_EVALUATIONS_COLLECTION,
    MonitoringCycleReport,
    run_monitoring_cycle,
)
from app.application.monitoring.trigger_manager import (
    RETRAINING_HISTORY_COLLECTION,
    RetrainingExecutionResult,
    RetrainingTriggerManager,
    TriggerDecision,
)
from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.loader import model_loader
from app.presentation.admin.model_router import authorize

router = APIRouter(
    prefix="/api/v1/eta/admin/monitoring",
    tags=["ETA Model Monitoring & Retraining"],
)


class EvaluateRequest(BaseModel):
    """Payload to trigger an on-demand monitoring evaluation cycle."""

    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Hours of historical predictions and completed queue events to evaluate",
    )
    min_samples: Optional[int] = Field(
        default=None,
        ge=5,
        le=10000,
        description="Optional override for minimum valid samples required for statistical reliability",
    )
    restaurant_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional filter for restaurant-specific evaluation",
    )
    auto_trigger: bool = Field(
        default=True,
        description="Whether to automatically dispatch candidate retraining if drift or degradation is confirmed",
    )
    dataset_mode: str = Field(
        default="synthetic",
        description="Dataset mode to use for candidate training ('synthetic' or 'production')",
    )


class ManualRetrainRequest(BaseModel):
    """Payload to request manual retraining of a candidate ETA model."""

    reason: str = Field(
        default="Operator requested manual candidate retraining",
        min_length=3,
        max_length=500,
        description="Audit reason for manual retraining trigger",
    )
    approver_id: str = Field(
        default="admin_operator",
        min_length=1,
        max_length=128,
        description="Identifier of authorizing operator",
    )
    dataset_mode: str = Field(
        default="synthetic",
        description="Dataset mode to use for candidate training ('synthetic' or 'production')",
    )


class ManualRetrainResponse(BaseModel):
    """Response returned upon manual retraining request."""

    decision: TriggerDecision
    execution: Optional[RetrainingExecutionResult] = None


@router.get("/status")
async def get_monitoring_status(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """Retrieve current model monitoring health, latest evaluation results, and retraining history."""
    authorize(x_admin_key)

    snapshot = model_loader.get_snapshot()
    active_version = getattr(snapshot, "model_version", "unknown")

    latest_evaluation: dict[str, Any] | None = None
    latest_retraining: dict[str, Any] | None = None

    db = MongoManager.get_database()
    if db is not None:
        try:
            eval_doc = await db[MONITORING_EVALUATIONS_COLLECTION].find_one(
                sort=[("evaluated_at", -1)]
            )
            if eval_doc and "_id" in eval_doc:
                eval_doc["_id"] = str(eval_doc["_id"])
                latest_evaluation = eval_doc
        except Exception as exc:
            logger.warning("Could not read latest monitoring evaluation", error=str(exc))

        try:
            retrain_doc = await db[RETRAINING_HISTORY_COLLECTION].find_one(
                sort=[("completed_at", -1), ("started_at", -1)]
            )
            if retrain_doc and "_id" in retrain_doc:
                retrain_doc["_id"] = str(retrain_doc["_id"])
                latest_retraining = retrain_doc
        except Exception as exc:
            logger.warning("Could not read latest retraining history", error=str(exc))

    return {
        "status": "operational",
        "active_model_version": active_version,
        "is_retraining_locked": RetrainingTriggerManager._lock.locked(),
        "latest_evaluation": latest_evaluation,
        "latest_retraining": latest_retraining,
    }


@router.post("/evaluate", response_model=MonitoringCycleReport)
async def trigger_monitoring_evaluation(
    request: EvaluateRequest,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> MonitoringCycleReport:
    """Run an on-demand monitoring cycle across prediction correlation, data quality, drift, and performance."""
    authorize(x_admin_key)

    return await run_monitoring_cycle(
        lookback_hours=request.lookback_hours,
        min_sample_size=request.min_samples,
        restaurant_id=request.restaurant_id,
        auto_trigger=request.auto_trigger,
        dataset_mode=request.dataset_mode,
    )


@router.post("/trigger-retrain", response_model=ManualRetrainResponse)
async def trigger_manual_retraining(
    request: ManualRetrainRequest,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> ManualRetrainResponse:
    """Manually request candidate model retraining (AC5, AC6).

    SAFETY BOUNDARY (AC6):
    Retraining produces candidate model artifacts only. It NEVER automatically promotes
    or activates a candidate model. Promotion requires explicit operator authorization
    via /api/v1/eta/admin/promote-candidate.
    """
    authorize(x_admin_key)

    manager = RetrainingTriggerManager()
    decision = await manager.request_manual_retraining(
        reason=request.reason,
        approver_id=request.approver_id,
    )

    execution: RetrainingExecutionResult | None = None
    if decision.should_retrain:
        execution = await manager.execute_retraining(
            decision=decision,
            dataset_mode=request.dataset_mode,
        )

    return ManualRetrainResponse(
        decision=decision,
        execution=execution,
    )
