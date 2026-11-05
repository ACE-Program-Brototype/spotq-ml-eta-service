"""Configurable retraining trigger manager with lifecycle safety boundary (AC5, AC6).

Evaluates error and drift conditions, controls concurrency, enforces cooldowns,
and submits retraining requests through the SCRUM-1197 pipeline to produce candidates.

CRITICAL SAFETY BOUNDARY (AC6):
Monitoring produces candidates only. It NEVER automatically promotes or activates
a candidate model. SCRUM-1228 operator authorization remains strictly authoritative.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.application.monitoring.drift_detector import FeatureDriftReport
from app.application.monitoring.performance import ModelPerformanceReport
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.loader import model_loader

RETRAINING_HISTORY_COLLECTION = "retraining_history"


class TriggerDecision(BaseModel):
    """Result of evaluating retraining trigger conditions."""

    trigger_id: str
    should_retrain: bool
    trigger_type: str = Field(
        ...,
        description="'performance_degradation', 'feature_drift', 'manual', or 'none'",
    )
    status: str = Field(
        ...,
        description="'accepted', 'rejected_disabled', 'rejected_cooldown', 'rejected_already_running', 'rejected_insufficient_samples', or 'no_trigger_needed'",
    )
    reason: str
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime


class RetrainingExecutionResult(BaseModel):
    """Result of an executed retraining pipeline run."""

    trigger_id: str
    status: str  # 'completed', 'failed', or 'rejected'
    candidate_artifact_id: Optional[str] = None
    candidate_path: Optional[str] = None
    is_eligible_for_promotion: Optional[bool] = None
    error_message: Optional[str] = None
    completed_at: datetime


class RetrainingTriggerManager:
    """Manages retraining trigger decisions, concurrency locks, and execution."""

    def __init__(
        self,
        cooldown_hours: float | None = None,
        enabled_auto_triggers: bool | None = None,
    ) -> None:
        self.cooldown_hours = (
            cooldown_hours
            if cooldown_hours is not None
            else settings.RETRAINING_COOLDOWN_HOURS
        )
        self.enabled_auto_triggers = (
            enabled_auto_triggers
            if enabled_auto_triggers is not None
            else settings.ENABLE_AUTOMATIC_RETRAINING_TRIGGERS
        )
        self._lock = asyncio.Lock()
        self._last_completed_retraining: Optional[datetime] = None

    @property
    def is_training_in_progress(self) -> bool:
        """Return True if a training execution currently holds the lock."""
        return self._lock.locked()

    async def get_last_retraining_time(self) -> Optional[datetime]:
        """Fetch the most recent completed retraining timestamp from memory or DB."""
        if self._last_completed_retraining:
            return self._last_completed_retraining

        db = MongoManager.get_database()
        if db is not None:
            try:
                record = await db[RETRAINING_HISTORY_COLLECTION].find_one(
                    {"status": "completed"},
                    sort=[("completed_at", -1)],
                )
                if record and "completed_at" in record:
                    completed_at = record["completed_at"]
                    if isinstance(completed_at, datetime):
                        self._last_completed_retraining = (
                            completed_at
                            if completed_at.tzinfo
                            else completed_at.replace(tzinfo=timezone.utc)
                        )
                        return self._last_completed_retraining
            except Exception as exc:
                logger.warning(
                    "Could not load last retraining time from database",
                    error=str(exc),
                )

        return None

    async def evaluate_triggers(
        self,
        performance_report: Optional[ModelPerformanceReport] = None,
        drift_report: Optional[FeatureDriftReport] = None,
        is_manual: bool = False,
        operator_id: Optional[str] = None,
        manual_notes: Optional[str] = None,
    ) -> TriggerDecision:
        """Evaluate performance and drift to decide whether retraining is needed."""
        trigger_id = f"trig_{uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc)

        # 1. Manual operator request
        if is_manual:
            reason = (
                f"Manual retraining requested by operator {operator_id or 'unknown'}"
            )
            if manual_notes:
                reason += f": {manual_notes}"
            return TriggerDecision(
                trigger_id=trigger_id,
                should_retrain=True,
                trigger_type="manual",
                status="accepted",
                reason=reason,
                metrics_snapshot={"operator_id": operator_id},
                evaluated_at=now_utc,
            )

        # 2. Check if automated triggers are disabled
        if not self.enabled_auto_triggers:
            return TriggerDecision(
                trigger_id=trigger_id,
                should_retrain=False,
                trigger_type="none",
                status="rejected_disabled",
                reason="Automatic retraining triggers are disabled by configuration",
                evaluated_at=now_utc,
            )

        # 3. Check performance degradation
        if performance_report and performance_report.has_degradation:
            degraded_versions = [
                v
                for v, m in performance_report.version_metrics.items()
                if m.is_degraded
            ]
            reasons = []
            for v in degraded_versions:
                reasons.extend(
                    performance_report.version_metrics[v].degradation_reasons
                )

            return await self._check_cooldown_and_lock(
                trigger_id=trigger_id,
                trigger_type="performance_degradation",
                reason=f"Model performance degradation detected on version(s) {degraded_versions}: {'; '.join(reasons)}",
                metrics_snapshot={
                    "overall_mae": performance_report.overall_mae_minutes,
                    "overall_rmse": performance_report.overall_rmse_minutes,
                    "degraded_versions": degraded_versions,
                },
                now_utc=now_utc,
            )

        # 4. Check significant feature drift
        if drift_report and drift_report.has_significant_drift:
            drifted_features = [
                name
                for name, res in drift_report.feature_results.items()
                if res.is_drifted
            ]
            reasons = [
                f"{name} (PSI={drift_report.feature_results[name].psi_score:.3f})"
                for name in drifted_features
            ]

            return await self._check_cooldown_and_lock(
                trigger_id=trigger_id,
                trigger_type="feature_drift",
                reason=f"Significant feature drift detected on {len(drifted_features)} feature(s): {', '.join(reasons)}",
                metrics_snapshot={
                    "drifted_features": drifted_features,
                    "sample_count": drift_report.sample_count,
                },
                now_utc=now_utc,
            )

        # No trigger conditions met
        return TriggerDecision(
            trigger_id=trigger_id,
            should_retrain=False,
            trigger_type="none",
            status="no_trigger_needed",
            reason="All evaluated model metrics and feature distributions remain within acceptable bounds",
            evaluated_at=now_utc,
        )

    async def _check_cooldown_and_lock(
        self,
        trigger_id: str,
        trigger_type: str,
        reason: str,
        metrics_snapshot: dict[str, Any],
        now_utc: datetime,
    ) -> TriggerDecision:
        """Verify concurrency lock and cooldown before accepting an auto-trigger."""
        if self._lock.locked():
            return TriggerDecision(
                trigger_id=trigger_id,
                should_retrain=False,
                trigger_type=trigger_type,
                status="rejected_already_running",
                reason="A model retraining job is already currently executing",
                metrics_snapshot=metrics_snapshot,
                evaluated_at=now_utc,
            )

        last_time = await self.get_last_retraining_time()
        if last_time is not None:
            cooldown_delta = timedelta(hours=self.cooldown_hours)
            elapsed = now_utc - last_time
            if elapsed < cooldown_delta:
                remaining_mins = (
                    cooldown_delta.total_seconds() - elapsed.total_seconds()
                ) / 60.0
                return TriggerDecision(
                    trigger_id=trigger_id,
                    should_retrain=False,
                    trigger_type=trigger_type,
                    status="rejected_cooldown",
                    reason=f"Retraining rejected: in cooldown period ({remaining_mins:.1f} minutes remaining of {self.cooldown_hours}h cooldown)",
                    metrics_snapshot=metrics_snapshot,
                    evaluated_at=now_utc,
                )

        return TriggerDecision(
            trigger_id=trigger_id,
            should_retrain=True,
            trigger_type=trigger_type,
            status="accepted",
            reason=reason,
            metrics_snapshot=metrics_snapshot,
            evaluated_at=now_utc,
        )

    async def execute_retraining(
        self,
        decision: TriggerDecision,
        dataset_mode: str = "synthetic",
    ) -> RetrainingExecutionResult:
        """Execute SCRUM-1197 training pipeline to produce a candidate model.

        SAFETY BOUNDARY (AC6):
        Produces and validates a candidate model artifact in the candidates directory
        and S3 registry. Does NOT promote the candidate to active production serving.
        """
        if not decision.should_retrain:
            return RetrainingExecutionResult(
                trigger_id=decision.trigger_id,
                status="rejected",
                error_message=f"Retraining was not requested: {decision.reason}",
                completed_at=datetime.now(timezone.utc),
            )

        if self._lock.locked():
            return RetrainingExecutionResult(
                trigger_id=decision.trigger_id,
                status="rejected",
                error_message="Retraining execution aborted: another run is in progress",
                completed_at=datetime.now(timezone.utc),
            )

        async with self._lock:
            start_snapshot = model_loader.get_snapshot()
            active_version_before = getattr(
                start_snapshot, "model_version", "unknown"
            )

            logger.info(
                "Starting automated model retraining execution",
                trigger_id=decision.trigger_id,
                trigger_type=decision.trigger_type,
                reason=decision.reason,
                active_model_version=active_version_before,
            )

            db = MongoManager.get_database()
            if db is not None:
                try:
                    await db[RETRAINING_HISTORY_COLLECTION].insert_one(
                        {
                            "trigger_id": decision.trigger_id,
                            "trigger_type": decision.trigger_type,
                            "trigger_reason": decision.reason,
                            "metrics_snapshot": decision.metrics_snapshot,
                            "status": "in_progress",
                            "started_at": datetime.now(timezone.utc),
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not log retraining start to database",
                        error=str(exc),
                    )

            try:
                # Import training pipeline dynamically
                from scripts.train_eta_model import parse_args, run_training

                training_args = parse_args(
                    [
                        "--dataset-mode",
                        dataset_mode,
                        "--min-sample-size",
                        "30",
                    ]
                )

                # Execute pipeline away from main thread if CPU bound
                report = await run_training(training_args)

                # Update cooldown tracker
                now_utc = datetime.now(timezone.utc)
                self._last_completed_retraining = now_utc

                candidate_path = report.get("candidate_model_path")
                is_eligible = report.get("is_eligible", False)

                # LIFECYCLE SAFETY ASSERTION (AC6):
                # Verify that the active production model has NOT been modified
                after_snapshot = model_loader.get_snapshot()
                active_version_after = getattr(
                    after_snapshot, "model_version", "unknown"
                )
                if active_version_after != active_version_before:
                    logger.critical(
                        "SECURITY VIOLATION: Active production model was mutated during retraining!",
                        before=active_version_before,
                        after=active_version_after,
                    )
                    raise RuntimeError(
                        "Active model was mutated during candidate retraining execution"
                    )

                logger.info(
                    "Automated model retraining completed successfully (candidate produced, NOT promoted)",
                    trigger_id=decision.trigger_id,
                    candidate_path=candidate_path,
                    is_eligible=is_eligible,
                    active_model_version=active_version_after,
                )

                if db is not None:
                    try:
                        await db[RETRAINING_HISTORY_COLLECTION].update_one(
                            {"trigger_id": decision.trigger_id},
                            {
                                "$set": {
                                    "status": "completed",
                                    "candidate_path": str(candidate_path),
                                    "is_eligible": is_eligible,
                                    "completed_at": now_utc,
                                }
                            },
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not update retraining status in database",
                            error=str(exc),
                        )

                return RetrainingExecutionResult(
                    trigger_id=decision.trigger_id,
                    status="completed",
                    candidate_path=str(candidate_path),
                    is_eligible_for_promotion=is_eligible,
                    completed_at=now_utc,
                )

            except Exception as exc:
                now_utc = datetime.now(timezone.utc)
                logger.exception(
                    "Model retraining execution failed",
                    trigger_id=decision.trigger_id,
                    error=str(exc),
                )
                if db is not None:
                    try:
                        await db[RETRAINING_HISTORY_COLLECTION].update_one(
                            {"trigger_id": decision.trigger_id},
                            {
                                "$set": {
                                    "status": "failed",
                                    "error": str(exc),
                                    "failed_at": now_utc,
                                }
                            },
                        )
                    except Exception:
                        pass

                return RetrainingExecutionResult(
                    trigger_id=decision.trigger_id,
                    status="failed",
                    error_message=str(exc),
                    completed_at=now_utc,
                )


retraining_trigger_manager = RetrainingTriggerManager()
