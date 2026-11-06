"""Monitoring orchestration service coordinating correlation, data quality, drift, and performance (AC7)."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.application.monitoring.correlation import (
    CorrelationSummary,
    correlate_predictions_with_outcomes,
)
from app.application.monitoring.data_quality import (
    DataQualityReport,
    validate_correlated_records,
)
from app.application.monitoring.drift_detector import (
    FeatureDriftDetector,
    FeatureDriftReport,
)
from app.application.monitoring.performance import (
    ModelPerformanceReport,
    PerformanceEvaluator,
)
from app.application.monitoring.trigger_manager import (
    RetrainingExecutionResult,
    RetrainingTriggerManager,
    TriggerDecision,
)
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import MongoManager
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.observability.metrics import (
    MONITORING_EVALUATIONS_TOTAL,
    MONITORING_FEATURE_DRIFT_PSI,
    MONITORING_MAE_GAUGE,
    MONITORING_RETRAINING_TRIGGERS_TOTAL,
    MONITORING_RMSE_GAUGE,
)

MONITORING_EVALUATIONS_COLLECTION = "monitoring_evaluations"


class MonitoringCycleReport(BaseModel):
    """Aggregated output from a single monitoring evaluation cycle."""

    evaluation_id: str
    evaluated_at: datetime
    lookback_hours: int
    active_model_version: Optional[str] = "unknown"
    correlation_summary: CorrelationSummary
    data_quality_report: DataQualityReport
    drift_report: FeatureDriftReport
    performance_report: ModelPerformanceReport
    trigger_decision: Optional[TriggerDecision] = None
    retraining_execution: Optional[RetrainingExecutionResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def update_prometheus_monitoring_metrics(
    drift_report: FeatureDriftReport,
    performance_report: ModelPerformanceReport,
    decision: Optional[TriggerDecision] = None,
) -> None:
    """Record latest monitoring evaluation observations into Prometheus."""
    try:
        MONITORING_EVALUATIONS_TOTAL.labels(status="success").inc()
    except Exception as exc:
        logger.debug("Failed updating evaluation counter", error=str(exc))

    try:
        for feature_name, result in drift_report.feature_results.items():
            MONITORING_FEATURE_DRIFT_PSI.labels(feature_name=feature_name).set(
                result.psi_score
            )
    except Exception as exc:
        logger.debug("Failed updating drift gauges", error=str(exc))

    try:
        for version, vperf in performance_report.version_metrics.items():
            if vperf.mae_minutes is not None:
                MONITORING_MAE_GAUGE.labels(model_version=version).set(
                    vperf.mae_minutes
                )
            if vperf.rmse_minutes is not None:
                MONITORING_RMSE_GAUGE.labels(model_version=version).set(
                    vperf.rmse_minutes
                )
    except Exception as exc:
        logger.debug("Failed updating performance gauges", error=str(exc))

    if decision is not None:
        try:
            MONITORING_RETRAINING_TRIGGERS_TOTAL.labels(
                trigger_type=decision.trigger_type,
                status=decision.status,
            ).inc()
        except Exception as exc:
            logger.debug("Failed updating trigger counter", error=str(exc))


async def run_monitoring_cycle(
    lookback_hours: int = 24,
    min_sample_size: int | None = None,
    restaurant_id: str | None = None,
    auto_trigger: bool = True,
    dataset_mode: str = "synthetic",
    trigger_manager: RetrainingTriggerManager | None = None,
) -> MonitoringCycleReport:
    """Execute end-to-end monitoring pipeline and optionally trigger retraining."""
    evaluation_id = str(uuid4())
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=lookback_hours)
    effective_min_samples = (
        min_sample_size
        if min_sample_size is not None
        else settings.MONITORING_MIN_SAMPLE_SIZE
    )

    active_snapshot = model_loader.get_snapshot()
    active_version = getattr(active_snapshot, "model_version", None) or "unknown"

    logger.info(
        "Starting model monitoring evaluation cycle",
        evaluation_id=evaluation_id,
        lookback_hours=lookback_hours,
        active_model_version=active_version,
    )

    # 1. Correlate predictions with completed outcomes (AC1)
    correlation_summary = await correlate_predictions_with_outcomes(
        start_time=start_time,
        end_time=now_utc,
        restaurant_id=restaurant_id,
    )

    # 2. Filter & validate data quality (AC2)
    quality_report = validate_correlated_records(
        records=correlation_summary.records,
        min_sample_size=effective_min_samples,
    )

    # 3. Detect feature drift across the 6 canonical features (AC3)
    feature_rows = [r.features for r in quality_report.valid_records]
    drift_detector = FeatureDriftDetector(
        min_sample_size=effective_min_samples,
        significant_drift_threshold=settings.MONITORING_PSI_DRIFT_THRESHOLD,
    )
    drift_report = drift_detector.evaluate_drift(feature_rows)

    # 4. Evaluate actual prediction accuracy (AC4)
    perf_evaluator = PerformanceEvaluator(
        min_sample_size=effective_min_samples,
        max_acceptable_mae=settings.MONITORING_MAX_ACCEPTABLE_MAE_MINUTES,
        max_acceptable_rmse=settings.MONITORING_MAX_ACCEPTABLE_RMSE_MINUTES,
    )
    performance_report = perf_evaluator.evaluate_performance(
        quality_report.valid_records
    )

    # 5. Retraining trigger evaluation (AC5, AC6)
    decision: TriggerDecision | None = None
    exec_result: RetrainingExecutionResult | None = None

    if auto_trigger:
        mgr = trigger_manager or RetrainingTriggerManager()
        decision = await mgr.evaluate_trigger(
            performance_report=performance_report,
            drift_report=drift_report,
        )
        if decision.should_retrain:
            logger.info(
                "Retraining condition satisfied; executing candidate training",
                evaluation_id=evaluation_id,
                trigger_type=decision.trigger_type,
            )
            exec_result = await mgr.execute_retraining(
                decision=decision,
                dataset_mode=dataset_mode,
            )

    # 6. Update observability metrics (AC7)
    update_prometheus_monitoring_metrics(
        drift_report=drift_report,
        performance_report=performance_report,
        decision=decision,
    )

    report = MonitoringCycleReport(
        evaluation_id=evaluation_id,
        evaluated_at=now_utc,
        lookback_hours=lookback_hours,
        active_model_version=active_version,
        correlation_summary=correlation_summary,
        data_quality_report=quality_report,
        drift_report=drift_report,
        performance_report=performance_report,
        trigger_decision=decision,
        retraining_execution=exec_result,
        metadata={
            "restaurant_id": restaurant_id,
            "min_sample_size": effective_min_samples,
            "auto_trigger_enabled": auto_trigger,
        },
    )

    # 7. Persist evaluation snapshot in MongoDB
    db = MongoManager.get_database()
    if db is not None:
        try:
            doc = report.model_dump(mode="json")
            await db[MONITORING_EVALUATIONS_COLLECTION].insert_one(doc)
        except Exception as exc:
            logger.warning(
                "Could not persist monitoring evaluation to database",
                error=str(exc),
            )

    logger.info(
        "Monitoring evaluation cycle completed",
        evaluation_id=evaluation_id,
        valid_samples=quality_report.valid_count,
        has_drift=drift_report.has_significant_drift,
        has_degradation=performance_report.has_degradation,
        triggered=decision.should_retrain if decision else False,
    )

    return report

