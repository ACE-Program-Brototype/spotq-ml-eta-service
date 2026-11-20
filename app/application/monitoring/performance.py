"""Outcome-based model performance evaluator with model-version attribution (AC4).

Calculates real-world error metrics (MAE and RMSE) using valid correlated
completed queue outcomes, associating results with specific model versions.
"""

from datetime import datetime, timezone
import math

import numpy as np
from pydantic import BaseModel, Field

from app.application.monitoring.correlation import CorrelatedRecord
from app.infrastructure.logging.logger import logger

DEFAULT_MIN_SAMPLE_SIZE = 30
DEFAULT_MAX_ACCEPTABLE_MAE = 5.0
DEFAULT_MAX_ACCEPTABLE_RMSE = 8.0


class SingleVersionPerformance(BaseModel):
    """Performance metrics attributed to a single model version."""

    model_version: str
    sample_count: int
    mae_minutes: float | None = None
    rmse_minutes: float | None = None
    max_absolute_error_minutes: float | None = None
    over_prediction_pct: float | None = None
    under_prediction_pct: float | None = None
    is_sufficient_samples: bool
    is_degraded: bool = Field(
        default=False,
        description="True if MAE or RMSE exceeds thresholds with sufficient samples",
    )
    degradation_reasons: list[str] = Field(default_factory=list)


class ModelPerformanceReport(BaseModel):
    """Aggregated performance monitoring report across model versions."""

    evaluation_timestamp: datetime
    total_evaluated_samples: int
    version_metrics: dict[str, SingleVersionPerformance]
    overall_mae_minutes: float | None = None
    overall_rmse_minutes: float | None = None
    has_degradation: bool = False


class PerformanceEvaluator:
    """Evaluates production model accuracy from correlated prediction-outcome records."""

    def __init__(
        self,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        max_acceptable_mae: float = DEFAULT_MAX_ACCEPTABLE_MAE,
        max_acceptable_rmse: float = DEFAULT_MAX_ACCEPTABLE_RMSE,
    ) -> None:
        self.min_sample_size = min_sample_size
        self.max_acceptable_mae = max_acceptable_mae
        self.max_acceptable_rmse = max_acceptable_rmse

    def evaluate_performance(
        self,
        valid_records: list[CorrelatedRecord],
    ) -> ModelPerformanceReport:
        """Calculate MAE and RMSE grouped strictly by model version."""
        now_utc = datetime.now(timezone.utc)
        total_samples = len(valid_records)

        # Group records by model_version to prevent cross-version contamination
        records_by_version: dict[str, list[CorrelatedRecord]] = {}
        for record in valid_records:
            version = record.model_version or "unknown"
            records_by_version.setdefault(version, []).append(record)

        version_metrics: dict[str, SingleVersionPerformance] = {}
        all_actuals: list[float] = []
        all_predictions: list[float] = []
        has_degradation = False

        for version, records in records_by_version.items():
            count = len(records)
            if count < self.min_sample_size:
                logger.info(
                    "Sample size insufficient for reliable performance evaluation",
                    model_version=version,
                    sample_count=count,
                    min_sample_size=self.min_sample_size,
                )
                version_metrics[version] = SingleVersionPerformance(
                    model_version=version,
                    sample_count=count,
                    is_sufficient_samples=False,
                    is_degraded=False,
                )
                continue

            actuals = np.asarray(
                [r.actual_wait_minutes for r in records], dtype=np.float64
            )
            predictions = np.asarray(
                [r.predicted_wait_minutes for r in records], dtype=np.float64
            )

            all_actuals.extend(actuals.tolist())
            all_predictions.extend(predictions.tolist())

            errors = predictions - actuals
            abs_errors = np.abs(errors)

            mae = float(np.mean(abs_errors))
            rmse = float(math.sqrt(np.mean(errors**2)))
            max_abs_err = float(np.max(abs_errors))

            over_pct = float(np.mean(errors > 0) * 100.0)
            under_pct = float(np.mean(errors < 0) * 100.0)

            degradation_reasons = []
            if mae > self.max_acceptable_mae:
                degradation_reasons.append(
                    f"MAE ({mae:.2f}m) exceeds threshold ({self.max_acceptable_mae:.2f}m)"
                )
            if rmse > self.max_acceptable_rmse:
                degradation_reasons.append(
                    f"RMSE ({rmse:.2f}m) exceeds threshold ({self.max_acceptable_rmse:.2f}m)"
                )

            is_degraded = len(degradation_reasons) > 0
            if is_degraded:
                has_degradation = True

            version_metrics[version] = SingleVersionPerformance(
                model_version=version,
                sample_count=count,
                mae_minutes=round(mae, 2),
                rmse_minutes=round(rmse, 2),
                max_absolute_error_minutes=round(max_abs_err, 2),
                over_prediction_pct=round(over_pct, 1),
                under_prediction_pct=round(under_pct, 1),
                is_sufficient_samples=True,
                is_degraded=is_degraded,
                degradation_reasons=degradation_reasons,
            )

        overall_mae = (
            round(float(np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))), 2)
            if all_actuals
            else None
        )
        overall_rmse = (
            round(
                float(math.sqrt(np.mean((np.array(all_predictions) - np.array(all_actuals)) ** 2))),
                2,
            )
            if all_actuals
            else None
        )

        logger.info(
            "Model performance evaluation completed",
            total_samples=total_samples,
            versions_evaluated=list(version_metrics.keys()),
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            has_degradation=has_degradation,
        )

        return ModelPerformanceReport(
            evaluation_timestamp=now_utc,
            total_evaluated_samples=total_samples,
            version_metrics=version_metrics,
            overall_mae_minutes=overall_mae,
            overall_rmse_minutes=overall_rmse,
            has_degradation=has_degradation,
        )
