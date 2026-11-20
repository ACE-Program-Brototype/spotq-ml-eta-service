"""Reproducible MLflow tracking and candidate eligibility validation for SpotQ ETA."""

from dataclasses import dataclass
from typing import Any
import mlflow
from mlflow.entities import RunStatus

from app.infrastructure.logging.logger import logger


@dataclass(frozen=True)
class EligibilityThresholds:
    """Configurable gates for candidate model promotion eligibility."""

    max_mae_minutes: float = 5.0
    max_rmse_minutes: float = 8.0
    require_beats_baseline: bool = True


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of evaluating candidate metrics against promotion criteria."""

    is_eligible: bool
    rejection_reasons: tuple[str, ...]
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_eligible": self.is_eligible,
            "checks": self.checks,
            "rejection_reasons": list(self.rejection_reasons),
        }


def evaluate_candidate_eligibility(
    test_metrics: dict[str, float | None],
    baseline_metrics: dict[str, Any],
    thresholds: EligibilityThresholds,
    dataset_mode: str,
) -> EligibilityResult:
    """Evaluate candidate model performance against configured criteria.

    Args:
        test_metrics: Evaluation metrics (MAE, RMSE, R2) on the test holdout.
        baseline_metrics: Baseline comparison metrics from compare_with_baselines().
        thresholds: Maximum acceptable error thresholds.
        dataset_mode: Mode of the dataset ('real' or 'synthetic').

    Returns:
        EligibilityResult with boolean decision, check statuses, and reasons.
    """
    rejection_reasons: list[str] = []
    checks: dict[str, bool] = {}

    mae = test_metrics.get("mae_minutes")
    rmse = test_metrics.get("rmse_minutes")

    if mae is None or rmse is None:
        return EligibilityResult(
            is_eligible=False,
            rejection_reasons=("Test metrics are incomplete or missing",),
            checks={"metrics_complete": False},
        )

    # 1. MAE Gate
    mae_pass = mae <= thresholds.max_mae_minutes
    checks["mae_within_threshold"] = mae_pass
    if not mae_pass:
        rejection_reasons.append(
            f"Test MAE ({mae:.2f}m) exceeds max threshold ({thresholds.max_mae_minutes:.2f}m)"
        )

    # 2. RMSE Gate
    rmse_pass = rmse <= thresholds.max_rmse_minutes
    checks["rmse_within_threshold"] = rmse_pass
    if not rmse_pass:
        rejection_reasons.append(
            f"Test RMSE ({rmse:.2f}m) exceeds max threshold ({thresholds.max_rmse_minutes:.2f}m)"
        )

    # 3. Baseline Comparison Gate (must outperform historical average baseline)
    metrics_by_model = baseline_metrics.get("metrics", {})
    hist_avg = metrics_by_model.get("historical_average", {})
    hist_avg_mae = hist_avg.get("mae_minutes")

    if hist_avg_mae is not None and hist_avg_mae > 0:
        beats_hist = mae < hist_avg_mae
        checks["beats_historical_average"] = beats_hist
        if thresholds.require_beats_baseline and not beats_hist:
            rejection_reasons.append(
                f"Candidate MAE ({mae:.2f}m) does not outperform historical average ({hist_avg_mae:.2f}m)"
            )
    else:
        checks["beats_historical_average"] = True

    # 4. Provenance Safeguard: Synthetic data can never claim production readiness
    is_real = dataset_mode == "real"
    checks["real_data_provenance"] = is_real
    if not is_real:
        rejection_reasons.append(
            "Synthetic data validates pipeline mechanics only; not eligible for production promotion"
        )

    is_eligible = len(rejection_reasons) == 0
    return EligibilityResult(
        is_eligible=is_eligible,
        rejection_reasons=tuple(rejection_reasons),
        checks=checks,
    )


class MLflowTracker:
    """Context manager for structured, fault-tolerant MLflow tracking."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.active_run = None

    def __enter__(self) -> "MLflowTracker":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.active_run = mlflow.start_run(run_name=self.run_name)
        logger.info(
            "MLflow run started",
            run_id=self.active_run.info.run_id,
            experiment=self.experiment_name,
            tracking_uri=self.tracking_uri,
        )
        return self

    @property
    def run_id(self) -> str:
        if self.active_run is None:
            raise RuntimeError("MLflow run is not active")
        return self.active_run.info.run_id

    def log_training_run(
        self,
        *,
        params: dict[str, Any],
        metrics: dict[str, float | None],
        tags: dict[str, str],
        artifact_paths: list[str],
    ) -> None:
        """Log parameters, metrics, tags, and candidate artifacts to the active run."""
        # Clean params for MLflow logging
        safe_params = {
            key: str(val) if isinstance(val, (list, tuple, dict)) else val
            for key, val in params.items()
        }
        mlflow.log_params(safe_params)

        # Log numeric metrics
        for key, val in metrics.items():
            if val is not None and isinstance(val, (int, float)):
                mlflow.log_metric(key, float(val))

        # Log tags
        mlflow.set_tags(tags)

        # Log artifacts
        for artifact_path in artifact_paths:
            mlflow.log_artifact(artifact_path)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            logger.error(
                "Training aborted with error; recording MLflow run as FAILED",
                error_type=exc_type.__name__,
                error=str(exc_val),
            )
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))
        else:
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FINISHED))

