"""Automated ETA Model Training & Retraining Pipeline with MLflow tracking and candidate validation."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBRegressor

from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.data_cleaning import clean_training_records
from app.infrastructure.ml_models.dataset import (
    chronological_train_test_split,
    make_purged_time_series_cv,
)
from app.infrastructure.ml_models.diagnostics import build_model_diagnostics
from app.infrastructure.ml_models.features import (
    FEATURE_NAMES,
    build_training_features,
)
from app.infrastructure.ml_models.tracking import (
    EligibilityThresholds,
    MLflowTracker,
    evaluate_candidate_eligibility,
)


class InsufficientDataError(ValueError):
    """Raised when training dataset has insufficient valid samples."""


class CandidateVerificationError(RuntimeError):
    """Raised when serialized candidate fails serving contract validation."""


def compute_regression_metrics(
    actual: list[float],
    predicted: list[float],
) -> dict[str, float | None]:
    """Calculate regression metrics for actual and predicted wait times.

    Args:
        actual: Observed wait times in minutes.
        predicted: Model-predicted wait times in minutes.

    Returns:
        MAE and RMSE in minutes, and R-squared when at least two
        observations are available.

    Raises:
        ValueError: If the arrays have different lengths, are empty,
            or contain non-finite values.
    """
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal nonzero length")

    if not all(math.isfinite(float(value)) for value in actual + predicted):
        raise ValueError("Metrics require finite values")

    return {
        "mae_minutes": float(mean_absolute_error(actual, predicted)),
        "rmse_minutes": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def compare_with_baselines(
    y_train: list[float],
    y_test: list[float],
    historical_average_predictions: list[float],
    random_forest_predictions: list[float],
    model_predictions: list[float],
) -> dict[str, Any]:
    """Compare baselines and both model families on the same holdout."""
    if not y_train:
        raise ValueError("Training targets must not be empty")
    if not y_test:
        raise ValueError("Holdout targets must not be empty")

    prediction_arrays = (
        historical_average_predictions,
        random_forest_predictions,
        model_predictions,
    )
    if any(len(values) != len(y_test) for values in prediction_arrays):
        raise ValueError("All holdout prediction arrays must have equal length")

    median_prediction = float(pd.Series(y_train, dtype=float).median())
    metrics = {
        "training_median": compute_regression_metrics(
            y_test, [median_prediction] * len(y_test)
        ),
        "historical_average": compute_regression_metrics(
            y_test, historical_average_predictions
        ),
        "random_forest": compute_regression_metrics(y_test, random_forest_predictions),
        "tuned_xgboost": compute_regression_metrics(y_test, model_predictions),
    }

    improvements: dict[str, dict[str, float | None]] = {}
    for model_name in ("random_forest", "tuned_xgboost"):
        improvements[model_name] = {}
        model_mae = metrics[model_name]["mae_minutes"]
        for baseline_name in ("training_median", "historical_average"):
            baseline_mae = metrics[baseline_name]["mae_minutes"]
            improvements[model_name][baseline_name] = (
                None
                if baseline_mae == 0
                else float((baseline_mae - model_mae) / baseline_mae * 100.0)
            )

    return {
        "metrics": metrics,
        "mae_improvement_percent_over_baselines": improvements,
        "interpretation": (
            "Positive improvement means lower MAE than the baseline; "
            "negative improvement means worse MAE."
        ),
    }


def validate_feature_columns(frame: pd.DataFrame) -> None:
    """Validate the ordered feature matrix against the model contract.

    Args:
        frame: DataFrame containing model input features.

    Raises:
        ValueError: If the feature names or order are incorrect, the
            matrix is empty, values are missing, or columns are non-numeric.
    """
    if tuple(frame.columns) != FEATURE_NAMES:
        raise ValueError(
            f"Feature schema mismatch. Expected {FEATURE_NAMES}; "
            f"received {tuple(frame.columns)}"
        )

    if frame.empty:
        raise ValueError("Feature matrix is empty")

    if frame.isna().any().any():
        raise ValueError("Feature matrix contains missing values")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All model features must be numeric")


def verify_candidate_artifact(candidate_path: Path, expected_dataset_mode: str) -> None:
    """Validate that the candidate artifact can be reloaded and conforms to serving schema.

    Args:
        candidate_path: Path to the serialized candidate booster JSON.
        expected_dataset_mode: Mode ('real' or 'synthetic').

    Raises:
        CandidateVerificationError: If the artifact cannot be loaded, has wrong
            feature count or ordering, missing metadata, or fails smoke test.
    """
    if not candidate_path.is_file():
        raise CandidateVerificationError(
            f"Candidate artifact not found at {candidate_path}"
        )

    candidate = xgb.Booster()
    try:
        candidate.load_model(str(candidate_path))
    except Exception as exc:
        raise CandidateVerificationError(
            f"Failed to reload serialized candidate artifact: {exc}"
        ) from exc

    actual_names = candidate.feature_names
    if (
        candidate.num_features() != len(FEATURE_NAMES)
        or actual_names is None
        or tuple(actual_names) != FEATURE_NAMES
    ):
        raise CandidateVerificationError(
            f"Candidate feature schema mismatch: expected {list(FEATURE_NAMES)}, got {actual_names}"
        )

    dataset_mode = candidate.attr("dataset_mode")
    if dataset_mode != expected_dataset_mode:
        raise CandidateVerificationError(
            f"Candidate dataset_mode metadata mismatch: expected {expected_dataset_mode!r}, "
            f"got {dataset_mode!r}"
        )

    # Smoke prediction verification
    smoke_matrix = xgb.DMatrix(
        np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32),
        feature_names=list(FEATURE_NAMES),
    )
    prediction = np.asarray(candidate.predict(smoke_matrix)).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise CandidateVerificationError(
            "Candidate smoke prediction was not a single finite value"
        )

    logger.info(
        "Candidate artifact verified against serving contract",
        artifact_path=str(candidate_path),
    )


async def load_mongo_records(
    collection_name: str,
    dataset_mode: str,
) -> list[dict[str, Any]]:
    """Load training records from MongoDB for one dataset mode.

    Args:
        collection_name: MongoDB collection to query.
        dataset_mode: Either ``synthetic`` or ``real``.

    Returns:
        Raw records matching the selected dataset mode.

    Raises:
        RuntimeError: If MongoDB connects without providing a database.
        Exception: Propagates database failures after connection cleanup.
    """
    if dataset_mode == "synthetic":
        query = {"is_synthetic": True}
    else:
        query = {"is_synthetic": {"$ne": True}}

    await mongo_manager.connect()
    try:
        database = mongo_manager.db
        if database is None:
            raise RuntimeError("MongoDB connection succeeded without a database")

        cursor = database[collection_name].find(query)
        return await cursor.to_list(length=None)
    finally:
        await mongo_manager.disconnect()


def load_timezone_map(path: Path) -> dict[str, str]:
    """Load explicit restaurant_id -> IANA timezone mapping."""
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Timezone map must be a non-empty JSON object")

    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError("Timezone map must contain string keys and values")

    return mapping


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in params.items()
    }


async def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate candidate ETA models using chronological data and track with MLflow.

    Args:
        args: Parsed training configuration, including dataset mode,
            timezone mapping, cross-validation settings, eligibility thresholds, and output directory.

    Returns:
        Training report containing metrics, diagnostics, eligibility, and artifact paths.

    Raises:
        InsufficientDataError: If the dataset is insufficient after cleaning.
        CandidateVerificationError: If candidate model validation fails.
    """
    logger.info(
        "Starting ETA model training",
        dataset_mode=args.dataset_mode,
        collection=args.collection,
        feature_count=len(FEATURE_NAMES),
        min_sample_size=args.min_sample_size,
    )
    timezone_map = load_timezone_map(args.timezone_map)
    raw_records = await load_mongo_records(
        args.collection,
        args.dataset_mode,
    )

    logger.info(
        "Loaded ETA training records from MongoDB",
        record_count=len(raw_records),
        dataset_mode=args.dataset_mode,
        collection=args.collection,
    )

    cleaned, cleaning_report = clean_training_records(raw_records)

    logger.info(
        "Completed ETA training data cleaning",
        input_records=len(raw_records),
        valid_records=len(cleaned),
        cleaning_report=cleaning_report,
    )

    if len(cleaned) < args.min_sample_size:
        raise InsufficientDataError(
            f"Only {len(cleaned)} valid records obtained from {len(raw_records)} raw records; "
            f"configured minimum sample size is {args.min_sample_size}. "
            f"Rejections breakdown: {cleaning_report['rejection_reasons']}"
        )

    split = chronological_train_test_split(
        cleaned,
        test_fraction=args.test_fraction,
    )

    logger.info(
        "Completed chronological ETA data split",
        train_records=len(split.train_indices),
        test_records=len(split.test_indices),
        embargo_records=len(split.embargo_indices),
        cutoff_utc=split.cutoff.isoformat(),
    )

    feature_rows, targets = build_training_features(
        split.records,
        cold_start_average=args.cold_start_average,
        restaurant_timezones=timezone_map,
    )

    all_features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    validate_feature_columns(all_features)

    train_indices = list(split.train_indices)
    test_indices = list(split.test_indices)

    x_train = all_features.iloc[train_indices].reset_index(drop=True)
    y_train = [targets[index] for index in train_indices]

    x_test = all_features.iloc[test_indices].reset_index(drop=True)
    y_test = [targets[index] for index in test_indices]
    test_records = [split.records[index] for index in test_indices]

    train_records = [split.records[index] for index in train_indices]
    cv = make_purged_time_series_cv(
        train_records,
        n_splits=args.n_splits,
        test_size=args.cv_test_size,
    )

    cv_splits = [
        (list(train_fold), list(validation_fold))
        for train_fold, validation_fold in cv.folds
    ]

    parameter_distributions = {
        "n_estimators": [100, 200, 300, 450, 600],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }

    estimator = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=args.seed,
        n_jobs=1,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=args.n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        refit=True,
        random_state=args.seed,
        n_jobs=args.search_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=0,
    )

    logger.info(
        "Starting XGBoost hyperparameter search",
        training_records=len(x_train),
        cv_folds=len(cv_splits),
        search_iterations=args.n_iter,
    )

    search.fit(x_train, y_train)

    logger.info(
        "Completed XGBoost hyperparameter search",
        best_cv_mae_minutes=float(-search.best_score_),
    )

    rf_estimator = RandomForestRegressor(
        random_state=args.seed,
        n_jobs=1,
    )
    rf_search = RandomizedSearchCV(
        estimator=rf_estimator,
        param_distributions={
            "n_estimators": [100, 200, 400, 600],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt", 0.8],
        },
        n_iter=args.n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        refit=True,
        random_state=args.seed,
        n_jobs=args.search_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=0,
    )

    logger.info(
        "Starting Random Forest hyperparameter search",
        training_records=len(x_train),
        cv_folds=len(cv_splits),
        search_iterations=args.n_iter,
    )

    rf_search.fit(x_train, y_train)

    logger.info(
        "Completed Random Forest hyperparameter search",
        best_cv_mae_minutes=float(-rf_search.best_score_),
    )

    validate_feature_columns(x_test)
    predictions = search.best_estimator_.predict(x_test)
    rf_predictions = rf_search.best_estimator_.predict(x_test)
    metrics = compute_regression_metrics(y_test, predictions.tolist())

    baseline_comparison = compare_with_baselines(
        y_train=y_train,
        y_test=y_test,
        historical_average_predictions=(
            x_test["historical_average_wait"].astype(float).tolist()
        ),
        random_forest_predictions=rf_predictions.astype(float).tolist(),
        model_predictions=predictions.astype(float).tolist(),
    )

    diagnostics = build_model_diagnostics(
        records=test_records,
        features=x_test,
        actual=y_test,
        predictions={
            "random_forest": rf_predictions.astype(float).tolist(),
            "tuned_xgboost": predictions.astype(float).tolist(),
        },
        xgboost_model=search.best_estimator_,
        random_forest_model=rf_search.best_estimator_,
    )

    # Candidate Eligibility Evaluation
    thresholds = EligibilityThresholds(
        max_mae_minutes=args.max_test_mae,
        max_rmse_minutes=args.max_test_rmse,
        require_beats_baseline=not args.allow_worse_than_baseline,
    )
    eligibility = evaluate_candidate_eligibility(
        test_metrics=metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode=args.dataset_mode,
    )

    logger.info(
        "Candidate eligibility evaluated",
        is_eligible=eligibility.is_eligible,
        rejection_reasons=eligibility.rejection_reasons,
    )

    candidate_path = args.output_dir / "eta_model_candidate.json"
    report_path = args.output_dir / "eta_training_report.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    booster = search.best_estimator_.get_booster()

    try:
        code_version = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        code_version = None

    # MLflow Tracking block
    tracker = MLflowTracker(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=f"eta-candidate-{args.dataset_mode}",
    )

    with tracker:
        booster.set_attr(
            feature_contract_version="v1",
            dataset_mode=args.dataset_mode,
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            model_version="eta_model_candidate",
            mlflow_run_id=tracker.run_id,
        )
        booster.save_model(str(candidate_path))

        # AC5: Post-save verification against serving contract without activating
        verify_candidate_artifact(
            candidate_path, expected_dataset_mode=args.dataset_mode
        )

        report = {
            "model_metadata": {
                "model_version": "eta_model_candidate",
                "artifact_format": "xgboost_json",
                "feature_contract_version": "v1",
                "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "code_version": code_version,
                "dataset_mode": args.dataset_mode,
                "is_synthetic": args.dataset_mode == "synthetic",
                "mlflow_run_id": tracker.run_id,
            },
            "status": "candidate_trained",
            "dataset_mode": args.dataset_mode,
            "synthetic_data_warning": (
                "Metrics from synthetic data validate pipeline mechanics only; "
                "they do not establish real-world ETA accuracy."
                if args.dataset_mode == "synthetic"
                else None
            ),
            "collection": args.collection,
            "input_records": len(raw_records),
            "cleaning": cleaning_report,
            "split": {
                "strategy": "chronological",
                "cutoff_utc": split.cutoff.isoformat(),
                "train_records": len(train_indices),
                "test_records": len(test_indices),
                "embargo_records": len(split.embargo_indices),
            },
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "cold_start_average_minutes": args.cold_start_average,
            "cv": {
                "strategy": "purged expanding-window time series",
                "folds": len(cv_splits),
                "search_iterations": args.n_iter,
                "selection_metric": "MAE",
                "best_cv_mae_minutes": float(-search.best_score_),
                "best_parameters": _json_safe_params(search.best_params_),
            },
            "test_metrics": metrics,
            "random_forest_cv": {
                "selection_metric": "MAE",
                "best_cv_mae_minutes": float(-rf_search.best_score_),
                "best_parameters": _json_safe_params(rf_search.best_params_),
            },
            "baseline_comparison": baseline_comparison,
            "eligibility": eligibility.to_dict(),
            "diagnostics": diagnostics,
            "mlflow": {
                "run_id": tracker.run_id,
                "tracking_uri": args.mlflow_tracking_uri,
                "experiment_name": args.mlflow_experiment_name,
            },
            "artifacts": {
                "candidate_model": str(candidate_path),
                "report": str(report_path),
            },
        }

        save_json(report_path, report)

        # Log details to MLflow
        mlflow_params = {
            "dataset_mode": args.dataset_mode,
            "collection": args.collection,
            "min_sample_size": args.min_sample_size,
            "test_fraction": args.test_fraction,
            "n_splits": args.n_splits,
            "search_iterations": args.n_iter,
            "seed": args.seed,
            "cold_start_average_minutes": args.cold_start_average,
            "max_test_mae": args.max_test_mae,
            "max_test_rmse": args.max_test_rmse,
            "code_version": code_version or "unknown",
            **_json_safe_params(search.best_params_),
        }

        hist_avg_mae = baseline_comparison["metrics"]["historical_average"][
            "mae_minutes"
        ]
        median_mae = baseline_comparison["metrics"]["training_median"]["mae_minutes"]
        rf_mae = baseline_comparison["metrics"]["random_forest"]["mae_minutes"]
        xgb_vs_hist = baseline_comparison["mae_improvement_percent_over_baselines"][
            "tuned_xgboost"
        ]["historical_average"]

        mlflow_metrics = {
            "input_records": float(len(raw_records)),
            "accepted_records": float(len(cleaned)),
            "rejected_records": float(cleaning_report["rejected_records"]),
            "cv_best_mae_minutes": float(-search.best_score_),
            "rf_cv_best_mae_minutes": float(-rf_search.best_score_),
            "test_mae_minutes": float(metrics["mae_minutes"]),
            "test_rmse_minutes": float(metrics["rmse_minutes"]),
            "test_r2": float(metrics["r2"]) if metrics["r2"] is not None else None,
            "baseline_hist_avg_mae_minutes": float(hist_avg_mae),
            "baseline_median_mae_minutes": float(median_mae),
            "baseline_rf_mae_minutes": float(rf_mae),
            "mae_improvement_pct_over_hist_avg": (
                float(xgb_vs_hist) if xgb_vs_hist is not None else None
            ),
        }

        mlflow_tags = {
            "dataset_mode": args.dataset_mode,
            "is_synthetic": str(args.dataset_mode == "synthetic").lower(),
            "candidate_eligible": str(eligibility.is_eligible).lower(),
            "feature_contract_version": "v1",
        }

        tracker.log_training_run(
            params=mlflow_params,
            metrics=mlflow_metrics,
            tags=mlflow_tags,
            artifact_paths=[str(candidate_path), str(report_path)],
        )

    logger.info(
        "ETA model training completed",
        candidate_model=str(candidate_path),
        report_path=str(report_path),
        test_mae_minutes=metrics["mae_minutes"],
        dataset_mode=args.dataset_mode,
        is_eligible=eligibility.is_eligible,
        mlflow_run_id=tracker.run_id,
    )

    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated ETA Model Training & Retraining Pipeline with MLflow tracking and candidate validation."
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("synthetic", "real"),
        required=True,
        help="Choose one dataset type explicitly; synthetic and real rows are never mixed.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION_NAME", settings.MONGO_COLLECTION_NAME),
        help="MongoDB collection name (defaults to MONGO_COLLECTION_NAME or 'queue_events').",
    )
    parser.add_argument(
        "--timezone-map",
        type=Path,
        default=Path("config/restaurant_timezones.json"),
        help="JSON file mapping every restaurant_id to an IANA timezone.",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=50,
        help="Minimum valid records required before training commences.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--cv-test-size", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cold-start-average",
        type=float,
        default=settings.COLD_START_AVERAGE_WAIT_MINUTES,
        help="Fallback historical average wait in minutes.",
    )
    parser.add_argument("--search-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-test-mae",
        type=float,
        default=5.0,
        help="Maximum acceptable holdout MAE in minutes for candidate promotion eligibility.",
    )
    parser.add_argument(
        "--max-test-rmse",
        type=float,
        default=8.0,
        help="Maximum acceptable holdout RMSE in minutes for candidate promotion eligibility.",
    )
    parser.add_argument(
        "--allow-worse-than-baseline",
        action="store_true",
        default=False,
        help="Allow candidate eligibility even if it does not outperform historical average baseline.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI),
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", settings.MLFLOW_EXPERIMENT_NAME),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("app/infrastructure/ml_models/artifacts/candidates"),
    )
    return parser.parse_args(args_list)


if __name__ == "__main__":
    try:
        asyncio.run(run_training(parse_args()))
    except InsufficientDataError as exc:
        logger.error("ETA model training rejected: insufficient data", error=str(exc))
        raise SystemExit(2)
    except CandidateVerificationError as exc:
        logger.critical("ETA candidate failed verification", error=str(exc))
        raise SystemExit(3)
    except Exception:
        logger.exception("ETA model training failed")
        raise SystemExit(1)
