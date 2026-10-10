"""Leakage-aware XGBoost training, tuning, and chronological evaluation."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.ml_models.data_cleaning import clean_training_records
from app.infrastructure.ml_models.dataset import (
    chronological_train_test_split,
    make_purged_time_series_cv,
)
from app.infrastructure.ml_models.features import (
    FEATURE_NAMES,
    build_training_features,
)


def compute_regression_metrics(
    actual: list[float],
    predicted: list[float],
) -> dict[str, float | None]:
    """Compute common regression metrics."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal nonzero length")

    if not all(math.isfinite(float(value)) for value in actual + predicted):
        raise ValueError("Metrics require finite values")

    return {
        "mae_minutes": float(mean_absolute_error(actual, predicted)),
        "rmse_minutes": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def validate_feature_columns(frame: pd.DataFrame) -> None:
    """Reject missing, extra, or reordered model features."""
    if tuple(frame.columns) != FEATURE_NAMES:
        raise ValueError(
            f"Feature schema mismatch. Expected {FEATURE_NAMES}; "
            f"received {tuple(frame.columns)}"
        )

    if frame.empty:
        raise ValueError("Feature matrix is empty")

    if frame.isna().any().any():
        raise ValueError("Feature matrix contains missing values")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise ValueError("All model features must be numeric")


async def load_mongo_records(
    collection_name: str,
    dataset_mode: str,
) -> list[dict[str, Any]]:
    """Load either synthetic or real records without mixing the two."""
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
    timezone_map = load_timezone_map(args.timezone_map)
    raw_records = await load_mongo_records(
        args.collection,
        args.dataset_mode,
    )

    cleaned, cleaning_report = clean_training_records(raw_records)
    if len(cleaned) < 30:
        raise ValueError(
            f"Only {len(cleaned)} valid records; at least 30 are required "
            "for this initial training workflow."
        )

    split = chronological_train_test_split(
        cleaned,
        test_fraction=args.test_fraction,
    )

    # Features are built in chronological order. A completed wait can affect
    # later rows only after its actual completion timestamp.
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

    train_records = [split.records[index] for index in train_indices]
    cv = make_purged_time_series_cv(
        train_records,
        n_splits=args.n_splits,
        test_size=args.cv_test_size,
    )

    # These folds are relative to x_train/y_train and purge labels that
    # would not have completed before each validation window.
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
        verbose=1,
    )

    search.fit(x_train, y_train)

    validate_feature_columns(x_test)
    predictions = search.best_estimator_.predict(x_test)
    metrics = compute_regression_metrics(y_test, predictions.tolist())

    candidate_path = args.output_dir / "eta_model_candidate.json"
    report_path = args.output_dir / "eta_training_report.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    booster = search.best_estimator_.get_booster()
    if booster.num_features() != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Candidate has {booster.num_features()} features; "
            f"expected {len(FEATURE_NAMES)}"
        )

    if tuple(booster.feature_names or ()) != FEATURE_NAMES:
        raise RuntimeError(
            f"Candidate feature names mismatch: {booster.feature_names}"
        )

    booster.set_attr(
        feature_contract_version="v1",
        dataset_mode=args.dataset_mode,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    booster.save_model(str(candidate_path))

    report = {
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
        "artifacts": {
            "candidate_model": str(candidate_path),
            "report": str(report_path),
        },
    }

    save_json(report_path, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a leakage-aware SpotQ ETA XGBoost model."
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("synthetic", "real"),
        required=True,
        help="Choose one dataset type explicitly; synthetic and real rows are never mixed.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION_NAME", "spotq_production"),
    )
    parser.add_argument(
        "--timezone-map",
        type=Path,
        required=True,
        help="JSON file mapping every restaurant_id to an IANA timezone.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--cv-test-size", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold-start-average", type=float, default=20.0)
    parser.add_argument("--search-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("app/infrastructure/ml_models/artifacts/candidates"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_training(parse_args()))
