"""Holdout diagnostics for SpotQ ETA regression models."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.infrastructure.ml_models.features import FEATURE_NAMES


def _metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float | int]:
    return {
        "records": int(len(actual)),
        "mae_minutes": float(mean_absolute_error(actual, predicted)),
        "rmse_minutes": float(math.sqrt(mean_squared_error(actual, predicted))),
    }


def _feature_importance(model: Any, model_type: str) -> dict[str, float]:
    if model_type == "random_forest":
        values = getattr(model, "feature_importances_", None)
        if values is None or len(values) != len(FEATURE_NAMES):
            raise ValueError("Random Forest importance does not match feature contract")
        pairs = zip(FEATURE_NAMES, values)
    else:
        raw = model.get_booster().get_score(importance_type="gain")
        pairs = (
            (name, raw.get(name, raw.get(f"f{index}", 0.0)))
            for index, name in enumerate(FEATURE_NAMES)
        )

    return {
        name: float(value)
        for name, value in sorted(pairs, key=lambda item: item[1], reverse=True)
    }


def build_model_diagnostics(
    *,
    records: Sequence[Mapping[str, Any]],
    features: pd.DataFrame,
    actual: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    xgboost_model: Any,
    random_forest_model: Any,
) -> dict[str, Any]:
    """Build holdout-only error, segment, sanity, and importance reports."""
    count = len(actual)
    if count == 0 or len(records) != count or len(features) != count:
        raise ValueError("Records, features, and targets must have equal nonzero length")
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Diagnostic feature columns do not match the feature contract")

    frame = features.reset_index(drop=True).copy()
    frame["actual_wait_minutes"] = [float(value) for value in actual]
    frame["restaurant_id"] = [
        str(record.get("restaurant_id", "unknown")) for record in records
    ]
    frame["party_size_band"] = frame["party_size"].map(
        lambda value: "1-2" if value <= 2 else ("3-4" if value <= 4 else "5+")
    )
    frame["queue_size_band"] = frame["total_people_ahead"].map(
        lambda value: "0-4" if value < 5 else (
            "5-10" if value <= 10 else ("11-20" if value <= 20 else "21+")
        )
    )

    model_reports: dict[str, Any] = {}

    for model_name, raw_predictions in predictions.items():
        values = [float(value) for value in raw_predictions]
        if len(values) != count:
            raise ValueError(f"Prediction length mismatch for {model_name}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite predictions found for {model_name}")

        residuals = [pred - target for pred, target in zip(values, actual)]
        absolute_errors = [abs(value) for value in residuals]

        frame["prediction"] = values
        frame["residual_minutes"] = residuals
        frame["absolute_error_minutes"] = absolute_errors

        segment_reports: dict[str, Any] = {}
        for dimension in (
            "restaurant_id",
            "hour_of_day",
            "party_size_band",
            "queue_size_band",
        ):
            grouped = {}
            for key, group in frame.groupby(dimension, dropna=False, sort=True):
                grouped[str(key)] = _metrics(
                    group["actual_wait_minutes"].tolist(),
                    group["prediction"].tolist(),
                )
            segment_reports[dimension] = grouped

        ranked = frame.sort_values("absolute_error_minutes", ascending=False).head(10)
        largest_errors = []
        for index, row in ranked.iterrows():
            largest_errors.append({
                "holdout_row": int(index),
                "restaurant_id": str(row["restaurant_id"]),
                "hour_of_day": int(row["hour_of_day"]),
                "actual_wait_minutes": float(row["actual_wait_minutes"]),
                "predicted_wait_minutes": float(row["prediction"]),
                "error_minutes": float(row["residual_minutes"]),
                "absolute_error_minutes": float(row["absolute_error_minutes"]),
            })

        model_reports[model_name] = {
            "overall": {
                **_metrics(actual, values),
                "mean_error_minutes": float(sum(residuals) / count),
                "median_absolute_error_minutes": float(
                    pd.Series(absolute_errors).median()
                ),
                "p90_absolute_error_minutes": float(
                    pd.Series(absolute_errors).quantile(0.90)
                ),
                "max_absolute_error_minutes": float(max(absolute_errors)),
            },
            "prediction_sanity": {
                "negative_predictions": int(sum(value < 0 for value in values)),
                "zero_predictions": int(sum(value == 0 for value in values)),
                "above_120_minutes": int(sum(value > 120 for value in values)),
                "within_0_to_120_minutes": int(
                    sum(0 <= value <= 120 for value in values)
                ),
            },
            "segment_metrics": segment_reports,
            "largest_errors": largest_errors,
        }

    return {
        "scope": "chronological holdout only",
        "holdout_records": count,
        "feature_importance": {
            "xgboost_gain": _feature_importance(xgboost_model, "xgboost"),
            "random_forest": _feature_importance(random_forest_model, "random_forest"),
        },
        "models": model_reports,
        "caveat": (
            "Feature importance is model-specific diagnostic information, not causal "
            "evidence. Synthetic-data diagnostics do not establish real-world ETA accuracy."
        ),
    }
