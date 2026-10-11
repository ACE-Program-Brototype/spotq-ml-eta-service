import pandas as pd
import pytest

from scripts.train_eta_model import (
    compute_regression_metrics,
    validate_feature_columns,
)
from app.infrastructure.ml_models.features import FEATURE_NAMES


def test_regression_metrics():
    metrics = compute_regression_metrics(
        [10.0, 20.0, 30.0],
        [12.0, 18.0, 33.0],
    )

    assert metrics["mae_minutes"] == pytest.approx(7 / 3)
    assert metrics["rmse_minutes"] == pytest.approx((17 / 3) ** 0.5)
    assert metrics["r2"] is not None


def test_feature_matrix_schema_is_validated():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, 15.0]],
        columns=FEATURE_NAMES,
    )
    validate_feature_columns(frame)


def test_feature_matrix_rejects_wrong_order():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, 15.0]],
        columns=list(reversed(FEATURE_NAMES)),
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_feature_columns(frame)


def test_feature_matrix_rejects_missing_values():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, float("nan")]],
        columns=FEATURE_NAMES,
    )

    with pytest.raises(ValueError, match="missing values"):
        validate_feature_columns(frame)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_regression_metrics([1.0, 2.0], [1.0])


def test_baseline_comparison_reports_all_predictors():
    from scripts.train_eta_model import compare_with_baselines

    result = compare_with_baselines(
        y_train=[10.0, 20.0, 30.0],
        y_test=[10.0, 20.0, 30.0],
        historical_average_predictions=[12.0, 18.0, 32.0],
        random_forest_predictions=[10.0, 22.0, 28.0],
        model_predictions=[11.0, 21.0, 29.0],
    )

    metrics = result["metrics"]
    assert set(metrics) == {
        "training_median",
        "historical_average",
        "random_forest",
        "tuned_xgboost",
    }
    assert metrics["training_median"]["mae_minutes"] == pytest.approx(20 / 3)
    assert metrics["historical_average"]["mae_minutes"] == pytest.approx(2.0)
    assert metrics["random_forest"]["mae_minutes"] == pytest.approx(4 / 3)
    assert metrics["tuned_xgboost"]["mae_minutes"] == pytest.approx(1.0)
    assert result["mae_improvement_percent_over_baselines"][
        "tuned_xgboost"
    ]["historical_average"] == pytest.approx(50.0)


def test_baseline_comparison_rejects_mismatched_predictions():
    from scripts.train_eta_model import compare_with_baselines

    with pytest.raises(ValueError, match="equal length"):
        compare_with_baselines(
            y_train=[10.0, 20.0],
            y_test=[10.0, 20.0],
            historical_average_predictions=[15.0],
            random_forest_predictions=[10.0, 20.0],
            model_predictions=[10.0, 20.0],
        )


def test_baseline_comparison_handles_zero_baseline_mae():
    from scripts.train_eta_model import compare_with_baselines

    result = compare_with_baselines(
        y_train=[10.0, 10.0],
        y_test=[10.0, 10.0],
        historical_average_predictions=[10.0, 10.0],
        random_forest_predictions=[10.0, 10.0],
        model_predictions=[10.0, 10.0],
    )

    assert result["mae_improvement_percent_over_baselines"][
        "tuned_xgboost"
    ]["training_median"] is None
    assert result["mae_improvement_percent_over_baselines"][
        "tuned_xgboost"
    ]["historical_average"] is None


def test_model_diagnostics_reports_segments_and_sanity():
    from app.infrastructure.ml_models.diagnostics import build_model_diagnostics

    class FakeBooster:
        def get_score(self, importance_type):
            assert importance_type == "gain"
            return {"party_size": 4.0, "hour_of_day": 2.0}

    class FakeXGBoost:
        def get_booster(self):
            return FakeBooster()

    class FakeRandomForest:
        feature_importances_ = [0.3, 0.2, 0.1, 0.1, 0.1, 0.2]

    features = pd.DataFrame(
        [
            [2, 3, 12, 0, 0, 15.0],
            [4, 8, 13, 0, 0, 18.0],
            [6, 25, 14, 0, 0, 20.0],
        ],
        columns=FEATURE_NAMES,
    )
    result = build_model_diagnostics(
        records=[
            {"restaurant_id": "r1"},
            {"restaurant_id": "r1"},
            {"restaurant_id": "r2"},
        ],
        features=features,
        actual=[15.0, 20.0, 30.0],
        predictions={
            "tuned_xgboost": [14.0, 22.0, 135.0],
            "random_forest": [16.0, 19.0, 28.0],
        },
        xgboost_model=FakeXGBoost(),
        random_forest_model=FakeRandomForest(),
    )

    xgb = result["models"]["tuned_xgboost"]
    assert result["holdout_records"] == 3
    assert xgb["prediction_sanity"]["above_120_minutes"] == 1
    assert xgb["segment_metrics"]["restaurant_id"]["r1"]["records"] == 2
    assert len(xgb["largest_errors"]) == 3
    assert "party_size" in result["feature_importance"]["xgboost_gain"]
    assert "party_size" in result["feature_importance"]["random_forest"]


def test_model_diagnostics_rejects_non_finite_predictions():
    from app.infrastructure.ml_models.diagnostics import build_model_diagnostics

    class FakeBooster:
        def get_score(self, importance_type):
            return {}

    class FakeXGBoost:
        def get_booster(self):
            return FakeBooster()

    class FakeRandomForest:
        feature_importances_ = [0.0] * len(FEATURE_NAMES)

    features = pd.DataFrame([[2, 3, 12, 0, 0, 15.0]], columns=FEATURE_NAMES)
    with pytest.raises(ValueError, match="Non-finite"):
        build_model_diagnostics(
            records=[{"restaurant_id": "r1"}],
            features=features,
            actual=[15.0],
            predictions={
                "tuned_xgboost": [float("nan")],
                "random_forest": [15.0],
            },
            xgboost_model=FakeXGBoost(),
            random_forest_model=FakeRandomForest(),
        )
