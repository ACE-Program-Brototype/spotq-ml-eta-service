"""Unit tests for ETA candidate eligibility evaluation."""

import pytest

from app.infrastructure.ml_models.tracking import (
    EligibilityThresholds,
    evaluate_candidate_eligibility,
)


@pytest.fixture
def baseline_comparison():
    return {
        "metrics": {
            "training_median": {"mae_minutes": 6.5, "rmse_minutes": 8.0},
            "historical_average": {"mae_minutes": 4.8, "rmse_minutes": 6.2},
            "random_forest": {"mae_minutes": 3.9, "rmse_minutes": 5.1},
            "tuned_xgboost": {"mae_minutes": 3.2, "rmse_minutes": 4.5},
        },
        "mae_improvement_percent_over_baselines": {
            "tuned_xgboost": {
                "historical_average": 33.3,
                "training_median": 50.7,
            }
        },
    }


def test_eligible_candidate_on_real_data(baseline_comparison):
    test_metrics = {"mae_minutes": 3.2, "rmse_minutes": 4.5, "r2": 0.85}
    thresholds = EligibilityThresholds(
        max_mae_minutes=5.0,
        max_rmse_minutes=8.0,
        require_beats_baseline=True,
    )

    result = evaluate_candidate_eligibility(
        test_metrics=test_metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode="real",
    )

    assert result.is_eligible is True
    assert len(result.rejection_reasons) == 0
    assert result.checks["mae_within_threshold"] is True
    assert result.checks["rmse_within_threshold"] is True
    assert result.checks["beats_historical_average"] is True
    assert result.checks["real_data_provenance"] is True


def test_synthetic_data_candidate_rejected_for_promotion(baseline_comparison):
    test_metrics = {"mae_minutes": 2.5, "rmse_minutes": 3.5, "r2": 0.90}
    thresholds = EligibilityThresholds(
        max_mae_minutes=5.0,
        max_rmse_minutes=8.0,
        require_beats_baseline=True,
    )

    result = evaluate_candidate_eligibility(
        test_metrics=test_metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode="synthetic",
    )

    assert result.is_eligible is False
    assert result.checks["real_data_provenance"] is False
    assert any("Synthetic data" in r for r in result.rejection_reasons)


def test_candidate_rejected_when_mae_exceeds_threshold(baseline_comparison):
    test_metrics = {"mae_minutes": 5.8, "rmse_minutes": 7.0, "r2": 0.70}
    thresholds = EligibilityThresholds(
        max_mae_minutes=5.0,
        max_rmse_minutes=8.0,
        require_beats_baseline=False,
    )

    result = evaluate_candidate_eligibility(
        test_metrics=test_metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode="real",
    )

    assert result.is_eligible is False
    assert result.checks["mae_within_threshold"] is False
    assert any("Test MAE" in r and "exceeds" in r for r in result.rejection_reasons)


def test_candidate_rejected_when_rmse_exceeds_threshold(baseline_comparison):
    test_metrics = {"mae_minutes": 4.0, "rmse_minutes": 9.2, "r2": 0.65}
    thresholds = EligibilityThresholds(
        max_mae_minutes=5.0,
        max_rmse_minutes=8.0,
        require_beats_baseline=False,
    )

    result = evaluate_candidate_eligibility(
        test_metrics=test_metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode="real",
    )

    assert result.is_eligible is False
    assert result.checks["rmse_within_threshold"] is False
    assert any("Test RMSE" in r and "exceeds" in r for r in result.rejection_reasons)


def test_candidate_rejected_when_failing_to_beat_historical_average(
    baseline_comparison,
):
    # Historical average is 4.8, candidate is 5.2
    test_metrics = {"mae_minutes": 5.2, "rmse_minutes": 6.8, "r2": 0.60}
    thresholds = EligibilityThresholds(
        max_mae_minutes=6.0,
        max_rmse_minutes=8.0,
        require_beats_baseline=True,
    )

    result = evaluate_candidate_eligibility(
        test_metrics=test_metrics,
        baseline_metrics=baseline_comparison,
        thresholds=thresholds,
        dataset_mode="real",
    )

    assert result.is_eligible is False
    assert result.checks["beats_historical_average"] is False
    assert any(
        "does not outperform historical average" in r for r in result.rejection_reasons
    )


def test_missing_metrics_handled_gracefully(baseline_comparison):
    result = evaluate_candidate_eligibility(
        test_metrics={"mae_minutes": None, "rmse_minutes": None},
        baseline_metrics=baseline_comparison,
        thresholds=EligibilityThresholds(),
        dataset_mode="real",
    )

    assert result.is_eligible is False
    assert "metrics_complete" in result.checks
    assert result.checks["metrics_complete"] is False
