"""Unit tests for statistical feature drift detector (AC3)."""

import pytest

from app.application.monitoring.drift_detector import (
    PSI_SIGNIFICANT_THRESHOLD,
    FeatureDriftDetector,
    compute_psi,
)
from app.infrastructure.ml_models.features import FEATURE_NAMES


def test_compute_psi_identical_distributions():
    """Identical distributions produce near-zero PSI."""
    counts = [100, 200, 300, 400]
    expected_props = [0.1, 0.2, 0.3, 0.4]

    psi = compute_psi(counts, expected_props)
    assert psi == pytest.approx(0.0, abs=0.01)


def test_compute_psi_drastic_shift():
    """Significant divergence produces high PSI (>= 0.25)."""
    # Inverse distribution
    counts = [400, 300, 200, 100]
    expected_props = [0.1, 0.2, 0.3, 0.4]

    psi = compute_psi(counts, expected_props)
    assert psi >= PSI_SIGNIFICANT_THRESHOLD


def test_drift_detector_handles_insufficient_samples():
    """AC3: Batches below min_sample_size are marked insufficient without false alarms."""
    detector = FeatureDriftDetector(min_sample_size=30)
    # Only 5 samples
    samples = [
        {
            "party_size": 2.0,
            "total_people_ahead": 4.0,
            "hour_of_day": 12.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "historical_average_wait": 15.0,
        }
        for _ in range(5)
    ]

    report = detector.evaluate_feature_drift(samples)

    assert report.is_sufficient_samples is False
    assert report.drifted_features_count == 0
    assert report.has_significant_drift is False
    for res in report.feature_results.values():
        assert res.drift_level == "insufficient_samples"
        assert res.is_drifted is False


def test_drift_detector_stable_on_baseline_distributed_data():
    """AC3: Samples drawn according to baseline are flagged as stable."""
    detector = FeatureDriftDetector(min_sample_size=30)

    # 100 samples faithfully representing baseline expected distribution
    samples = []
    for i in range(100):
        # party_size expected: [0.18, 0.42, 0.14, 0.16, 0.07, 0.03]
        if i < 18:
            ps = 1.0
        elif i < 60:
            ps = 2.0
        elif i < 74:
            ps = 3.0
        elif i < 90:
            ps = 4.0
        elif i < 97:
            ps = 5.0
        else:
            ps = 8.0

        # total_people_ahead expected: [0.25, 0.28, 0.22, 0.15, 0.10]
        if i < 25:
            tpa = 1.0
        elif i < 53:
            tpa = 4.0
        elif i < 75:
            tpa = 8.0
        elif i < 90:
            tpa = 14.0
        else:
            tpa = 25.0

        # hour_of_day expected: [0.10, 0.35, 0.15, 0.35, 0.05]
        if i < 10:
            hr = 8.0
        elif i < 45:
            hr = 13.0
        elif i < 60:
            hr = 16.0
        elif i < 95:
            hr = 20.0
        else:
            hr = 23.0

        # day_of_week expected: [0.12, 0.12, 0.12, 0.14, 0.16, 0.18, 0.16]
        if i < 12:
            dow = 0.0
        elif i < 24:
            dow = 1.0
        elif i < 36:
            dow = 2.0
        elif i < 50:
            dow = 3.0
        elif i < 66:
            dow = 4.0
        elif i < 84:
            dow = 5.0
        else:
            dow = 6.0

        # historical_average_wait expected: [0.15, 0.35, 0.30, 0.15, 0.05]
        if i < 15:
            haw = 5.0
        elif i < 50:
            haw = 14.0
        elif i < 80:
            haw = 22.0
        elif i < 95:
            haw = 35.0
        else:
            haw = 55.0

        samples.append(
            {
                "party_size": ps,
                "total_people_ahead": tpa,
                "hour_of_day": hr,
                "day_of_week": dow,
                "is_weekend": 1.0 if dow in (5.0, 6.0) else 0.0,
                "historical_average_wait": haw,
            }
        )

    report = detector.evaluate_feature_drift(samples)

    assert report.is_sufficient_samples is True
    # Most or all features should be stable or at worst moderate, not significant drift
    assert report.has_significant_drift is False


def test_drift_detector_detects_significant_shift():
    """AC3: Drastic shift in party_size and hour_of_day trips significant drift."""
    detector = FeatureDriftDetector(min_sample_size=30)

    # 50 samples with massive party sizes and anomalous 3 AM hours
    shifted_samples = [
        {
            "party_size": 25.0,  # Baseline expects 1-4
            "total_people_ahead": 80.0,
            "hour_of_day": 3.0,  # 3 AM baseline expects 0.05
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "historical_average_wait": 95.0,
        }
        for _ in range(50)
    ]

    report = detector.evaluate_feature_drift(shifted_samples)

    assert report.is_sufficient_samples is True
    assert report.has_significant_drift is True
    assert report.drifted_features_count >= 2

    party_result = report.feature_results["party_size"]
    assert party_result.is_drifted is True
    assert party_result.drift_level == "significant_drift"
    assert party_result.psi_score >= PSI_SIGNIFICANT_THRESHOLD


def test_drift_detector_all_canonical_features_evaluated():
    """AC3: All 6 canonical features are present in the evaluation results."""
    detector = FeatureDriftDetector(min_sample_size=30)
    samples = [
        {
            "party_size": 2.0,
            "total_people_ahead": 3.0,
            "hour_of_day": 13.0,
            "day_of_week": 1.0,
            "is_weekend": 0.0,
            "historical_average_wait": 18.0,
        }
        for _ in range(35)
    ]

    report = detector.evaluate_feature_drift(samples)
    for name in FEATURE_NAMES:
        assert name in report.feature_results
