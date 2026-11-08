"""Statistical feature drift detector for the 6 canonical ETA features (AC3).

Calculates Population Stability Index (PSI) against a reference baseline distribution
to detect covariate shift across numerical and categorical features.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES

EPSILON = 1e-4
DEFAULT_MIN_SAMPLE_SIZE = 30
DEFAULT_BASELINE_PATH = Path("config/feature_drift_baseline.json")

# Standard MLOps PSI thresholds
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


class SingleFeatureDriftResult(BaseModel):
    """Drift calculation result for a single model feature."""

    feature_name: str
    feature_type: str
    psi_score: float
    drift_level: str = Field(
        ..., description="'stable', 'moderate_drift', 'significant_drift', or 'insufficient_samples'"
    )
    is_drifted: bool = Field(
        ..., description="True if PSI >= PSI_SIGNIFICANT_THRESHOLD"
    )
    sample_count: int
    mean_actual: float | None = None
    mean_baseline: float | None = None


class FeatureDriftReport(BaseModel):
    """Batch feature drift evaluation report."""

    evaluation_timestamp: datetime
    baseline_version: str
    sample_count: int
    is_sufficient_samples: bool
    drifted_features_count: int
    has_significant_drift: bool
    feature_results: dict[str, SingleFeatureDriftResult]


def compute_psi(
    actual_counts: list[int] | np.ndarray,
    expected_proportions: list[float] | np.ndarray,
) -> float:
    """Calculate the Population Stability Index (PSI) with Laplace smoothing."""
    total_actual = float(sum(actual_counts))
    if total_actual <= 0:
        return 0.0

    k = float(len(actual_counts))
    alpha = 0.5
    smoothed_total = total_actual + (alpha * k)

    psi = 0.0
    for count, exp_prop in zip(actual_counts, expected_proportions, strict=False):
        actual_prop = (float(count) + alpha) / smoothed_total
        expected_prop = max(float(exp_prop), EPSILON)

        contribution = (actual_prop - expected_prop) * math.log(
            actual_prop / expected_prop
        )
        psi += contribution

    return round(float(psi), 4)


def load_baseline_distribution(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> dict[str, Any]:
    """Load the reference baseline distributions from disk."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Drift baseline distribution file not found at {baseline_path}"
        )

    with baseline_path.open("r", encoding="utf-8") as file:
        return json.load(file)


class FeatureDriftDetector:
    """Detects covariate drift across the 6 canonical ETA model features."""

    def __init__(
        self,
        baseline_path: Path = DEFAULT_BASELINE_PATH,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        significant_drift_threshold: float = PSI_SIGNIFICANT_THRESHOLD,
    ) -> None:
        self.baseline_path = baseline_path
        self.min_sample_size = min_sample_size
        self.significant_drift_threshold = significant_drift_threshold
        self._baseline_data = load_baseline_distribution(baseline_path)
        self.baseline_version = self._baseline_data.get("baseline_version", "unknown")

    def evaluate_drift(
        self,
        feature_rows: list[dict[str, float]],
    ) -> FeatureDriftReport:
        """Alias for evaluate_feature_drift."""
        return self.evaluate_feature_drift(feature_rows)

    def evaluate_feature_drift(
        self,
        feature_rows: list[dict[str, float]],
    ) -> FeatureDriftReport:
        """Evaluate covariate drift across all canonical features for a sample batch."""
        sample_count = len(feature_rows)
        now_utc = datetime.now(timezone.utc)

        if sample_count < self.min_sample_size:
            logger.info(
                "Sample size insufficient for statistical drift evaluation",
                sample_count=sample_count,
                min_sample_size=self.min_sample_size,
            )
            empty_results = {
                name: SingleFeatureDriftResult(
                    feature_name=name,
                    feature_type=self._baseline_data["features"]
                    .get(name, {})
                    .get("type", "unknown"),
                    psi_score=0.0,
                    drift_level="insufficient_samples",
                    is_drifted=False,
                    sample_count=sample_count,
                )
                for name in FEATURE_NAMES
            }
            return FeatureDriftReport(
                evaluation_timestamp=now_utc,
                baseline_version=self.baseline_version,
                sample_count=sample_count,
                is_sufficient_samples=False,
                drifted_features_count=0,
                has_significant_drift=False,
                feature_results=empty_results,
            )

        baseline_features = self._baseline_data["features"]
        feature_results: dict[str, SingleFeatureDriftResult] = {}
        drifted_count = 0

        for feature_name in FEATURE_NAMES:
            if feature_name not in baseline_features:
                continue

            config = baseline_features[feature_name]
            f_type = config.get("type", "numerical")
            expected_props = config["expected_proportions"]

            values = [row[feature_name] for row in feature_rows if feature_name in row]
            if len(values) < self.min_sample_size:
                feature_results[feature_name] = SingleFeatureDriftResult(
                    feature_name=feature_name,
                    feature_type=f_type,
                    psi_score=0.0,
                    drift_level="insufficient_samples",
                    is_drifted=False,
                    sample_count=len(values),
                )
                continue

            # Binning logic
            if f_type == "categorical":
                categories = config["categories"]
                counts = [0] * len(categories)
                for v in values:
                    try:
                        int_v = int(round(v))
                        if int_v in categories:
                            counts[categories.index(int_v)] += 1
                        else:
                            # Put unseen categories into the first bucket
                            counts[0] += 1
                    except (ValueError, TypeError):
                        counts[0] += 1
            else:
                # Numerical binning
                bins = config["bins"]
                # np.histogram with right=False gives bins [b0, b1), [b1, b2), etc.
                counts, _ = np.histogram(values, bins=bins)
                counts = counts.tolist()

            psi_score = compute_psi(counts, expected_props)

            if psi_score >= self.significant_drift_threshold:
                drift_level = "significant_drift"
                is_drifted = True
                drifted_count += 1
            elif psi_score >= PSI_MODERATE_THRESHOLD:
                drift_level = "moderate_drift"
                is_drifted = False
            else:
                drift_level = "stable"
                is_drifted = False

            mean_actual = float(np.mean(values)) if values else None
            mean_baseline = config.get("mean")

            feature_results[feature_name] = SingleFeatureDriftResult(
                feature_name=feature_name,
                feature_type=f_type,
                psi_score=psi_score,
                drift_level=drift_level,
                is_drifted=is_drifted,
                sample_count=len(values),
                mean_actual=round(mean_actual, 2) if mean_actual is not None else None,
                mean_baseline=round(mean_baseline, 2)
                if mean_baseline is not None
                else None,
            )

        has_significant_drift = drifted_count > 0

        logger.info(
            "Feature drift evaluation completed",
            sample_count=sample_count,
            drifted_features_count=drifted_count,
            has_significant_drift=has_significant_drift,
        )

        return FeatureDriftReport(
            evaluation_timestamp=now_utc,
            baseline_version=self.baseline_version,
            sample_count=sample_count,
            is_sufficient_samples=True,
            drifted_features_count=drifted_count,
            has_significant_drift=has_significant_drift,
            feature_results=feature_results,
        )
