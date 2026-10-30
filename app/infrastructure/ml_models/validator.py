"""Candidate model pre-promotion validation engine for SCRUM-1228."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.registry import ModelManifest
from app.infrastructure.storage.s3_storage import verify_checksum


class PromotionValidationError(ValueError):
    """Raised when candidate model fails pre-promotion validation gates."""


@dataclass(frozen=True)
class ValidationResult:
    """Result of candidate model validation."""

    is_valid: bool
    rejection_reason: Optional[str] = None
    feature_count: int = 0
    dataset_mode: Optional[str] = None
    model_version: Optional[str] = None


def validate_candidate_for_promotion(
    manifest: ModelManifest,
    model_file_path: Path,
) -> ValidationResult:
    """Perform pre-promotion validation in an isolated step before activation.

    Args:
        manifest: Candidate model version metadata manifest.
        model_file_path: Local path to the downloaded booster artifact.

    Returns:
        ValidationResult with validation status and diagnostic failure details.
    """
    if not model_file_path.is_file():
        return ValidationResult(
            is_valid=False,
            rejection_reason=f"Candidate artifact file missing at {model_file_path}",
        )

    # 1. Verify Checksum Integrity
    if not verify_checksum(model_file_path, manifest.checksum_sha256):
        return ValidationResult(
            is_valid=False,
            rejection_reason=(
                f"Checksum validation failed: file digest does not match manifest "
                f"({manifest.checksum_sha256})"
            ),
        )

    # 2. Isolated Booster Load
    booster = xgb.Booster()
    try:
        booster.load_model(str(model_file_path))
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            rejection_reason=f"Corrupted model artifact; failed to load XGBoost booster: {exc}",
        )

    # 3. Canonical 6-Feature Schema & Order Validation
    actual_features = booster.feature_names
    expected_features = list(FEATURE_NAMES)
    if (
        booster.num_features() != len(expected_features)
        or actual_features is None
        or list(actual_features) != expected_features
    ):
        return ValidationResult(
            is_valid=False,
            rejection_reason=(
                f"Canonical feature contract mismatch. Expected {expected_features}; "
                f"model has {actual_features}"
            ),
            feature_count=booster.num_features(),
        )

    # 4. Environment & Provenance Safeguard
    dataset_mode = booster.attr("dataset_mode") or manifest.dataset_mode
    if dataset_mode not in {"real", "synthetic"}:
        return ValidationResult(
            is_valid=False,
            rejection_reason="Missing or invalid dataset_mode metadata attribute",
            dataset_mode=dataset_mode,
        )

    if (
        settings.ENVIRONMENT == EnvironmentType.PRODUCTION
        and dataset_mode != "real"
    ):
        return ValidationResult(
            is_valid=False,
            rejection_reason=(
                "Production policy violation: models trained on synthetic data "
                "cannot be promoted to production"
            ),
            dataset_mode=dataset_mode,
        )

    # 5. Smoke Inference Verification
    try:
        smoke_input = np.asarray(
            [[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32
        )
        dmatrix = xgb.DMatrix(smoke_input, feature_names=expected_features)
        prediction = np.asarray(booster.predict(dmatrix)).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            return ValidationResult(
                is_valid=False,
                rejection_reason="Smoke inference returned non-finite or empty prediction",
            )
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            rejection_reason=f"Smoke test inference failed: {exc}",
        )

    logger.info(
        "Candidate model passed all pre-promotion validation gates",
        model_version=manifest.model_version,
        dataset_mode=dataset_mode,
        checksum_sha256=manifest.checksum_sha256,
    )
    return ValidationResult(
        is_valid=True,
        feature_count=len(expected_features),
        dataset_mode=dataset_mode,
        model_version=manifest.model_version,
    )
