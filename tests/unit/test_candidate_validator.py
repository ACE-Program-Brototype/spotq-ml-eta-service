"""Unit tests for candidate model pre-promotion validation."""

from pathlib import Path
import numpy as np
import pytest
import xgboost as xgb

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.registry import ModelManifest
from app.infrastructure.ml_models.validator import validate_candidate_for_promotion
from app.infrastructure.storage.s3_storage import compute_sha256


@pytest.fixture
def valid_candidate_booster(tmp_path: Path) -> tuple[Path, ModelManifest]:
    """Create a valid 6-feature candidate booster and manifest."""
    booster_path = tmp_path / "valid_candidate.json"
    X = np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32)
    y = np.asarray([15.0], dtype=np.float32)

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
    booster.set_attr(dataset_mode="real", model_version="cand-v1")
    booster.save_model(str(booster_path))

    checksum = compute_sha256(booster_path)
    manifest = ModelManifest(
        model_version="cand-v1",
        s3_uri=f"file://{booster_path}",
        checksum_sha256=checksum,
        dataset_mode="real",
        is_eligible=True,
    )
    return booster_path, manifest


def test_valid_candidate_passes_validation(valid_candidate_booster):
    booster_path, manifest = valid_candidate_booster
    result = validate_candidate_for_promotion(manifest, booster_path)
    assert result.is_valid is True
    assert result.feature_count == 6
    assert result.rejection_reason is None


def test_validator_rejects_checksum_mismatch(valid_candidate_booster):
    booster_path, manifest = valid_candidate_booster
    manifest.checksum_sha256 = "f" * 64  # Corrupt manifest checksum
    result = validate_candidate_for_promotion(manifest, booster_path)
    assert result.is_valid is False
    assert "Checksum validation failed" in result.rejection_reason


def test_validator_rejects_feature_schema_mismatch(tmp_path: Path):
    booster_path = tmp_path / "wrong_schema.json"
    # Create booster with wrong features
    X = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.asarray([10.0], dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y, feature_names=["f1", "f2", "f3"])
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
    booster.save_model(str(booster_path))

    manifest = ModelManifest(
        model_version="cand-bad-schema",
        s3_uri=f"file://{booster_path}",
        checksum_sha256=compute_sha256(booster_path),
        dataset_mode="real",
    )

    result = validate_candidate_for_promotion(manifest, booster_path)
    assert result.is_valid is False
    assert "Canonical feature contract mismatch" in result.rejection_reason


def test_validator_rejects_synthetic_model_in_production(valid_candidate_booster, monkeypatch):
    booster_path, manifest = valid_candidate_booster
    manifest.dataset_mode = "synthetic"

    # Save booster with synthetic attr
    booster = xgb.Booster()
    booster.load_model(str(booster_path))
    booster.set_attr(dataset_mode="synthetic")
    booster.save_model(str(booster_path))
    manifest.checksum_sha256 = compute_sha256(booster_path)

    # Set environment to production
    monkeypatch.setattr(settings, "ENVIRONMENT", EnvironmentType.PRODUCTION)

    result = validate_candidate_for_promotion(manifest, booster_path)
    assert result.is_valid is False
    assert "Production policy violation" in result.rejection_reason
