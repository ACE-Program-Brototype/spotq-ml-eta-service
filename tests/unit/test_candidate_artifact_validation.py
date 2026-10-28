"""Tests for candidate artifact validation and production isolation (AC5)."""

from pathlib import Path
import pytest
import xgboost as xgb

from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.loader import model_loader
from scripts.train_eta_model import (
    CandidateVerificationError,
    verify_candidate_artifact,
)


def create_candidate_booster(
    path: Path,
    feature_names: tuple[str, ...],
    dataset_mode: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[float(i + j) for j in range(len(feature_names))] for i in range(5)]
    labels = [10.0, 15.0, 20.0, 25.0, 30.0]

    dtrain = xgb.DMatrix(rows, label=labels, feature_names=list(feature_names))
    booster = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.1},
        dtrain,
        num_boost_round=2,
    )
    booster.set_attr(
        dataset_mode=dataset_mode,
        feature_contract_version="v1",
        model_version="test_candidate",
    )
    booster.save_model(str(path))
    return path


def test_valid_candidate_verification_succeeds(tmp_path):
    artifact_path = create_candidate_booster(
        tmp_path / "valid_candidate.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="real",
    )

    verify_candidate_artifact(artifact_path, expected_dataset_mode="real")


def test_candidate_verification_rejects_schema_mismatch(tmp_path):
    bad_features = ("f1", "f2", "f3")
    artifact_path = create_candidate_booster(
        tmp_path / "bad_schema.json",
        feature_names=bad_features,
        dataset_mode="real",
    )

    with pytest.raises(CandidateVerificationError, match="schema mismatch"):
        verify_candidate_artifact(artifact_path, expected_dataset_mode="real")


def test_candidate_verification_rejects_dataset_mode_mismatch(tmp_path):
    artifact_path = create_candidate_booster(
        tmp_path / "synthetic_candidate.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="synthetic",
    )

    with pytest.raises(
        CandidateVerificationError, match="dataset_mode metadata mismatch"
    ):
        verify_candidate_artifact(artifact_path, expected_dataset_mode="real")


def test_candidate_generation_does_not_mutate_active_production_model(tmp_path):
    """Verify AC5: Candidate generation does not overwrite or activate the production model."""
    initial_snapshot = model_loader.get_snapshot()

    # Generate and verify candidate in a separate directory
    candidate_path = create_candidate_booster(
        tmp_path / "candidates" / "eta_model_candidate.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="real",
    )
    verify_candidate_artifact(candidate_path, expected_dataset_mode="real")

    # Production state must remain completely identical
    after_snapshot = model_loader.get_snapshot()
    assert initial_snapshot.booster is after_snapshot.booster
    assert initial_snapshot.status == after_snapshot.status
    assert initial_snapshot.model_version == after_snapshot.model_version
    assert initial_snapshot.artifact_id == after_snapshot.artifact_id
    assert initial_snapshot.loaded_at == after_snapshot.loaded_at
