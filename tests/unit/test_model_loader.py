
"""Tests for ETA model schema and training-data provenance safeguards."""

from pathlib import Path

import xgboost as xgb

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.loader import ModelLoader


INCOMPATIBLE_FEATURE_NAMES = (
    "feature_one",
    "feature_two",
    "feature_three",
)


def create_test_artifact(
    path: Path,
    *,
    feature_names: tuple[str, ...],
    dataset_mode: str | None = None,
) -> Path:
    """Create a small deterministic XGBoost artifact for loader tests."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        [float(row + column) for column in range(len(feature_names))]
        for row in range(1, 4)
    ]
    labels = [10.0, 20.0, 30.0]

    training_data = xgb.DMatrix(
        rows,
        label=labels,
        feature_names=list(feature_names),
    )

    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 1,
            "eta": 0.1,
            "nthread": 1,
            "seed": 42,
        },
        training_data,
        num_boost_round=1,
    )
    if dataset_mode is not None:
        booster.set_attr(dataset_mode=dataset_mode)
    booster.save_model(str(path))

    return path


def configure_loader(
    monkeypatch,
    *,
    environment: EnvironmentType,
    allow_synthetic: bool,
    model_path: Path,
) -> ModelLoader:
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "ALLOW_SYNTHETIC_MODEL", allow_synthetic)
    monkeypatch.setattr(settings, "MODEL_PATH", str(model_path))
    return ModelLoader()


def test_synthetic_candidate_rejected_by_default(tmp_path, monkeypatch):
    artifact = create_test_artifact(
        tmp_path / "synthetic_model.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="synthetic",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=False,
        model_path=artifact,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "Synthetic model rejected" in loader.failure_reason


def test_synthetic_candidate_allowed_only_with_development_opt_in(
    tmp_path, monkeypatch
):
    artifact = create_test_artifact(
        tmp_path / "synthetic_model.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="synthetic",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is True
    assert loader.status == "loaded_synthetic"
    assert loader.is_healthy() is True


def test_production_rejects_synthetic_candidate_even_if_flag_is_set(
    tmp_path, monkeypatch
):
    artifact = create_test_artifact(
        tmp_path / "synthetic_model.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="synthetic",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.PRODUCTION,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"

def test_model_without_dataset_mode_metadata_is_rejected(
    tmp_path, monkeypatch
):
    artifact = create_test_artifact(
        tmp_path / "missing_metadata.json",
        feature_names=FEATURE_NAMES,
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "dataset_mode metadata" in loader.failure_reason


def test_model_with_invalid_dataset_mode_metadata_is_rejected(
    tmp_path, monkeypatch
):
    artifact = create_test_artifact(
        tmp_path / "invalid_metadata.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="unknown",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "dataset_mode metadata" in loader.failure_reason


def test_active_model_with_incompatible_schema_is_rejected(
    tmp_path, monkeypatch
):
    artifact = create_test_artifact(
        tmp_path / "incompatible_model.json",
        feature_names=INCOMPATIBLE_FEATURE_NAMES,
        dataset_mode="real",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "feature schema mismatch" in loader.failure_reason



def test_failed_reload_preserves_active_model(tmp_path, monkeypatch):
    """Keep the active model when a replacement artifact is missing."""
    artifact = create_test_artifact(
        tmp_path / "valid_model.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="real",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=artifact,
    )

    assert loader.load_model() is True
    active_model = loader.booster
    active_version = loader.model_version

    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "missing.json"))

    assert loader.load_model() is False
    assert loader.booster is active_model
    assert loader.model_version == active_version
    assert loader.is_healthy() is True


def test_rollback_restores_previous_model(tmp_path, monkeypatch):
    """Restore the previous model after a successful replacement."""
    first = create_test_artifact(
        tmp_path / "model_v1.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="real",
    )
    second = create_test_artifact(
        tmp_path / "model_v2.json",
        feature_names=FEATURE_NAMES,
        dataset_mode="real",
    )
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=first,
    )

    assert loader.load_model() is True
    assert loader.model_version == "model_v1"

    monkeypatch.setattr(settings, "MODEL_PATH", str(second))
    assert loader.load_model() is True
    assert loader.model_version == "model_v2"

    assert loader.rollback_model() is True
    assert loader.model_version == "model_v1"
    assert loader.is_healthy() is True
