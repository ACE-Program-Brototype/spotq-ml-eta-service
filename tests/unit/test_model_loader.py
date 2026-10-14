"""Tests for ETA model schema and training-data provenance safeguards."""

from pathlib import Path

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.ml_models.loader import ModelLoader


CANDIDATE_PATH = (
    Path("app/infrastructure/ml_models/artifacts/candidates/eta_model_candidate.json")
)
ACTIVE_PATH = Path("app/infrastructure/ml_models/artifacts/eta_model_v1.json")


def configure_loader(monkeypatch, *, environment, allow_synthetic, model_path):
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "ALLOW_SYNTHETIC_MODEL", allow_synthetic)
    monkeypatch.setattr(settings, "MODEL_PATH", str(model_path))
    return ModelLoader()


def test_synthetic_candidate_rejected_by_default(monkeypatch):
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=False,
        model_path=CANDIDATE_PATH,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "Synthetic model rejected" in loader.failure_reason


def test_synthetic_candidate_allowed_only_with_development_opt_in(monkeypatch):
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=CANDIDATE_PATH,
    )

    assert loader.load_model() is True
    assert loader.status == "loaded_synthetic"
    assert loader.is_healthy() is True


def test_production_rejects_synthetic_candidate_even_if_flag_is_set(monkeypatch):
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.PRODUCTION,
        allow_synthetic=True,
        model_path=CANDIDATE_PATH,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"


def test_active_model_with_incompatible_schema_is_rejected(monkeypatch):
    loader = configure_loader(
        monkeypatch,
        environment=EnvironmentType.DEVELOPMENT,
        allow_synthetic=True,
        model_path=ACTIVE_PATH,
    )

    assert loader.load_model() is False
    assert loader.booster is None
    assert loader.status == "incompatible"
    assert "feature schema mismatch" in loader.failure_reason
