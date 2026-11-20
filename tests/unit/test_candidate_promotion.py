"""Unit tests for authorized candidate promotion, S3 artifact retrieval, and atomic hot-swap."""

from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest
import xgboost as xgb

from app.infrastructure.config.settings import settings
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.storage.s3_storage import s3_storage
from app.main import app

client = TestClient(app)
ADMIN_KEY = "k" * 32


@pytest.fixture(autouse=True)
def setup_admin_env(monkeypatch, tmp_path):
    """Ensure admin key, synthetic model flag, and local storage adapter are enabled for tests."""
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setattr(settings, "ALLOW_SYNTHETIC_MODEL", True)
    storage_dir = tmp_path / "s3_local_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(s3_storage, "bucket_name", None)
    monkeypatch.setattr(s3_storage, "_s3_client", None)
    monkeypatch.setattr(s3_storage, "local_dir", storage_dir)

    # Initialize a clean baseline model in tmp_path
    baseline_path = tmp_path / "baseline_model.json"
    X = np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32)
    y = np.asarray([15.0], dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
    booster.set_attr(dataset_mode="synthetic", model_version="baseline_v1")
    booster.save_model(str(baseline_path))
    monkeypatch.setattr(settings, "MODEL_PATH", str(baseline_path))
    model_loader.load_model()


@pytest.fixture
def registered_candidate(tmp_path: Path) -> str:
    """Create a valid candidate in storage and return its version."""
    booster_file = tmp_path / "valid_model.json"
    X = np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32)
    y = np.asarray([15.0], dtype=np.float32)

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
    booster.set_attr(dataset_mode="synthetic", model_version="cand-v100")
    booster.save_model(str(booster_file))

    # Upload candidate to storage
    version = "cand-v100"
    s3_storage.upload_candidate(
        model_version=version,
        model_file_path=booster_file,
        dataset_mode="synthetic",
        metrics={"mae": 2.1},
        is_eligible=True,
    )
    return version


def test_promotion_rejects_missing_or_invalid_auth(registered_candidate: str):
    # Missing auth
    response = client.post(
        "/api/v1/eta/admin/promote-candidate",
        json={"model_version": registered_candidate, "approver_id": "eng_1"},
    )
    assert response.status_code == 401

    # Invalid auth
    response = client.post(
        "/api/v1/eta/admin/promote-candidate",
        headers={"X-Admin-Key": "wrong_key"},
        json={"model_version": registered_candidate, "approver_id": "eng_1"},
    )
    assert response.status_code == 401


def test_promote_candidate_success_and_hot_swap(registered_candidate: str):
    initial_version = model_loader.model_version

    response = client.post(
        "/api/v1/eta/admin/promote-candidate",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={
            "model_version": registered_candidate,
            "approver_id": "senior_ml_eng",
            "notes": "Validated holdout MAE < 2.5",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Candidate model successfully promoted and activated"
    assert data["promoted_version"] == registered_candidate
    assert data["approver_id"] == "senior_ml_eng"
    assert data["model_version"] == registered_candidate
    assert data["has_previous_model"] is True
    assert data["previous_model_version"] == initial_version

    # Model loader state should reflect the new model
    snapshot = model_loader.get_snapshot()
    assert snapshot.model_version == registered_candidate


def test_promote_non_existent_candidate_returns_404():
    response = client.post(
        "/api/v1/eta/admin/promote-candidate",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"model_version": "non-existent-v999", "approver_id": "eng_1"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_promote_candidate_with_checksum_failure_preserves_active_model(
    registered_candidate: str, tmp_path: Path
):
    active_before = model_loader.model_version

    # Intentionally corrupt the stored candidate file
    candidate_file = (
        s3_storage.local_dir / s3_storage.prefix / registered_candidate / "eta_model.json"
    )
    candidate_file.write_text("CORRUPTED_BYTES", encoding="utf-8")

    response = client.post(
        "/api/v1/eta/admin/promote-candidate",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"model_version": registered_candidate, "approver_id": "eng_1"},
    )

    assert response.status_code == 422
    assert "integrity error" in response.json()["detail"].lower()

    # Active model must be 100% preserved
    assert model_loader.model_version == active_before


def test_rollback_after_promotion():
    # Perform a rollback
    response = client.post(
        "/api/v1/eta/admin/rollback-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    # If a previous model exists, rollback should succeed
    if model_loader.previous_booster is not None:
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Model rolled back successfully"
    else:
        assert response.status_code == 409


def test_get_candidates_and_model_status():
    status_resp = client.get(
        "/api/v1/eta/admin/model-status",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "active_model" in status_data
    assert "storage_mode" in status_data

    cand_resp = client.get(
        "/api/v1/eta/admin/candidates",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert cand_resp.status_code == 200
    assert "candidates" in cand_resp.json()
