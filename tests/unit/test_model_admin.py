
"""Tests for authenticated model administration endpoints."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.infrastructure.config.settings import settings
from app.infrastructure.ml_models.loader import model_loader
from app.main import app

client = TestClient(app)
ADMIN_KEY = "a" * 32


def active_model_snapshot():
    """Return safe metadata representing an active model."""
    return SimpleNamespace(
        status="loaded",
        model_version="eta_model_v1",
        artifact_id="real",
        loaded_at="2026-10-23T00:00:00+00:00",
    )


def test_reload_rejects_missing_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)

    response = client.post("/api/v1/eta/admin/reload-model")

    assert response.status_code == 401


def test_rollback_rejects_invalid_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)

    response = client.post(
        "/api/v1/eta/admin/rollback-model",
        headers={"X-Admin-Key": "wrong"},
    )

    assert response.status_code == 401


def test_reload_fails_closed_when_admin_key_is_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", None)

    response = client.post(
        "/api/v1/eta/admin/reload-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    assert response.status_code == 503


def test_successful_reload_returns_model_metadata(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setattr(model_loader, "load_model", lambda is_reload=False: True)
    monkeypatch.setattr(
        model_loader, "get_snapshot", active_model_snapshot
    )

    response = client.post(
        "/api/v1/eta/admin/reload-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Model reloaded successfully"
    assert body["status"] == "loaded"
    assert body["model_version"] == "eta_model_v1"
    assert body["artifact_id"] == "real"
    assert "path" not in body


def test_failed_reload_returns_503_and_preserves_active_model(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setattr(model_loader, "load_model", lambda is_reload=False: False)
    monkeypatch.setattr(
        model_loader, "get_snapshot", active_model_snapshot
    )

    response = client.post(
        "/api/v1/eta/admin/reload-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    assert response.status_code == 503
    assert "existing model was preserved" in response.json()["detail"]


def test_successful_rollback_returns_model_metadata(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setattr(model_loader, "rollback_model", lambda: True)
    monkeypatch.setattr(
        model_loader, "get_snapshot", active_model_snapshot
    )

    response = client.post(
        "/api/v1/eta/admin/rollback-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Model rolled back successfully"
    assert body["model_version"] == "eta_model_v1"


def test_rollback_returns_409_when_no_previous_model_exists(monkeypatch):
    monkeypatch.setattr(settings, "ETA_MODEL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setattr(model_loader, "rollback_model", lambda: False)

    response = client.post(
        "/api/v1/eta/admin/rollback-model",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    assert response.status_code == 409
    assert "No previous model" in response.json()["detail"]
