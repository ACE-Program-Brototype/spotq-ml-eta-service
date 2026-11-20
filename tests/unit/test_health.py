"""Unit tests for service health endpoint."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from types import SimpleNamespace

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds properly."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "spotq-eta-service"


@patch("app.presentation.health.router.redis_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.mongo_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.model_loader.get_snapshot")
def test_healthz_endpoint(mock_snapshot, mock_mongo_ping, mock_redis_ping):
    """Verify healthz diagnostics endpoint structure."""
    mock_redis_ping.return_value = (True, 1.25)
    mock_mongo_ping.return_value = (True, 2.50)
    mock_snapshot.return_value = SimpleNamespace(
        booster=object(),
        status="loaded",
        model_version="test-model-v1",
        artifact_id="test-artifact",
        loaded_at="2026-10-22T00:00:00+00:00",
        failure_reason=None,
    )

    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["dependencies"]["redis_feature_store"]["status"] == "connected"
    assert data["dependencies"]["mongodb_buffer"]["status"] == "connected"
    assert data["dependencies"]["xgboost_model_artifact"]["status"] == "loaded"


@patch("app.presentation.health.router.redis_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.mongo_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.model_loader.get_snapshot")
def test_readiness_allows_redis_degradation(
    mock_snapshot, mock_mongo_ping, mock_redis_ping
):
    """Remain ready when Redis is down but fallback is available."""
    mock_redis_ping.return_value = (False, None)
    mock_mongo_ping.return_value = (True, 1.0)
    mock_snapshot.return_value = SimpleNamespace(
        booster=object(),
        status="loaded",
        model_version="test-model-v1",
        artifact_id="test-artifact",
        loaded_at="2026-10-22T00:00:00+00:00",
        failure_reason=None,
    )

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["redis"] == "degraded_fallback_available"
    assert response.json()["model"] == "loaded"


def test_liveness_does_not_depend_on_external_services():
    """Report process liveness without checking external dependencies."""
    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
