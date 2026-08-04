"""Unit tests for service health endpoint."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds properly."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "spotq-eta-service"


@patch("app.presentation.health.router.redis_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.mongo_manager.ping", new_callable=AsyncMock)
@patch("app.presentation.health.router.model_loader.is_healthy")
def test_healthz_endpoint(mock_model, mock_mongo_ping, mock_redis_ping):
    """Verify healthz diagnostics endpoint structure."""
    mock_redis_ping.return_value = (True, 1.25)
    mock_mongo_ping.return_value = (True, 2.50)
    mock_model.return_value = True

    response = client.get("/healthz")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dependencies"]["redis_feature_store"]["status"] == "connected"
    assert data["dependencies"]["mongodb_buffer"]["status"] == "connected"