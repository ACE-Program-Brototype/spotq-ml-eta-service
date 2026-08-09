"""Integration and unit tests for application lifespan and entry point."""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def mock_infrastructure():
    """Mock external dependency connections for lifespan execution."""
    with patch("app.main.redis_manager.connect", new_callable=AsyncMock) as mock_redis_conn, \
         patch("app.main.redis_manager.disconnect", new_callable=AsyncMock) as mock_redis_disc, \
         patch("app.main.mongo_manager.connect", new_callable=AsyncMock) as mock_mongo_conn, \
         patch("app.main.mongo_manager.disconnect", new_callable=AsyncMock) as mock_mongo_disc, \
         patch("app.main.model_loader.load_model", return_value=True) as mock_load_model:
        yield {
            "redis_conn": mock_redis_conn,
            "redis_disc": mock_redis_disc,
            "mongo_conn": mock_mongo_conn,
            "mongo_disc": mock_mongo_disc,
            "load_model": mock_load_model,
        }


def test_lifespan_startup_and_shutdown(mock_infrastructure):
    """Verify connections open on context entry and close on exit."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

        mock_infrastructure["redis_conn"].assert_called_once()
        mock_infrastructure["mongo_conn"].assert_called_once()
        mock_infrastructure["load_model"].assert_called_once()

    mock_infrastructure["redis_disc"].assert_called_once()
    mock_infrastructure["mongo_disc"].assert_called_once()


def test_root_endpoint(mock_infrastructure):
    """Test standard JSON response from root router."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        payload = response.json()
        assert payload["service"] == "spotq-eta-service"
        assert "environment" in payload


def test_metrics_endpoint_exposed(mock_infrastructure):
    """Verify Prometheus metrics route is mounted correctly."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "").lower()
        assert "http_requests_total" in response.text or "python_gc_objects_collected_total" in response.text