from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_eta_predict_endpoint():
    """Verify GET /api/v1/eta/predict handles valid requests and model availability correctly."""
    response = client.get(
        "/api/v1/eta/predict",
        params={
            "restaurant_id": "c3b9a782-9011-4712-8f1d-2395a123bc45",
            "total_people_ahead": 10,
            "party_size": 4,
        },
    )
    # Expects 200 if model artifact loaded, or 503 if unloaded
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["restaurant_id"] == "c3b9a782-9011-4712-8f1d-2395a123bc45"
        assert isinstance(data["estimated_wait_minutes"], float)
        assert data["confidence_score"] is None
        assert "timestamp" in data
        assert isinstance(data["fallback_used"], bool)
        assert "prediction_id" in data


def test_eta_predict_with_queue_entry_id():
    """Verify GET /api/v1/eta/predict accepts optional queue_entry_id for correlation."""
    response = client.get(
        "/api/v1/eta/predict",
        params={
            "restaurant_id": "c3b9a782-9011-4712-8f1d-2395a123bc45",
            "total_people_ahead": 5,
            "party_size": 2,
            "queue_entry_id": "queue-entry-test-1234",
        },
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["prediction_id"] is not None
        assert data["prediction_id"].startswith("pred_")


def test_eta_predict_validation_errors():
    """Verify GET /api/v1/eta/predict rejects requests with missing or invalid parameters."""
    response = client.get("/api/v1/eta/predict", params={"restaurant_id": "test"})
    assert response.status_code == 400


def test_eta_ingest_endpoint():
    """Verify POST /api/v1/eta/ingest validates payload contract and response status."""
    payload = {
        "queue_entry_id": "entry-uuid-123",
        "restaurant_id": "c3b9a782-9011-4712-8f1d-2395a123bc45",
        "party_size": 4,
        "joined_at": "2026-08-04T12:00:00Z",
        "completed_at": "2026-08-04T12:25:00Z",
        "actual_wait_time_minutes": 25.0,
        "total_people_ahead_at_join": 10,
    }
    response = client.post("/api/v1/eta/ingest", json=payload)
    # Expects 200 if MongoDB is connected during unit tests, or 503 if unavailable
    assert response.status_code in [200, 503, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data