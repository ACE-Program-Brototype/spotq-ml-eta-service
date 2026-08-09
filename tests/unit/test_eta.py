from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_eta_predict_endpoint():
    """Verify GET /api/v1/eta/predict returns correct schema and values."""
    response = client.get(
        "/api/v1/eta/predict",
        params={
            "restaurant_id": "c3b9a782-9011-4712-8f1d-2395a123bc45",
            "total_people_ahead": 10,
            "party_size": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["restaurant_id"] == "c3b9a782-9011-4712-8f1d-2395a123bc45"
    assert isinstance(data["estimated_wait_minutes"], float)
    assert isinstance(data["confidence_score"], float)
    assert "timestamp" in data


def test_eta_ingest_endpoint():
    """Verify POST /api/v1/eta/ingest accepts queue payload successfully."""
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
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "message" in data