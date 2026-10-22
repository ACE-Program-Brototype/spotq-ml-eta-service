import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.presentation.predict import router as eta_router
from types import SimpleNamespace

client = TestClient(app)
VALID_PARAMS = {
    "restaurant_id": "restaurant-test",
    "total_people_ahead": 8,
    "party_size": 3,
}


def _mock_booster(prediction):
    booster = MagicMock()
    booster.predict.return_value = np.array([prediction])
    return booster


@pytest.fixture
def prediction_dependencies(monkeypatch):
    """Use a fake compatible model and deterministic cold-start history."""
    booster = _mock_booster(15.0)

    snapshot = SimpleNamespace(
        booster=booster,
        status="loaded",
        model_version="test-model-v1",
        artifact_id="test-artifact",
        loaded_at="2026-10-22T00:00:00+00:00",
        failure_reason=None,
    )

    monkeypatch.setattr(
        eta_router.model_loader,
        "get_snapshot",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        eta_router.model_loader,
        "is_healthy",
        lambda: True,
    )
    monkeypatch.setattr(
        eta_router,
        "get_historical_average",
        AsyncMock(return_value=(20.0, "cold_start")),
    )
    monkeypatch.setattr(
        eta_router,
        "_restaurant_timezone",
        lambda restaurant_id: "UTC",
    )

    return booster


def test_v1_endpoint_is_registered():
    response = client.get("/api/v1/eta/predict", params=VALID_PARAMS)
    # The real active model may be incompatible; registration must still work.
    assert response.status_code in (200, 503)


def test_v1_prediction_uses_cold_start_and_returns_contract(
    prediction_dependencies,
):
    response = client.get("/api/v1/eta/predict", params=VALID_PARAMS)

    assert response.status_code == 200
    data = response.json()
    assert data["restaurant_id"] == VALID_PARAMS["restaurant_id"]
    assert data["estimated_wait_minutes"] == 15.0
    assert data["confidence_score"] is None
    assert data["fallback_used"] is True
    assert data["timestamp"].endswith("+00:00")


def test_v1_prediction_clamps_high_output(prediction_dependencies, monkeypatch):
    prediction_dependencies.predict.return_value = np.array([250.0])

    response = client.get("/api/v1/eta/predict", params=VALID_PARAMS)

    assert response.status_code == 200
    assert response.json()["estimated_wait_minutes"] == 120.0


def test_v1_prediction_clamps_low_output(prediction_dependencies):
    prediction_dependencies.predict.return_value = np.array([-5.0])

    response = client.get("/api/v1/eta/predict", params=VALID_PARAMS)

    assert response.status_code == 200
    assert response.json()["estimated_wait_minutes"] == 1.0


@pytest.mark.parametrize("prediction", [math.nan, math.inf, -math.inf])
def test_v1_prediction_rejects_non_finite_output(
    prediction_dependencies,
    prediction,
):
    prediction_dependencies.predict.return_value = np.array([prediction])

    response = client.get("/api/v1/eta/predict", params=VALID_PARAMS)

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "eta_prediction_invalid_output"


def test_v1_prediction_rejects_invalid_query_parameters():
    response = client.get(
        "/api/v1/eta/predict",
        params={**VALID_PARAMS, "party_size": 0},
    )

    assert response.status_code == 400


def test_feature_schema_has_exactly_six_ordered_features():
    assert FEATURE_NAMES == (
        "party_size",
        "total_people_ahead",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "historical_average_wait",
    )
