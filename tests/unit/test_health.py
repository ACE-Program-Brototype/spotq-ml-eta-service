"""Unit tests for spotq-eta-service health endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify root health check returns status 200 and expected JSON payload."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "spotq-eta-service"
    assert data["status"] == "running"