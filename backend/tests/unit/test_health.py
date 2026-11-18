"""Unit tests for the health check endpoint."""

from fastapi import status
from fastapi.testclient import TestClient

from backend.app.schemas.health import HealthResponse


def test_health_endpoint_root_returns_200(client: TestClient) -> None:
    """Verify that GET /health returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK


def test_health_endpoint_root_json_structure(client: TestClient) -> None:
    """Verify that GET /health returns the exact expected JSON response structure."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data == {
        "status": "healthy",
        "application": "VisionGuard",
        "version": "0.1.0",
    }


def test_health_endpoint_validates_schema(client: TestClient) -> None:
    """Verify that the health response matches the Pydantic HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    health_obj = HealthResponse.model_validate(response.json())
    assert health_obj.status == "healthy"
    assert health_obj.application == "VisionGuard"
    assert health_obj.version == "0.1.0"


def test_health_endpoint_api_v1_prefix(client: TestClient) -> None:
    """Verify that GET /api/v1/health is also accessible and returns identical response."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "healthy",
        "application": "VisionGuard",
        "version": "0.1.0",
    }


def test_nonexistent_route_returns_404(client: TestClient) -> None:
    """Verify that requests to unregistered paths return HTTP 404 Not Found."""
    response = client.get("/api/v1/unregistered-path")
    assert response.status_code == status.HTTP_404_NOT_FOUND
