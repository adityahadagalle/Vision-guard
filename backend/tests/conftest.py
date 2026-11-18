"""Pytest fixtures for VisionGuard backend tests."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Provide isolated test configuration."""
    return Settings(
        app_name="VisionGuard",
        app_version="0.1.0",
        env="test",
        offline_mode=True,
        log_level="WARNING",
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """Create test application instance."""
    return create_app(custom_settings=test_settings)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create synchronous FastAPI test client."""
    return TestClient(app)
