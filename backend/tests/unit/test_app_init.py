"""Unit tests for application initialization and configuration."""

from fastapi import FastAPI

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import setup_logging
from backend.app.main import create_app


def test_create_app_initialization() -> None:
    """Verify that create_app initializes a valid FastAPI instance with correct metadata."""
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "VisionGuard"
    assert app.version == "0.1.0"
    assert "offline-first" in app.description.lower()


def test_create_app_with_custom_settings() -> None:
    """Verify that create_app accepts customized settings overrides."""
    custom = Settings(
        app_name="VisionGuard-Custom",
        app_version="0.2.0-alpha",
        env="test",
        log_level="DEBUG",
    )
    app = create_app(custom_settings=custom)
    assert app.title == "VisionGuard-Custom"
    assert app.version == "0.2.0-alpha"


def test_default_settings_values() -> None:
    """Verify default configuration values and security constraints."""
    settings = get_settings()
    assert settings.app_name == "VisionGuard"
    assert settings.app_version == "0.1.0"
    assert settings.offline_mode is True
    assert settings.max_upload_mb > 0
    assert settings.max_image_pixels > 0
    assert settings.database_url.startswith("sqlite")


def test_logging_setup() -> None:
    """Verify setup_logging configures the logger properly without exceptions."""
    logger = setup_logging("INFO")
    assert logger.name == "visionguard"
    assert logger.level > 0
