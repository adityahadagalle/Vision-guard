"""Application configuration management using Pydantic Settings.

Reads configuration from environment variables (prefixed with VISIONGUARD_)
and optional local .env files. Never hardcode credentials or secrets here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VisionGuard application settings."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Metadata
    app_name: str = Field(default="VisionGuard", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    env: str = Field(
        default="development", description="Environment mode (development/production/test)"
    )

    # Server Binding
    host: str = Field(default="127.0.0.1", description="Server host interface")
    port: int = Field(default=8000, description="Server port")

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Storage & Persistence
    data_dir: Path = Field(
        default=Path("./data"), description="Path to local persistent data storage"
    )
    database_url: str = Field(
        default="sqlite:///./data/visionguard.db", description="SQLite database connection URL"
    )
    keys_dir: Path = Field(
        default=Path("data/keys"),
        description="Path to offline cryptographic key custody directory",
    )

    @property
    def KEYS_DIR(self) -> Path:  # noqa: N802
        """Uppercase alias for keys_dir."""
        return self.keys_dir

    # Security & Guardrails
    max_upload_mb: int = Field(default=250, description="Maximum allowed upload size in megabytes")
    max_model_upload_bytes: int = Field(
        default=500 * 1024 * 1024,
        description="Maximum allowed model checkpoint upload size in bytes (500 MB)",
    )
    global_quarantine_quota_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024,
        description="Global quarantine storage quota in bytes (5 GB)",
    )
    max_concurrent_uploads: int = Field(
        default=3,
        description="Maximum concurrent upload streams allowed across processes",
    )
    upload_reservation_timeout_seconds: int = Field(
        default=900,
        description="Upload reservation expiry lease in seconds (15 minutes)",
    )
    max_image_pixels: int = Field(
        default=40_000_000,
        description="Maximum image pixel threshold (decompression bomb protection)",
    )
    offline_mode: bool = Field(default=True, description="Strict offline mode enforcement flag")

    # Routing
    api_v1_prefix: str = Field(default="/api/v1", description="URL prefix for API version 1")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
