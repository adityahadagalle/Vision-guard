"""Configuration parameters and guardrails for dataset ingestion and quarantine.

Enforces defensive thresholds against denial-of-service, decompression bombs,
path traversals, and excessive resource consumption during archive handling.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class IngestionConfig(BaseModel):
    """Configuration settings for secure archive ingestion, inspection, and snapshot creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Upload & Archive Constraints
    max_upload_bytes: int = Field(
        default=250 * 1024 * 1024,  # 250 MB
        description="Maximum permissible compressed archive upload size.",
    )
    max_uncompressed_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,  # 10 GB
        description="Maximum permissible total uncompressed size across all archive entries.",
    )
    max_file_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,  # 2 GB
        description="Maximum permissible uncompressed size for any single file entry.",
    )
    max_file_count: int = Field(
        default=50_000,
        description="Maximum number of files allowed in an archive (inode exhaustion defense).",
    )
    max_compression_ratio: float = Field(
        default=10.0,
        description="Maximum allowed uncompressed-to-compressed ratio (decompression bomb defense).",
    )

    # Image Decompression & Dimension Constraints
    max_image_pixels: int = Field(
        default=40_000_000,  # 40M pixels (e.g. 8000x5000)
        description="Pillow decompression bomb threshold (MAX_IMAGE_PIXELS).",
    )
    max_image_dimension: int = Field(
        default=10_000,
        description="Maximum permissible image width or height in pixels.",
    )

    # Worker Execution & Timeout Constraints
    worker_timeout_seconds: int = Field(
        default=60,
        description="Hard timeout in seconds before terminating isolated extraction worker subprocess.",
    )
    chunk_size_bytes: int = Field(
        default=64 * 1024,  # 64 KB
        description="Streaming buffer size for deterministic SHA-256 computation.",
    )

    # Directory Paths
    quarantine_dir: Path = Field(
        default=Path("./data/quarantine"),
        description="Root directory for storing untrusted raw uploaded archives.",
    )
    staging_dir: Path = Field(
        default=Path("./data/staging"),
        description="Ephemeral directory for isolated archive extraction and validation.",
    )
    snapshots_dir: Path = Field(
        default=Path("./data/processed/snapshots"),
        description="Content-addressed root directory for sealed, immutable snapshots.",
    )
