"""Configuration settings for PyTorch model ingestion and inspection.

Enforces resource limits, memory boundaries, execution timeouts,
and content-addressed storage locations for model artifacts.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelIntegrityConfig(BaseModel):
    """Configuration governing model artifact ingestion, inspection, and isolation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Ingestion constraints
    max_model_bytes: int = Field(
        default=500 * 1024 * 1024,  # 500 MB
        description="Maximum permissible model artifact size in bytes.",
    )
    chunk_size_bytes: int = Field(
        default=64 * 1024,  # 64 KB
        description="Chunk size for streaming reads and SHA-256 hashing.",
    )

    # Worker isolation & resource constraints
    worker_timeout_seconds: float = Field(
        default=30.0,
        description="Hard execution timeout for isolated model worker subprocess.",
    )
    max_tensor_count: int = Field(
        default=5000,
        description="Maximum permissible number of tensors in a state dict.",
    )
    max_parameter_count: int = Field(
        default=200_000_000,
        description="Maximum total parameter count in a candidate model.",
    )
    max_tensor_memory_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,  # 2 GB
        description="Maximum cumulative memory consumed by model tensors in bytes.",
    )

    # Storage paths
    quarantine_dir: Path = Field(
        default_factory=lambda: Path("storage/quarantine/models"),
        description="Path to staged/quarantined untrusted model uploads.",
    )
    staging_dir: Path = Field(
        default_factory=lambda: Path("storage/staging/models"),
        description="Path to temporary extraction and inspection scratch directory.",
    )
    models_dir: Path = Field(
        default_factory=lambda: Path("storage/models"),
        description="Path to content-addressed approved model artifact store.",
    )
    snapshots_dir: Path = Field(
        default_factory=lambda: Path("storage/snapshots"),
        description="Path to immutable asset snapshot directory.",
    )
