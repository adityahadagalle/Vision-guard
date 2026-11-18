"""Domain models for content-addressed assets and immutable snapshots.

Encapsulates dataset snapshot identities, manifest records, and strict
domain-level path and cryptographic digest validations.
"""

import posixpath
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.domain.entities.enums import AssetType

_SHA256_HEX_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{3,128}$")
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def validate_sha256_hex(value: str) -> str:
    """Validates that a string is a strict 64-character lowercase hexadecimal SHA-256 digest."""
    if not isinstance(value, str):
        raise ValueError("SHA-256 digest must be a string.")
    if not _SHA256_HEX_PATTERN.match(value):
        raise ValueError(
            "Invalid SHA-256 digest format: must be exactly 64 lowercase hexadecimal characters."
        )
    return value


def validate_timezone_aware(dt: datetime) -> datetime:
    """Validates that a datetime object is timezone-aware."""
    if not isinstance(dt, datetime):
        raise ValueError("Timestamp must be a datetime instance.")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Datetime must be timezone-aware (UTC recommended).")
    return dt


class ManifestEntry(BaseModel):
    """Represents an individual file entry within an immutable asset snapshot.

    Enforces path safety invariants: rejects absolute paths, parent directory
    traversals (..), Windows Alternate Data Streams, and reserved device names.
    Applies deterministic POSIX path normalization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    sha256: str
    size_bytes: int
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        return validate_sha256_hex(v)

    @field_validator("size_bytes")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"File size cannot be negative, got {v}.")
        return v

    @field_validator("relative_path")
    @classmethod
    def _validate_and_normalize_path(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("relative_path must be a string.")

        clean = v.strip()
        if not clean:
            raise ValueError("relative_path cannot be empty or whitespace-only.")

        if "\0" in clean:
            raise ValueError("relative_path cannot contain null bytes.")

        # Reject Windows Alternate Data Streams or drive letters
        if ":" in clean:
            raise ValueError(
                "relative_path cannot contain colons (Windows Alternate Data Streams or drive letters)."
            )

        # Reject absolute paths (Unix or Windows backslash)
        if clean.startswith("/") or clean.startswith("\\"):
            raise ValueError(f"Absolute paths are forbidden: '{v}'.")

        # Normalize backslashes to forward slashes for cross-platform consistency
        normalized = clean.replace("\\", "/")

        # Strip redundant leading './' prefixes
        while normalized.startswith("./"):
            normalized = normalized[2:]

        # Split components and inspect each segment
        segments = [seg for seg in normalized.split("/") if seg and seg != "."]
        if not segments:
            raise ValueError("relative_path cannot resolve to an empty path.")

        for segment in segments:
            if segment == "..":
                raise ValueError(f"Path traversal component ('..') is strictly prohibited: '{v}'.")

            # Check for Windows reserved device names (e.g., CON, PRN, AUX, COM1, etc.)
            base_name = segment.split(".")[0].lower()
            if base_name in _WINDOWS_DEVICE_NAMES:
                raise ValueError(f"Reserved Windows device name prohibited in path: '{segment}'.")

        # Build clean POSIX path
        result = posixpath.normpath("/".join(segments))
        if result == "." or result.startswith("../"):
            raise ValueError(f"Invalid normalized path: '{v}'.")

        return result


class AssetSnapshot(BaseModel):
    """Represents an immutable, content-addressed asset snapshot.

    An asset snapshot captures the complete state of a validated dataset or
    artifact collection, anchored by the SHA-256 digest of its canonical manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    asset_id: str
    asset_type: AssetType
    manifest_sha256: str
    created_at: datetime
    file_count: int
    total_size_bytes: int
    schema_version: str = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_id", "asset_id")
    @classmethod
    def _validate_identifier(cls, v: str, info) -> str:
        if not isinstance(v, str) or not _IDENTIFIER_PATTERN.match(v):
            raise ValueError(
                f"Field '{info.field_name}' must be an alphanumeric identifier (3-128 chars, "
                f"allowing hyphens, underscores, and dots), got '{v}'."
            )
        return v

    @field_validator("manifest_sha256")
    @classmethod
    def _validate_manifest_sha256(cls, v: str) -> str:
        return validate_sha256_hex(v)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        return validate_timezone_aware(v)

    @field_validator("file_count")
    @classmethod
    def _validate_file_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"file_count cannot be negative, got {v}.")
        return v

    @field_validator("total_size_bytes")
    @classmethod
    def _validate_total_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"total_size_bytes cannot be negative, got {v}.")
        return v

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("schema_version cannot be empty.")
        return v.strip()
