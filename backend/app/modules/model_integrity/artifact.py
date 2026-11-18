"""Secure model artifact ingestion and snapshot generation.

Streams candidate PyTorch checkpoint files (.pt/.pth) into content-addressed
storage, enforcing size limits, computing streaming SHA-256 digests, and
anchoring artifacts to immutable AssetSnapshot entities without executing or
deserializing checkpoints in the API process.
"""

import hashlib
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.snapshot import (
    SnapshotManager,
    canonical_json_bytes,
)
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.exceptions import (
    EmptyModelFileError,
    InvalidFilenameError,
    InvalidModelExtensionError,
    InvalidModelIdError,
    ModelDigestMismatchError,
    ModelFileTooLargeError,
    PathTraversalError,
    SymlinkNotPermittedError,
)

_ALLOWED_EXTENSIONS = frozenset({".pt", ".pth"})
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
_RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def validate_model_id(model_id: str) -> str:
    """Validates model identifier against strict allowlist."""
    if not model_id or not model_id.strip():
        raise InvalidModelIdError("Model ID cannot be empty.")
    if not _MODEL_ID_PATTERN.match(model_id):
        raise InvalidModelIdError(
            f"Invalid model ID '{model_id}'. Must strictly match ^[a-zA-Z0-9_-]+$."
        )
    return model_id


def validate_model_filename(filename: str) -> str:
    """Strictly validates externally supplied candidate model filenames against path traversal."""
    if not filename or not filename.strip():
        raise InvalidFilenameError("Model filename cannot be empty.")

    if "\0" in filename:
        raise PathTraversalError("Filename contains forbidden NUL bytes.")

    if "/" in filename or "\\" in filename:
        raise PathTraversalError(f"Filename '{filename}' contains path separators.")

    if ".." in filename:
        raise PathTraversalError(
            f"Filename '{filename}' contains directory traversal sequence '..'."
        )

    if ":" in filename:
        raise PathTraversalError(
            f"Filename '{filename}' contains forbidden colon (drive prefix or alternate data stream)."
        )

    if filename.endswith(".") or filename.endswith(" "):
        raise PathTraversalError(f"Filename '{filename}' has trailing dots or spaces.")

    p = Path(filename)
    if p.is_absolute():
        raise PathTraversalError(f"Absolute filenames are forbidden: '{filename}'.")

    clean_ext = p.suffix.lower()
    if clean_ext not in _ALLOWED_EXTENSIONS:
        raise InvalidModelExtensionError(
            f"Unsupported model file extension '{clean_ext}'. Supported formats: {sorted(_ALLOWED_EXTENSIONS)}.",
            details={"filename": filename, "extension": clean_ext},
        )

    if not _SAFE_FILENAME_PATTERN.match(filename):
        raise InvalidFilenameError(f"Filename '{filename}' contains forbidden characters.")

    stem_upper = p.stem.upper()
    if stem_upper in _RESERVED_WINDOWS_NAMES:
        raise PathTraversalError(
            f"Filename '{filename}' uses reserved Windows device name '{stem_upper}'."
        )

    return filename


class ModelIngestionResult(BaseModel):
    """Result of secure model artifact ingestion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    filename: str
    sha256: str
    size_bytes: int
    storage_path: Path
    snapshot: AssetSnapshot
    snapshot_dir: Path


class ModelArtifactManager:
    """Manages the secure ingestion and snapshot creation of PyTorch model artifacts."""

    def __init__(self, config: ModelIntegrityConfig | None = None) -> None:
        self.config = config or ModelIntegrityConfig()
        self.models_dir = Path(self.config.models_dir).resolve()
        self.snapshots_dir = Path(self.config.snapshots_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_config = IngestionConfig(
            snapshots_dir=self.snapshots_dir,
            quarantine_dir=Path(self.config.quarantine_dir).resolve(),
            staging_dir=Path(self.config.staging_dir).resolve(),
        )

    def ingest_model(
        self,
        stream_or_path: Path | BinaryIO,
        filename: str,
        model_id: str | None = None,
        expected_sha256: str | None = None,
    ) -> ModelIngestionResult:
        """Streams and persists a candidate PyTorch model checkpoint.

        Args:
            stream_or_path: File path or open binary stream of candidate model.
            filename: Original filename (must end with .pt or .pth).
            model_id: Optional model identifier (generated if omitted).
            expected_sha256: Optional expected SHA-256 digest to verify.

        Returns:
            ModelIngestionResult containing artifact metadata and AssetSnapshot.

        Raises:
            PathTraversalError: If filename or model_id attempts path traversal.
            InvalidFilenameError: If filename violates naming rules.
            InvalidModelIdError: If model_id contains illegal characters.
            InvalidModelExtensionError: If filename extension is not .pt or .pth.
            EmptyModelFileError: If file is 0 bytes.
            ModelFileTooLargeError: If file exceeds max_model_bytes.
            ModelDigestMismatchError: If expected_sha256 is provided and does not match.
            SymlinkNotPermittedError: If model paths involve symlinks.
        """
        valid_filename = validate_model_filename(filename)

        # Staging file for streaming writes before content hash is known
        temp_target = self.models_dir / f".staging_{uuid.uuid4().hex[:12]}.tmp"

        hasher = hashlib.sha256()
        total_bytes = 0
        chunk_size = self.config.chunk_size_bytes

        try:
            with open(temp_target, "wb") as out_f:
                if isinstance(stream_or_path, (str, Path)):
                    src_path = Path(stream_or_path)
                    if not src_path.exists() or not src_path.is_file():
                        raise FileNotFoundError(f"Source model file '{src_path}' not found.")
                    with open(src_path, "rb") as in_f:
                        while chunk := in_f.read(chunk_size):
                            total_bytes += len(chunk)
                            if total_bytes > self.config.max_model_bytes:
                                raise ModelFileTooLargeError(
                                    f"Model artifact exceeds maximum allowed size of {self.config.max_model_bytes} bytes.",
                                    details={
                                        "max_bytes": self.config.max_model_bytes,
                                        "received_bytes": total_bytes,
                                    },
                                )
                            hasher.update(chunk)
                            out_f.write(chunk)
                else:
                    while chunk := stream_or_path.read(chunk_size):
                        total_bytes += len(chunk)
                        if total_bytes > self.config.max_model_bytes:
                            raise ModelFileTooLargeError(
                                f"Model artifact exceeds maximum allowed size of {self.config.max_model_bytes} bytes.",
                                details={
                                    "max_bytes": self.config.max_model_bytes,
                                    "received_bytes": total_bytes,
                                },
                            )
                        hasher.update(chunk)
                        out_f.write(chunk)

            if total_bytes == 0:
                raise EmptyModelFileError(
                    "Model artifact is empty (0 bytes).",
                    details={"filename": filename},
                )

            computed_sha256 = hasher.hexdigest()

            if expected_sha256 and expected_sha256.lower() != computed_sha256.lower():
                raise ModelDigestMismatchError(
                    f"Model artifact SHA-256 mismatch. Expected '{expected_sha256}', got '{computed_sha256}'.",
                    details={
                        "expected_sha256": expected_sha256.lower(),
                        "computed_sha256": computed_sha256,
                    },
                )

            mid = validate_model_id(model_id) if model_id else f"mod_{computed_sha256[:12]}"
            target_model_dir = (self.models_dir / mid).resolve()
            if not target_model_dir.is_relative_to(self.models_dir):
                raise PathTraversalError(f"Target directory for model '{mid}' escapes models root.")

            if target_model_dir.is_symlink():
                raise SymlinkNotPermittedError("Model directory cannot be a symlink.")

            target_model_dir.mkdir(parents=True, exist_ok=True)
            target_file_path = (target_model_dir / valid_filename).resolve()

            if not target_file_path.is_relative_to(target_model_dir):
                raise PathTraversalError(
                    f"Target file path '{target_file_path}' escapes model directory."
                )

            if target_file_path.is_symlink():
                raise SymlinkNotPermittedError("Model file path cannot be a symlink.")

            # Atomic rename into permanent destination, safely handling concurrent ingestion
            try:
                temp_target.replace(target_file_path)
            except (PermissionError, OSError):
                if target_file_path.exists():
                    temp_target.unlink(missing_ok=True)
                else:
                    raise

        except Exception:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            raise

        manifest_canonical = {
            "files": [
                {
                    "relative_path": valid_filename,
                    "sha256": computed_sha256,
                    "size_bytes": total_bytes,
                }
            ],
            "manifest_version": "1.0.0",
            "total_bytes": total_bytes,
            "total_files": 1,
        }
        manifest_bytes = canonical_json_bytes(manifest_canonical)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        # Deterministic snapshot ID anchored to manifest digest prefix
        snap_id = f"snap_{manifest_sha256[:16]}"
        snapshot_dir = self.snapshots_dir / snap_id

        snapshot = AssetSnapshot(
            snapshot_id=snap_id,
            asset_id=mid,
            asset_type=AssetType.MODEL_CHECKPOINT,
            manifest_sha256=manifest_sha256,
            created_at=datetime.now(UTC),
            file_count=1,
            total_size_bytes=total_bytes,
            schema_version="1.0.0",
            metadata={
                "model_id": mid,
                "filename": valid_filename,
                "storage_path": str(target_file_path),
                "snapshot_path": str(snapshot_dir),
                "artifact_sha256": computed_sha256,
            },
        )

        snapshot_mgr = SnapshotManager(config=self.snapshot_config)
        with snapshot_mgr._acquire_snapshot_lock(snap_id):
            if snapshot_dir.exists():
                verif = snapshot_mgr.verify_snapshot(
                    snap_id, expected_manifest_sha256=manifest_sha256
                )
                if verif.is_valid:
                    metadata_file = snapshot_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            existing_snapshot = AssetSnapshot.model_validate_json(
                                metadata_file.read_text("utf-8")
                            )
                            return ModelIngestionResult(
                                model_id=mid,
                                filename=valid_filename,
                                sha256=computed_sha256,
                                size_bytes=total_bytes,
                                storage_path=target_file_path,
                                snapshot=existing_snapshot,
                                snapshot_dir=snapshot_dir,
                            )
                        except Exception:
                            pass

                # If existing snapshot is invalid/tampered, quarantine it
                snapshot_mgr.quarantine_snapshot(
                    snapshot_id=snap_id,
                    reason=f"Corrupted or tampered existing model snapshot: {'; '.join(verif.errors)}",
                    verification_errors=verif.errors,
                    expected_manifest_sha256=manifest_sha256,
                    observed_manifest_sha256=verif.manifest_sha256,
                )

            # Atomic staging directory
            staging_dir = self.snapshots_dir / f".staging_{snap_id}_{uuid.uuid4().hex[:8]}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                manifest_path = staging_dir / "manifest.json"
                with open(manifest_path, "wb") as mf:
                    mf.write(manifest_bytes)

                snapshot_meta_path = staging_dir / "metadata.json"
                with open(snapshot_meta_path, "w", encoding="utf-8") as sf:
                    sf.write(snapshot.model_dump_json(indent=2))

                dst_model_file = staging_dir / valid_filename
                shutil.copy2(target_file_path, dst_model_file)

                # Set read-only protection
                snapshot_mgr._set_read_only(staging_dir)

                # Atomic promotion
                staging_dir.rename(snapshot_dir)

            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)

        return ModelIngestionResult(
            model_id=mid,
            filename=valid_filename,
            sha256=computed_sha256,
            size_bytes=total_bytes,
            storage_path=target_file_path,
            snapshot=snapshot,
            snapshot_dir=snapshot_dir,
        )
