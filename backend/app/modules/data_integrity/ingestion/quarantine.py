"""Quarantine subsystem for receiving, staging, and hashing untrusted incoming archives.

Ensures raw untrusted data is isolated in a restricted filesystem boundary,
hashed via streaming reads without unbounded memory allocation, and gated
by explicit ingestion status transitions.
"""

import hashlib
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    QuarantineError,
)

_SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,255}$")


def sanitize_filename(filename: str) -> str:
    """Sanitizes an incoming filename, stripping directory components and unsafe characters."""
    basename = Path(filename).name.strip()
    if not basename:
        return f"unnamed_upload_{uuid.uuid4().hex[:8]}.bin"

    # Replace spaces and prohibited characters with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", basename)
    # Strip leading dots to prevent hidden files
    while cleaned.startswith("."):
        cleaned = cleaned[1:]

    if not cleaned or not _SAFE_FILENAME_PATTERN.match(cleaned):
        return f"safe_upload_{uuid.uuid4().hex[:8]}.bin"
    return cleaned


class QuarantineRecord(BaseModel):
    """Immutable audit record tracking an untrusted asset in quarantine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ingestion_id: str
    original_filename: str
    quarantine_path: Path
    sha256: str
    size_bytes: int
    status: IngestionStatus
    created_at: datetime
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QuarantineManager:
    """Manages the quarantine staging lifecycle for untrusted incoming uploads."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        self.config.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, QuarantineRecord] = {}

    def quarantine_stream(
        self,
        stream: BinaryIO,
        filename: str,
        ingestion_id: str | None = None,
    ) -> QuarantineRecord:
        """Streams untrusted data into quarantine, computing SHA-256 and enforcing size bounds.

        Args:
            stream: Readable binary stream of incoming upload.
            filename: Original user-provided filename.
            ingestion_id: Optional preset ingestion ID (generated if None).

        Returns:
            QuarantineRecord documenting the stored asset and computed digest.

        Raises:
            EmptyFileError: If the uploaded stream is 0 bytes.
            FileTooLargeError: If bytes written exceed config.max_upload_bytes.
            QuarantineError: On I/O or staging failures.
        """
        iid = ingestion_id or f"ingest_{uuid.uuid4().hex}"
        safe_name = sanitize_filename(filename)
        target_dir = self.config.quarantine_dir / iid
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with target_path.open("wb") as out_file:
                while True:
                    chunk = stream.read(self.config.chunk_size_bytes)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > self.config.max_upload_bytes:
                        out_file.close()
                        self._safe_remove_dir(target_dir)
                        raise FileTooLargeError(
                            f"Upload exceeded maximum permissible size of {self.config.max_upload_bytes} bytes "
                            f"(received {total_bytes} bytes).",
                            details={
                                "max_bytes": self.config.max_upload_bytes,
                                "received_bytes": total_bytes,
                            },
                        )

                    hasher.update(chunk)
                    out_file.write(chunk)

            if total_bytes == 0:
                self._safe_remove_dir(target_dir)
                raise EmptyFileError(
                    f"Uploaded file '{filename}' contains 0 bytes.",
                    details={"ingestion_id": iid, "filename": filename},
                )

            record = QuarantineRecord(
                ingestion_id=iid,
                original_filename=filename,
                quarantine_path=target_path.resolve(),
                sha256=hasher.hexdigest(),
                size_bytes=total_bytes,
                status=IngestionStatus.QUARANTINED,
                created_at=datetime.now(UTC),
                metadata={"sanitized_filename": safe_name},
            )
            self._records[iid] = record
            return record

        except (FileTooLargeError, EmptyFileError):
            raise
        except Exception as err:
            self._safe_remove_dir(target_dir)
            raise QuarantineError(
                f"Failed to quarantine incoming file '{filename}': {err}",
                details={"ingestion_id": iid, "error": str(err)},
            ) from err

    def quarantine_file(self, source_path: Path, filename: str | None = None) -> QuarantineRecord:
        """Quarantines an existing local file via streaming copy and SHA-256 calculation."""
        if not source_path.exists() or not source_path.is_file():
            raise QuarantineError(
                f"Source file does not exist or is not a regular file: {source_path}",
                details={"source_path": str(source_path)},
            )

        name = filename or source_path.name
        with source_path.open("rb") as f:
            return self.quarantine_stream(f, filename=name)

    def update_status(
        self,
        ingestion_id: str,
        new_status: IngestionStatus,
        metadata_update: dict[str, str | int | float | bool] | None = None,
    ) -> QuarantineRecord:
        """Updates the tracked lifecycle state of an ingested quarantine record."""
        record = self._records.get(ingestion_id)
        if not record:
            raise QuarantineError(
                f"No quarantine record found for ingestion ID '{ingestion_id}'.",
                details={"ingestion_id": ingestion_id},
            )

        updated_meta = dict(record.metadata)
        if metadata_update:
            updated_meta.update(metadata_update)

        new_record = QuarantineRecord(
            ingestion_id=record.ingestion_id,
            original_filename=record.original_filename,
            quarantine_path=record.quarantine_path,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            status=new_status,
            created_at=record.created_at,
            metadata=updated_meta,
        )
        self._records[ingestion_id] = new_record
        return new_record

    def get_record(self, ingestion_id: str) -> QuarantineRecord | None:
        """Retrieves a quarantine tracking record by ingestion ID."""
        return self._records.get(ingestion_id)

    def cleanup_quarantine(self, ingestion_id: str) -> bool:
        """Removes the quarantine directory and stored raw files for a given ingestion."""
        target_dir = self.config.quarantine_dir / ingestion_id
        if target_dir.exists():
            self._safe_remove_dir(target_dir)
            return True
        return False

    @staticmethod
    def _safe_remove_dir(target_dir: Path) -> None:
        """Safely removes a directory tree without following external symlinks."""
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
