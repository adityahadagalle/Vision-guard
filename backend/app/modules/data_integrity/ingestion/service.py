"""Orchestration service for the complete dataset ingestion and snapshot pipeline.

Coordinates upload quarantine, defensive pre-inspection, isolated worker extraction,
defensive image validation, format detection, normalization, and immutable snapshot sealing.
"""

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    IngestionError,
)
from backend.app.modules.data_integrity.ingestion.formats.detector import DatasetFormatDetector
from backend.app.modules.data_integrity.ingestion.image_validator import SafeImageValidator
from backend.app.modules.data_integrity.ingestion.inspector import DefensiveArchiveInspector
from backend.app.modules.data_integrity.ingestion.normalizer import (
    DatasetNormalizer,
    NormalizedDataset,
)
from backend.app.modules.data_integrity.ingestion.quarantine import (
    QuarantineManager,
    QuarantineRecord,
)
from backend.app.modules.data_integrity.ingestion.snapshot import SnapshotManager
from backend.app.modules.data_integrity.ingestion.worker import SubprocessExtractionWorker

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class IngestionResult(BaseModel):
    """Structured report returned upon successful ingestion and snapshot sealing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: AssetSnapshot
    status: IngestionStatus
    quarantine_record: QuarantineRecord
    normalized_dataset: NormalizedDataset
    snapshot_path: Path
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IngestionService:
    """Coordinates the end-to-end dataset ingestion lifecycle."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        self.quarantine_mgr = QuarantineManager(self.config)
        self.inspector = DefensiveArchiveInspector()
        self.worker = SubprocessExtractionWorker(self.config)
        self.image_validator = SafeImageValidator(self.config)
        self.format_detector = DatasetFormatDetector()
        self.normalizer = DatasetNormalizer()
        self.snapshot_mgr = SnapshotManager(self.config)

    def ingest_archive(
        self,
        stream_or_path: BinaryIO | Path,
        filename: str | None = None,
        asset_id: str | None = None,
        require_hardened_isolation: bool = False,
    ) -> IngestionResult:
        """Ingests, inspects, extracts, normalizes, and seals an untrusted archive dataset.

        Args:
            stream_or_path: Either a readable binary stream or local Path to the archive.
            filename: Original user filename (required if stream_or_path is a stream).
            asset_id: Optional internal asset identifier.
            require_hardened_isolation: If True, requires Linux container isolation; fails on Windows.

        Returns:
            IngestionResult containing the created AssetSnapshot and normalized metadata.

        Raises:
            IngestionError: If any quarantine, inspection, extraction, or validation check fails.
        """
        # Step 1: Stage file into Quarantine
        if isinstance(stream_or_path, Path):
            fname = filename or stream_or_path.name
            quarantine_rec = self.quarantine_mgr.quarantine_file(stream_or_path, filename=fname)
        else:
            if not filename:
                raise ValueError("Filename is required when ingesting a binary stream.")
            quarantine_rec = self.quarantine_mgr.quarantine_stream(
                stream_or_path, filename=filename
            )

        iid = quarantine_rec.ingestion_id
        quarantine_file = quarantine_rec.quarantine_path
        staging_dir = self.config.staging_dir / f"stage_{iid}_{uuid.uuid4().hex[:8]}"

        try:
            # Step 2: Defensive Container Pre-Inspection
            self.quarantine_mgr.update_status(iid, IngestionStatus.INSPECTING)
            inspection = self.inspector.inspect_zip(quarantine_file, self.config)

            # Step 3: Isolated Subprocess Worker Extraction
            self.quarantine_mgr.update_status(iid, IngestionStatus.EXTRACTING)
            worker_result = self.worker.extract(
                quarantine_file,
                staging_dir,
                expected_sha256=quarantine_rec.sha256,
                require_hardened_isolation=require_hardened_isolation,
            )

            # Step 4: Defensive Image Validation
            self.quarantine_mgr.update_status(iid, IngestionStatus.VALIDATING)
            image_warnings: list[str] = list(inspection.warnings)
            image_errors: list[str] = []

            for ext in _IMAGE_EXTS:
                for img_file in staging_dir.rglob(f"*{ext}"):
                    rel_name = str(img_file.relative_to(staging_dir)).replace("\\", "/")
                    try:
                        self.image_validator.validate_image_file(img_file, relative_path=rel_name)
                    except Exception as img_err:
                        image_errors.append(f"Image validation failed for '{rel_name}': {img_err}")

            # Step 5: Format Detection & Normalization
            self.quarantine_mgr.update_status(iid, IngestionStatus.NORMALIZING)
            adapter = self.format_detector.detect(staging_dir)
            parsed_dataset = adapter.parse_and_validate(staging_dir)

            all_warnings = image_warnings + parsed_dataset.validation_warnings
            all_errors = image_errors + parsed_dataset.validation_errors

            normalized_ds = self.normalizer.normalize(
                parsed_dataset,
                extra_metadata={
                    "ingestion_id": iid,
                    "original_archive_sha256": quarantine_rec.sha256,
                    "original_filename": quarantine_rec.original_filename,
                    "worker_isolation_level": worker_result.isolation_level,
                    "total_uncompressed_bytes": worker_result.total_bytes_extracted,
                },
            )

            # Step 6: Immutable Snapshot Creation
            snapshot_entity, snapshot_path = self.snapshot_mgr.create_snapshot(
                staging_dir,
                normalized_ds,
                asset_id=asset_id,
            )

            # Update final status to ACCEPTED
            updated_quarantine = self.quarantine_mgr.update_status(
                iid,
                IngestionStatus.ACCEPTED,
                metadata_update={"snapshot_id": snapshot_entity.snapshot_id},
            )

            return IngestionResult(
                snapshot=snapshot_entity,
                status=IngestionStatus.ACCEPTED,
                quarantine_record=updated_quarantine,
                normalized_dataset=normalized_ds,
                snapshot_path=snapshot_path,
                warnings=all_warnings,
                errors=all_errors,
            )

        except IngestionError:
            self.quarantine_mgr.update_status(iid, IngestionStatus.REJECTED)
            raise
        except Exception as err:
            self.quarantine_mgr.update_status(iid, IngestionStatus.FAILED)
            raise IngestionError(
                f"Ingestion pipeline failed unexpectedly for '{quarantine_rec.original_filename}': {err}",
                details={"ingestion_id": iid, "error": str(err)},
            ) from err
        finally:
            # Clean up ephemeral staging directory
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
