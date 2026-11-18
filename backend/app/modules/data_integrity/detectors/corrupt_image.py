"""Corrupt image detector for dataset snapshots.

Inspects every image asset recorded in the snapshot manifest, verifying decoding
integrity, header magic bytes, pixel count bounds, and dimensions using the safe
image validator. Emits structured UniversalFinding records for every damaged asset.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities.enums import (
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.image_validator import SafeImageValidator

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class CorruptImageDetector(BaseDetector):
    """Detects damaged, truncated, or unreadable image files within a snapshot."""

    @property
    def detector_id(self) -> str:
        return "data_integrity.corrupt_image"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_labels=False, requires_embeddings=False)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Inspects snapshot images and returns findings for any decoding defects."""
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()

        manifest_path = snapshot_dir / "manifest.json"
        manifest_files: list[dict] = []

        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_files = manifest_data.get("files", [])
            except Exception as err:
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=f"Failed to read snapshot manifest: {err}",
                )
        else:
            # Fall back to scanning files in snapshot directory
            for p in snapshot_dir.rglob("*"):
                if p.is_file() and p.name not in ("manifest.json", "metadata.json"):
                    rel = str(p.relative_to(snapshot_dir)).replace("\\", "/")
                    manifest_files.append({"relative_path": rel, "sha256": None})

        # Filter for image files
        image_entries = [
            f for f in manifest_files if Path(f["relative_path"]).suffix.lower() in _IMAGE_EXTS
        ]

        if not image_entries:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.NOT_APPLICABLE,
                status_reason="No image files found in snapshot manifest to evaluate.",
                metadata={"total_files": len(manifest_files)},
            )

        config = IngestionConfig()
        validator = SafeImageValidator(config)

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        scanned_count = 0
        now = datetime.now(UTC)

        for entry in image_entries:
            rel_path = entry["relative_path"]
            file_path = snapshot_dir / Path(rel_path)
            scanned_count += 1
            file_sha256 = entry.get("sha256")

            try:
                validator.validate_image_file(file_path, relative_path=rel_path)
            except Exception as err:
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_corrupt_{uid}"
                ev_id = f"ev_corrupt_{uid}"

                file_size = file_path.stat().st_size if file_path.exists() else 0

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.IMAGE_CORRUPTION,
                    finding_id=fnd_id,
                    signal_name="image_structural_decoding",
                    rationale=f"Structural validation failed: {err}",
                    created_at=now,
                    sample_reference=rel_path,
                    evidence_digest=file_sha256,
                    observed_value=1.0,
                    threshold_value=0.0,
                    metadata={
                        "relative_path": rel_path,
                        "error_type": type(err).__name__,
                        "error_message": str(err),
                        "file_size": file_size,
                    },
                )

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=context.config.get(
                        "assessment_id", f"asm_{context.snapshot.snapshot_id}"
                    ),
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.CORRUPTED_FILE,
                    severity=SeverityLevel.HIGH,
                    title=f"Corrupt or Unreadable Image: {rel_path}",
                    description=(
                        f"Image file '{rel_path}' failed structural decoding verification: {err}"
                    ),
                    confidence=1.0,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={"max_image_pixels": config.max_image_pixels},
                    ),
                    created_at=now,
                    suggested_disposition=SuggestedDisposition.QUARANTINE,
                    anomaly_score=1.0,
                    affected_assets=[
                        AffectedAssetReference(
                            asset_id=context.snapshot.asset_id,
                            sample_path=rel_path,
                            sha256=file_sha256,
                        )
                    ],
                    evidence_ids=[ev_id],
                    details={"relative_path": rel_path, "error": str(err)},
                )

                findings.append(fnd)
                evidence_list.append(ev)

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "scanned_images": scanned_count,
                "corrupted_count": len(findings),
            },
        )
