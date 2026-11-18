"""Artifact hash integrity detector for PyTorch model checkpoints.

Verifies the cryptographic SHA-256 digest of a model artifact against the
authoritative expected digest. Flags any unauthorized modifications or file
corruption with CRITICAL severity findings.
"""

import hashlib
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


class ModelArtifactHashDetector(BaseDetector):
    """Verifies that the candidate model checkpoint matches the expected SHA-256 digest."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.artifact_hash"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Computes candidate model file digest and compares against expected digest."""
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()

        # Locate candidate model file
        model_file: Path | None = None
        target_path_str = context.snapshot.metadata.get("storage_path")
        if target_path_str and Path(target_path_str).exists():
            model_file = Path(target_path_str).resolve()
        else:
            # Look inside snapshot directory
            for p in snapshot_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".pt", ".pth"):
                    model_file = p
                    break

        if not model_file or not model_file.exists():
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.NOT_APPLICABLE,
                status_reason="No PyTorch model checkpoint (.pt/.pth) found in snapshot.",
            )

        expected_sha256 = context.config.get("expected_sha256") or context.snapshot.metadata.get(
            "expected_sha256"
        )

        if not expected_sha256:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.SKIPPED,
                status_reason="No expected SHA-256 digest provided for artifact verification.",
                metadata={"model_path": str(model_file)},
            )

        # Compute file digest
        hasher = hashlib.sha256()
        with open(model_file, "rb") as f:
            while chunk := f.read(64 * 1024):
                hasher.update(chunk)
        observed_sha256 = hasher.hexdigest()

        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        if observed_sha256.lower() != expected_sha256.lower():
            uid = uuid.uuid4().hex[:12]
            fnd_id = f"fnd_hash_{uid}"
            ev_id = f"ev_hash_{uid}"

            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METADATA_MISMATCH,
                finding_id=fnd_id,
                signal_name="model_artifact_sha256",
                rationale=(
                    f"Observed artifact digest '{observed_sha256}' does not match "
                    f"authoritative reference digest '{expected_sha256}'."
                ),
                created_at=now,
                sample_reference=model_file.name,
                evidence_digest=observed_sha256,
                observed_value=1.0,
                threshold_value=0.0,
                metadata={
                    "expected_sha256": expected_sha256.lower(),
                    "observed_sha256": observed_sha256.lower(),
                    "file_size": model_file.stat().st_size,
                },
            )

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
                severity=SeverityLevel.CRITICAL,
                title=f"Model Artifact Digest Mismatch: {model_file.name}",
                description=(
                    f"The candidate PyTorch model checkpoint '{model_file.name}' has digest "
                    f"'{observed_sha256}', which does not match the approved digest '{expected_sha256}'. "
                    "Possible unauthorized weight substitution or file tampering."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={"algorithm": "sha256"},
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.QUARANTINE,
                anomaly_score=1.0,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                        sha256=observed_sha256,
                    )
                ],
                evidence_ids=[ev_id],
                details={
                    "expected_sha256": expected_sha256.lower(),
                    "observed_sha256": observed_sha256.lower(),
                },
            )

            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.COMPLETED,
                findings=[fnd],
                evidence=[ev],
                metadata={
                    "matched": False,
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                },
            )

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=[],
            evidence=[],
            metadata={
                "matched": True,
                "sha256": observed_sha256,
            },
        )
