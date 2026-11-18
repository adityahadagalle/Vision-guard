"""Checkpoint structure detector for PyTorch models.

Safely evaluates candidate checkpoints within an isolated worker subprocess under
strict weights_only=True enforcement. Detects unpickler errors, pickle exploit payloads,
and invalid top-level structures without executing untrusted bytecode in the supervisor.
"""

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
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.inspection.worker import (
    CheckpointInspectionReport,
    SubprocessModelWorker,
)


class CheckpointStructureDetector(BaseDetector):
    """Inspects candidate checkpoint structure and deserialization safety."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.checkpoint_structure"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Runs isolated checkpoint inspection and verifies structural integrity."""
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()

        model_file: Path | None = None
        target_path_str = context.snapshot.metadata.get("storage_path")
        if target_path_str and Path(target_path_str).exists():
            model_file = Path(target_path_str).resolve()
        else:
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

        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")
        config = ModelIntegrityConfig()
        worker = SubprocessModelWorker(config)

        # Check for cached report or inspect
        report: CheckpointInspectionReport
        cached = context.config.get("_cached_inspection_report")
        if isinstance(cached, CheckpointInspectionReport):
            report = cached
        else:
            try:
                report = worker.inspect(model_file)
                if isinstance(context.config, dict):
                    context.config["_cached_inspection_report"] = report
            except Exception as exc:
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=f"Model inspection worker failed: {exc}",
                )

        if report.capability_status == CapabilityStatus.UNAVAILABLE:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.UNAVAILABLE,
                status_reason=report.error_message
                or "Checkpoint inspection capability is UNAVAILABLE due to lack of OS sandbox isolation.",
                metadata={"isolation_level": report.isolation_level},
            )

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        if not report.success:
            uid = uuid.uuid4().hex[:12]
            fnd_id = f"fnd_struct_{uid}"
            ev_id = f"ev_struct_{uid}"

            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METADATA_MISMATCH,
                finding_id=fnd_id,
                signal_name="safe_checkpoint_deserialization",
                rationale=f"Failed weights_only inspection: {report.error_message}",
                created_at=now,
                sample_reference=model_file.name,
                observed_value=1.0,
                threshold_value=0.0,
                metadata={
                    "error_message": report.error_message,
                    "error_type": report.error_type,
                    "weights_only_verified": False,
                },
            )

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.STRUCTURAL_DEFECT,
                severity=SeverityLevel.CRITICAL,
                title=f"Unsafe or Malformed Checkpoint: {model_file.name}",
                description=(
                    f"The candidate checkpoint could not be deserialized under safe weights_only=True rules: "
                    f"{report.error_message}. Possible unpickler exploit payload or corrupted file structure."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.QUARANTINE,
                anomaly_score=1.0,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                    )
                ],
                evidence_ids=[ev_id],
                details={"error": report.error_message, "error_type": report.error_type},
            )

            findings.append(fnd)
            evidence_list.append(ev)

        elif report.defects:
            for defect in report.defects:
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_defect_{uid}"
                ev_id = f"ev_defect_{uid}"

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.METADATA_MISMATCH,
                    finding_id=fnd_id,
                    signal_name="state_dict_defect",
                    rationale=defect,
                    created_at=now,
                    sample_reference=model_file.name,
                    observed_value=1.0,
                    threshold_value=0.0,
                )

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=assessment_id,
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.STRUCTURAL_DEFECT,
                    severity=SeverityLevel.HIGH,
                    title=f"State Dict Structural Defect: {defect[:60]}",
                    description=defect,
                    confidence=1.0,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                    ),
                    created_at=now,
                    suggested_disposition=SuggestedDisposition.REVIEW,
                    anomaly_score=1.0,
                    affected_assets=[
                        AffectedAssetReference(
                            asset_id=context.snapshot.asset_id,
                            sample_path=model_file.name,
                        )
                    ],
                    evidence_ids=[ev_id],
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
                "success": report.success,
                "format": report.format,
                "weights_only_verified": report.weights_only_verified,
                "state_dict_fingerprint": report.state_dict_fingerprint,
                "total_tensors": report.total_tensors,
                "total_parameters": report.total_parameters,
                "total_memory_bytes": report.total_memory_bytes,
                "isolation_level": report.isolation_level,
            },
        )
