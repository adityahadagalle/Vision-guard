"""Architecture compatibility detector for PyTorch models.

Compares candidate state-dict parameters against reference architecture contracts
from the TrustedArchitectureRegistry (ResNet-18, MobileNetV2, SimpleCNN).
Detects missing layers, extraneous layers, shape mismatches, and data type discrepancies.
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
from backend.app.modules.model_integrity.architecture.registry import (
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.inspection.worker import (
    CheckpointInspectionReport,
    SubprocessModelWorker,
)


class ArchitectureCompatibilityDetector(BaseDetector):
    """Evaluates whether checkpoint layers conform to a trusted model architecture."""

    def __init__(self, registry: TrustedArchitectureRegistry | None = None) -> None:
        self._registry = registry or TrustedArchitectureRegistry()

    @property
    def detector_id(self) -> str:
        return "model_integrity.architecture_compatibility"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Compares state-dict tensors against declared architecture specifications."""
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

        expected_arch = context.config.get(
            "expected_architecture"
        ) or context.snapshot.metadata.get("expected_architecture")

        if not expected_arch:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.SKIPPED,
                status_reason="No expected architecture specified in configuration.",
            )

        if not self._registry.has(expected_arch):
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    f"Unknown architecture '{expected_arch}'. "
                    f"Available registered architectures: {self._registry.list_ids()}."
                ),
            )

        # Retrieve inspection report
        report: CheckpointInspectionReport
        cached = context.config.get("_cached_inspection_report")
        if isinstance(cached, CheckpointInspectionReport):
            report = cached
        else:
            config = ModelIntegrityConfig()
            worker = SubprocessModelWorker(config)
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

        if not report.success:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.SKIPPED,
                status_reason=f"Checkpoint deserialization failed: {report.error_message}",
            )

        compat = self._registry.validate_compatibility(expected_arch, report.tensors)

        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        if not compat.compatible:
            uid = uuid.uuid4().hex[:12]
            fnd_id = f"fnd_compat_{uid}"
            ev_id = f"ev_compat_{uid}"

            total_mismatches = (
                len(compat.missing_layers)
                + len(compat.unexpected_layers)
                + len(compat.shape_mismatches)
                + len(compat.dtype_mismatches)
            )

            # Choose category: if unexpected layers are added, unauthorized change is suspected
            category = (
                FindingCategory.UNAUTHORIZED_MODEL_CHANGE
                if compat.unexpected_layers
                else FindingCategory.STRUCTURAL_DEFECT
            )

            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METADATA_MISMATCH,
                finding_id=fnd_id,
                signal_name="architecture_compatibility_check",
                rationale=(
                    f"Candidate checkpoint does not match reference contract for '{expected_arch}'. "
                    f"Missing: {len(compat.missing_layers)}, Unexpected: {len(compat.unexpected_layers)}, "
                    f"Shape mismatches: {len(compat.shape_mismatches)}, Dtype mismatches: {len(compat.dtype_mismatches)}."
                ),
                created_at=now,
                sample_reference=model_file.name,
                observed_value=float(total_mismatches),
                threshold_value=0.0,
                metadata={
                    "architecture_id": expected_arch,
                    "missing_layers": compat.missing_layers[:10],
                    "unexpected_layers": compat.unexpected_layers[:10],
                    "shape_mismatches": compat.shape_mismatches[:10],
                    "dtype_mismatches": compat.dtype_mismatches[:10],
                    "compatibility_score": compat.compatibility_score,
                },
            )

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=category,
                severity=SeverityLevel.HIGH,
                title=f"Architecture Incompatibility with '{expected_arch}'",
                description=(
                    f"Checkpoint '{model_file.name}' deviates from the trusted '{expected_arch}' specification: "
                    f"{len(compat.missing_layers)} missing layer(s), {len(compat.unexpected_layers)} unexpected layer(s), "
                    f"{len(compat.shape_mismatches)} shape mismatch(es). Overall compatibility score: {compat.compatibility_score}."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={"expected_architecture": expected_arch},
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                anomaly_score=round(1.0 - compat.compatibility_score, 4),
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                    )
                ],
                evidence_ids=[ev_id],
                details={
                    "architecture_id": expected_arch,
                    "compatibility_score": compat.compatibility_score,
                    "missing_count": len(compat.missing_layers),
                    "unexpected_count": len(compat.unexpected_layers),
                    "shape_mismatch_count": len(compat.shape_mismatches),
                },
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
                "architecture_id": expected_arch,
                "compatible": compat.compatible,
                "compatibility_score": compat.compatibility_score,
                "expected_layers": compat.total_expected_layers,
                "candidate_layers": compat.total_candidate_layers,
            },
        )
