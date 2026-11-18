"""State-dict integrity detector for PyTorch models.

Validates mathematical and tensor-level integrity of model weights, including
detection of non-finite floating point anomalies (NaN, Inf, -Inf) and verification
of canonical state-dict fingerprints.
"""

import re
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


class StateDictIntegrityDetector(BaseDetector):
    """Detects numerical anomalies (NaN/Inf) and verifies state-dict parameters."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.state_dict_integrity"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Inspects state-dict tensors for numerical integrity defects."""
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

        # Get inspection report from cache or run worker
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
                    status_reason=f"Model worker failed during inspection: {exc}",
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

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        # Check for NaN / Inf anomalies
        if report.has_nan or report.has_inf:
            uid = uuid.uuid4().hex[:12]
            fnd_id = f"fnd_nonfinite_{uid}"
            ev_id = f"ev_nonfinite_{uid}"

            total_non_finite = report.total_nan + report.total_inf
            sample_affected = report.affected_non_finite_tensors[:5]

            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                finding_id=fnd_id,
                signal_name="tensor_numerical_finiteness",
                rationale=(
                    f"Detected {report.total_nan} NaN and {report.total_inf} Inf values "
                    f"across {len(report.affected_non_finite_tensors)} tensor(s)."
                ),
                created_at=now,
                sample_reference=model_file.name,
                observed_value=float(total_non_finite),
                threshold_value=0.0,
                metadata={
                    "total_nan": report.total_nan,
                    "total_inf": report.total_inf,
                    "affected_tensors": report.affected_non_finite_tensors,
                },
            )

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.STRUCTURAL_DEFECT,
                severity=SeverityLevel.HIGH,
                title=f"Non-Finite Weights Detected in Checkpoint ({total_non_finite} values)",
                description=(
                    f"Checkpoint '{model_file.name}' contains {report.total_nan} NaN and {report.total_inf} Inf values "
                    f"in tensors: {', '.join(sample_affected)}. Non-finite weights prevent valid inference."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={"check": "isnan_or_isinf"},
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
                details={
                    "total_nan": report.total_nan,
                    "total_inf": report.total_inf,
                    "affected_tensors": report.affected_non_finite_tensors,
                },
            )

            findings.append(fnd)
            evidence_list.append(ev)

        # Expected state-dict fingerprint comparison (SEC-3A-05)
        expected_fp = None
        if "expected_state_dict_fingerprint" in context.config:
            expected_fp = context.config["expected_state_dict_fingerprint"]
        elif "expected_state_dict_fingerprint" in context.snapshot.metadata:
            expected_fp = context.snapshot.metadata["expected_state_dict_fingerprint"]

        fingerprint_verified: bool | None = None
        fingerprint_comparison: str = "SKIPPED_NO_REFERENCE"

        if expected_fp is not None:
            if not isinstance(expected_fp, str) or not re.match(
                r"^[a-fA-F0-9]{64}$", expected_fp.strip()
            ):
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=(
                        f"Malformed expected_state_dict_fingerprint '{expected_fp}'. "
                        "Must be a 64-character hexadecimal SHA-256 digest string."
                    ),
                )

            expected_norm = expected_fp.strip().lower()
            observed_norm = report.state_dict_fingerprint.lower()

            if observed_norm == expected_norm:
                fingerprint_verified = True
                fingerprint_comparison = "VERIFIED_MATCH"
            else:
                fingerprint_verified = False
                fingerprint_comparison = "VERIFIED_MISMATCH"
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_fp_mismatch_{uid}"
                ev_id = f"ev_fp_mismatch_{uid}"

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.METADATA_MISMATCH,
                    finding_id=fnd_id,
                    signal_name="state_dict_fingerprint_mismatch",
                    rationale=(
                        f"Observed state-dict fingerprint '{observed_norm}' does not match "
                        f"authoritative reference fingerprint '{expected_norm}'."
                    ),
                    created_at=now,
                    sample_reference=model_file.name,
                    evidence_digest=observed_norm,
                    observed_value=1.0,
                    threshold_value=0.0,
                    metadata={
                        "observed_fingerprint": observed_norm,
                        "expected_fingerprint": expected_norm,
                    },
                )

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=assessment_id,
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
                    severity=SeverityLevel.CRITICAL,
                    title=f"State-Dict Fingerprint Mismatch: {model_file.name}",
                    description=(
                        f"The candidate PyTorch checkpoint '{model_file.name}' has state-dict fingerprint "
                        f"'{observed_norm}', which does not match the approved reference '{expected_norm}'. "
                        "Model parameters have been substituted or modified."
                    ),
                    confidence=1.0,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={"algorithm": "sha256", "check": "state_dict_fingerprint"},
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
                    details={
                        "observed_fingerprint": observed_norm,
                        "expected_fingerprint": expected_norm,
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
                "state_dict_fingerprint": report.state_dict_fingerprint,
                "total_tensors": report.total_tensors,
                "total_parameters": report.total_parameters,
                "finite_weights_verified": not (report.has_nan or report.has_inf),
                "fingerprint_verified": fingerprint_verified,
                "fingerprint_comparison": fingerprint_comparison,
            },
        )
