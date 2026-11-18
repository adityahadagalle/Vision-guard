"""Trusted reference behavioral comparison detector for PyTorch computer vision models.

Compares candidate model outputs against an explicitly trusted reference model across
identical deterministic reference probes. Evaluates top-1 agreement and logit divergence.
"""

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.session import get_session_factory
from backend.app.domain.entities.enums import (
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    IngestionStatus,
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
from backend.app.modules.model_integrity.behavioral.common import (
    get_or_run_behavioral_report,
)


class ReferenceComparisonDetector(BaseDetector):
    """Compares candidate model predictions and logits against an authoritative trusted reference model."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.reference_comparison"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(
            requires_model_checkpoint=True,
            requires_reference_set=True,
        )

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Evaluates prediction agreement and logit divergence against trusted reference."""
        # 1. Check prerequisite: trusted reference model must be explicitly provided
        ref_path_str = context.config.get("reference_model_path") or context.snapshot.metadata.get(
            "reference_model_path"
        )
        ref_model_id = context.config.get("reference_model_id") or context.snapshot.metadata.get(
            "reference_model_id"
        )
        if not ref_path_str and not ref_model_id:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.SKIPPED,
                status_reason=(
                    "No trusted reference model specified in assessment configuration or metadata "
                    "for behavioral comparison."
                ),
            )

        # 2. Resolve candidate model checkpoint
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
                status_reason="No candidate PyTorch model checkpoint (.pt/.pth) found in snapshot.",
            )

        candidate_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()

        # 3. Locate persistence repository to verify authoritative records
        model_repo = context.config.get("_model_repo")
        if model_repo is None:
            db_session = context.config.get("_db_session")
            if db_session is not None:
                model_repo = SqliteModelRepository(db_session)
            else:
                try:
                    session_factory = get_session_factory()
                    session = session_factory()
                    model_repo = SqliteModelRepository(session)
                except Exception:
                    model_repo = None

        if model_repo is None:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason="Persistence layer unavailable to verify authoritative reference model record.",
            )

        # 4. Resolve reference file path
        authoritative_record = None
        if ref_model_id:
            authoritative_record = model_repo.get_model(str(ref_model_id))
            if authoritative_record is None:
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=f"Authoritative ModelRepository record not found for reference model ID '{ref_model_id}'.",
                    metadata={"reference_model_id": str(ref_model_id)},
                )
            ref_path = Path(authoritative_record.storage_path).resolve()
        else:
            ref_path = Path(ref_path_str).resolve()

        if not ref_path.exists():
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=f"Specified reference model checkpoint '{ref_path}' does not exist on disk.",
                metadata={"reference_path": str(ref_path)},
            )

        # 5. Strictly reject candidate self-reference (Path & Digest)
        if ref_path.resolve() == model_file.resolve():
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    f"Candidate checkpoint cannot serve as its own reference model: "
                    f"canonical reference path matches candidate path '{model_file}'."
                ),
                metadata={
                    "candidate_path": str(model_file),
                    "reference_path": str(ref_path),
                    "error_type": "SELF_REFERENCE_PATH_DETECTED",
                },
            )

        ref_file_bytes = ref_path.read_bytes()
        ref_file_sha256 = hashlib.sha256(ref_file_bytes).hexdigest()

        if ref_file_sha256 == candidate_sha256:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    "Candidate checkpoint cannot serve as its own reference model: "
                    "reference content SHA-256 digest is identical to candidate checkpoint."
                ),
                metadata={
                    "candidate_sha256": candidate_sha256,
                    "reference_sha256": ref_file_sha256,
                    "error_type": "SELF_REFERENCE_DIGEST_DETECTED",
                },
            )

        # 6. Authoritative ModelRepository verification
        if authoritative_record is None:
            authoritative_record = model_repo.get_model_by_storage_path(str(ref_path))
            if authoritative_record is None and ref_path_str:
                authoritative_record = model_repo.get_model_by_storage_path(str(ref_path_str))
            if authoritative_record is None:
                authoritative_record = model_repo.get_model_by_sha256(ref_file_sha256)

        if authoritative_record is None:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    f"Reference model '{ref_path.name}' has no authoritative record in ModelRepository. "
                    f"Unregistered reference models cannot be used."
                ),
                metadata={
                    "reference_path": str(ref_path),
                    "error_type": "UNREGISTERED_REFERENCE_MODEL",
                },
            )

        if authoritative_record.status != IngestionStatus.ACCEPTED:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    f"Authoritative reference model '{authoritative_record.model_id}' is not in ACCEPTED state "
                    f"(current status: '{authoritative_record.status}')."
                ),
                metadata={
                    "reference_model_id": authoritative_record.model_id,
                    "reference_status": str(authoritative_record.status),
                    "error_type": "REFERENCE_NOT_ACCEPTED",
                },
            )

        expected_ref_sha256 = authoritative_record.sha256
        if ref_file_sha256 != expected_ref_sha256:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=(
                    f"Reference file '{ref_path.name}' digest '{ref_file_sha256}' does not match "
                    f"authoritative ModelRepository digest '{expected_ref_sha256}'. "
                    f"Reference artifact may be modified or tampered."
                ),
                metadata={
                    "reference_model_id": authoritative_record.model_id,
                    "observed_sha256": ref_file_sha256,
                    "authoritative_sha256": expected_ref_sha256,
                    "error_type": "REFERENCE_DIGEST_MISMATCH",
                },
            )

        # 7. Execute isolated behavioral inference via common dispatch
        dispatch = get_or_run_behavioral_report(
            detector_id=self.detector_id,
            detector_version=self.version,
            context=context,
            reference_checkpoint_path=ref_path,
            reference_sha256=ref_file_sha256,
        )
        if isinstance(dispatch, DetectorResult):
            return dispatch

        model_file, architecture_id, report = dispatch
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        if not report.reference_comparison:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.SKIPPED,
                status_reason="Reference model comparison metrics were not produced by the behavioral worker.",
            )

        ref_comp = report.reference_comparison
        agreement_rate = float(ref_comp.get("top1_agreement_rate", 1.0))
        max_logit_div = float(ref_comp.get("max_logit_divergence", 0.0))
        mean_logit_div = float(ref_comp.get("mean_logit_divergence", 0.0))
        mean_conf_diff = float(ref_comp.get("mean_confidence_difference", 0.0))
        fp_match = bool(ref_comp.get("fingerprints_match", True))
        ref_fp = str(ref_comp.get("reference_behavioral_fingerprint", ""))

        min_agreement = float(context.config.get("reference_agreement_threshold", 0.95))
        max_div_tol = float(context.config.get("reference_divergence_threshold", 1e-4))

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        uid = uuid.uuid4().hex[:12]
        ev_id = f"ev_refcomp_{uid}"

        if agreement_rate < 1.0 or max_logit_div > max_div_tol:
            fnd_id = f"fnd_ref_diverge_{uid}"
            if agreement_rate < 0.90:
                severity = SeverityLevel.HIGH
                disposition = SuggestedDisposition.QUARANTINE
            elif agreement_rate < 1.0:
                severity = SeverityLevel.MEDIUM
                disposition = SuggestedDisposition.REVIEW
            else:
                severity = SeverityLevel.LOW
                disposition = SuggestedDisposition.ACCEPT

            anomaly_score = max(0.0, round(1.0 - agreement_rate, 4))
            if anomaly_score == 0.0 and max_logit_div > max_div_tol:
                anomaly_score = min(1.0, round(max_logit_div, 4))

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
                severity=severity,
                title=f"Behavioral Divergence from Trusted Reference Model: {model_file.name}",
                description=(
                    f"Candidate model behavior diverged from trusted reference model '{ref_path.name}' "
                    f"(ID: '{authoritative_record.model_id}'): "
                    f"top-1 prediction agreement is {agreement_rate:.1%} (threshold: {min_agreement:.1%}), "
                    f"maximum logit divergence is {max_logit_div:.6f} (tolerance: {max_div_tol:.6f}), "
                    f"mean logit divergence is {mean_logit_div:.6f}. "
                    f"Behavioral fingerprints match: {fp_match}."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={
                        "reference_model": str(ref_path),
                        "reference_model_id": authoritative_record.model_id,
                        "agreement_threshold": min_agreement,
                        "divergence_threshold": max_div_tol,
                        "architecture_id": architecture_id,
                    },
                ),
                created_at=now,
                suggested_disposition=disposition,
                anomaly_score=anomaly_score,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                    )
                ],
                evidence_ids=[ev_id],
                details={
                    "reference_model_id": authoritative_record.model_id,
                    "reference_authoritative_sha256": authoritative_record.sha256,
                    "reference_verified_sha256": ref_file_sha256,
                    "candidate_sha256": candidate_sha256,
                    "top1_agreement_rate": agreement_rate,
                    "max_logit_divergence": max_logit_div,
                    "mean_logit_divergence": mean_logit_div,
                    "mean_confidence_difference": mean_conf_diff,
                    "reference_fingerprint": ref_fp,
                    "candidate_fingerprint": report.behavioral_fingerprint,
                },
            )
            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                finding_id=fnd_id,
                signal_name="reference_model_behavioral_agreement",
                rationale=(
                    f"Evaluated candidate against authoritative reference model '{ref_path.name}' "
                    f"(ID: '{authoritative_record.model_id}', digest: '{expected_ref_sha256[:16]}...') over {report.probes_count} probes. "
                    f"Observed top-1 agreement of {agreement_rate:.1%}, max logit divergence of {max_logit_div:.6f}."
                ),
                created_at=now,
                sample_reference=model_file.name,
                observed_value=agreement_rate,
                threshold_value=min_agreement,
                metadata={
                    "reference_model": str(ref_path),
                    "reference_model_id": authoritative_record.model_id,
                    "reference_authoritative_sha256": authoritative_record.sha256,
                    "reference_verified_sha256": ref_file_sha256,
                    "candidate_sha256": candidate_sha256,
                    "top1_agreement_rate": agreement_rate,
                    "max_logit_divergence": max_logit_div,
                    "mean_logit_divergence": mean_logit_div,
                    "mean_confidence_difference": mean_conf_diff,
                    "fingerprints_match": fp_match,
                    "reference_fingerprint": ref_fp,
                    "candidate_fingerprint": report.behavioral_fingerprint,
                },
            )
            evidence_list.append(ev)
            findings.append(fnd)

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "reference_model": str(ref_path),
                "reference_model_id": authoritative_record.model_id,
                "reference_verified_sha256": ref_file_sha256,
                "candidate_sha256": candidate_sha256,
                "top1_agreement_rate": agreement_rate,
                "max_logit_divergence": max_logit_div,
                "fingerprints_match": fp_match,
            },
        )
