"""Behavioral fingerprint detector for PyTorch computer vision models.

Generates a deterministic canonical behavioral fingerprint from controlled reference probes
and compares against authoritative expected reference behavior when available.
"""

import re
import uuid
from datetime import UTC, datetime

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
from backend.app.modules.model_integrity.behavioral.common import (
    get_or_run_behavioral_report,
)


class BehavioralFingerprintDetector(BaseDetector):
    """Generates and verifies canonical behavioral fingerprints across deterministic reference probes."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.behavioral_fingerprint"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Executes behavioral fingerprint extraction and evaluates reference consistency."""
        dispatch = get_or_run_behavioral_report(self.detector_id, self.version, context)
        if isinstance(dispatch, DetectorResult):
            return dispatch

        model_file, architecture_id, report = dispatch
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        uid = uuid.uuid4().hex[:12]
        ev_id = f"ev_bfp_{uid}"

        # 1. Authoritative expected behavioral fingerprint comparison
        expected_bfp = context.config.get(
            "expected_behavioral_fingerprint"
        ) or context.snapshot.metadata.get("expected_behavioral_fingerprint")

        comparison_status = "SKIPPED_NO_REFERENCE"
        fingerprint_matches = None

        if expected_bfp is not None:
            if not isinstance(expected_bfp, str) or not re.match(
                r"^[a-fA-F0-9]{64}$", expected_bfp.strip()
            ):
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=(
                        f"Malformed expected_behavioral_fingerprint '{expected_bfp}'. "
                        "Must be a 64-character hexadecimal SHA-256 digest string."
                    ),
                )

            exp_norm = expected_bfp.strip().lower()
            obs_norm = report.behavioral_fingerprint.lower()

            if obs_norm == exp_norm:
                comparison_status = "VERIFIED_MATCH"
                fingerprint_matches = True
            else:
                comparison_status = "VERIFIED_MISMATCH"
                fingerprint_matches = False
                fnd_id = f"fnd_bfp_mismatch_{uid}"

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=assessment_id,
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
                    severity=SeverityLevel.HIGH,
                    title=f"Model Behavioral Fingerprint Mismatch: {model_file.name}",
                    description=(
                        f"Observed behavioral fingerprint '{obs_norm}' diverges from reference '{exp_norm}'. "
                        f"Predictions or output confidence distributions across deterministic reference probes "
                        f"do not match authorized model baseline."
                    ),
                    confidence=1.0,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={
                            "architecture_id": architecture_id,
                            "probe_count": report.probes_count,
                            "probe_seed": report.probe_seed,
                        },
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
                        "observed_fingerprint": obs_norm,
                        "expected_fingerprint": exp_norm,
                        "architecture_id": architecture_id,
                    },
                )
                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                    finding_id=fnd_id,
                    signal_name="model_behavioral_fingerprint",
                    rationale=(
                        f"Generated deterministic canonical behavioral fingerprint across {report.probes_count} probes. "
                        f"Comparison status: {comparison_status}."
                    ),
                    created_at=now,
                    sample_reference=model_file.name,
                    evidence_digest=report.behavioral_fingerprint,
                    observed_value=1.0,
                    threshold_value=0.0,
                    metadata={
                        "behavioral_fingerprint": report.behavioral_fingerprint,
                        "expected_behavioral_fingerprint": expected_bfp,
                        "comparison_status": comparison_status,
                        "architecture_id": architecture_id,
                        "probes_count": report.probes_count,
                        "probe_seed": report.probe_seed,
                        "clean_concentration": report.clean_concentration,
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
                "behavioral_fingerprint": report.behavioral_fingerprint,
                "comparison_status": comparison_status,
                "fingerprint_matches": fingerprint_matches,
                "probes_count": report.probes_count,
            },
        )
