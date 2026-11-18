"""Prediction consistency detector for PyTorch computer vision models.

Evaluates inference determinism across repeated executions under fixed eval() conditions.
Flags unexpected prediction flips or numerical output divergence.
"""

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


class PredictionConsistencyDetector(BaseDetector):
    """Evaluates prediction stability and determinism across repeated identical evaluations."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.prediction_consistency"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Evaluates prediction determinism across repeated evaluations."""
        dispatch = get_or_run_behavioral_report(self.detector_id, self.version, context)
        if isinstance(dispatch, DetectorResult):
            return dispatch

        model_file, architecture_id, report = dispatch
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        consistency = report.consistency
        flip_count = int(consistency.get("flip_count", 0))
        max_logit_drift = float(consistency.get("max_logit_drift", 0.0))
        repetitions = int(consistency.get("repetitions", 3))
        is_deterministic = bool(consistency.get("is_deterministic", True))

        uid = uuid.uuid4().hex[:12]
        ev_id = f"ev_cons_{uid}"

        if not is_deterministic or flip_count > 0 or max_logit_drift > 1e-4:
            fnd_id = f"fnd_inconsistent_{uid}"
            severity = SeverityLevel.HIGH if flip_count > 0 else SeverityLevel.MEDIUM
            anomaly_score = min(1.0, flip_count / max(1, report.probes_count) + max_logit_drift)

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.STRUCTURAL_DEFECT,
                severity=severity,
                title=f"Inconsistent Inference Output Detected: {model_file.name}",
                description=(
                    f"Model exhibited nondeterministic inference behavior across {repetitions} identical passes "
                    f"under fixed model.eval() conditions. Observed {flip_count} prediction class changes "
                    f"and maximum logit divergence of {max_logit_drift:.6f}. Indicates unpinned stochastic layers, "
                    f"unsupported stateful behavior, or non-deterministic GPU/CPU kernels."
                ),
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={
                        "repetitions": repetitions,
                        "tolerance": 1e-4,
                        "architecture_id": architecture_id,
                    },
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                anomaly_score=round(anomaly_score, 4),
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                    )
                ],
                evidence_ids=[ev_id],
                details={
                    "flip_count": flip_count,
                    "max_logit_drift": max_logit_drift,
                    "repetitions": repetitions,
                },
            )
            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.STATISTICAL_TEST,
                finding_id=fnd_id,
                signal_name="inference_prediction_consistency",
                rationale=(
                    f"Evaluated {repetitions} repeated inference executions on {report.probes_count} probes. "
                    f"Observed {flip_count} prediction flips and max logit drift of {max_logit_drift:.6f}."
                ),
                created_at=now,
                sample_reference=model_file.name,
                observed_value=float(flip_count),
                threshold_value=0.0,
                metadata={
                    "is_deterministic": is_deterministic,
                    "flip_count": flip_count,
                    "max_logit_drift": max_logit_drift,
                    "repetitions": repetitions,
                    "architecture_id": architecture_id,
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
                "is_deterministic": is_deterministic,
                "flip_count": flip_count,
                "max_logit_drift": max_logit_drift,
            },
        )
