"""Input sensitivity detector for PyTorch computer vision models.

Evaluates whether small, bounded input perturbations cause disproportionately large
prediction changes or decision boundary instability.
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


class InputSensitivityDetector(BaseDetector):
    """Evaluates prediction stability under bounded input perturbations."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.input_sensitivity"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Evaluates prediction sensitivity and logit amplification under bounded noise."""
        dispatch = get_or_run_behavioral_report(self.detector_id, self.version, context)
        if isinstance(dispatch, DetectorResult):
            return dispatch

        model_file, architecture_id, report = dispatch
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        sensitivity = report.sensitivity
        epsilon = float(sensitivity.get("epsilon", 0.02))
        flip_count = int(sensitivity.get("flip_count", 0))
        flip_rate = float(sensitivity.get("flip_rate", 0.0))
        mean_conf_delta = float(sensitivity.get("mean_confidence_delta", 0.0))
        mean_amplification = float(sensitivity.get("mean_logit_amplification", 0.0))

        flip_threshold = float(context.config.get("sensitivity_flip_threshold", 0.40))

        uid = uuid.uuid4().hex[:12]
        ev_id = f"ev_sens_{uid}"

        if flip_rate > flip_threshold:
            fnd_id = f"fnd_sensitivity_{uid}"

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=assessment_id,
                detector_id=self.detector_id,
                detector_version=self.version,
                category=FindingCategory.STRUCTURAL_DEFECT,
                severity=SeverityLevel.MEDIUM,
                title=f"Elevated Input Sensitivity Detected: {model_file.name}",
                description=(
                    f"Model displayed high prediction instability under bounded input perturbations (epsilon={epsilon}): "
                    f"{flip_rate:.1%} of probes changed predicted class ({flip_count}/{report.probes_count} samples), "
                    f"with average confidence shift of {mean_conf_delta:.4f} and logit amplification of {mean_amplification:.2f}. "
                    f"Indicates fragile decision boundaries or heightened susceptibility to adversarial noise. "
                    f"Note: Elevated sensitivity indicates decision fragility and does not prove a deliberate backdoor."
                ),
                confidence=0.85,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={
                        "epsilon": epsilon,
                        "threshold": flip_threshold,
                        "architecture_id": architecture_id,
                    },
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                anomaly_score=round(flip_rate, 4),
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=context.snapshot.asset_id,
                        sample_path=model_file.name,
                    )
                ],
                evidence_ids=[ev_id],
                details={
                    "epsilon": epsilon,
                    "flip_count": flip_count,
                    "flip_rate": flip_rate,
                    "threshold": flip_threshold,
                    "mean_confidence_delta": mean_conf_delta,
                    "mean_logit_amplification": mean_amplification,
                },
            )
            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                finding_id=fnd_id,
                signal_name="input_perturbation_sensitivity",
                rationale=(
                    f"Evaluated input sensitivity with epsilon={epsilon} on {report.probes_count} probes. "
                    f"Observed flip rate of {flip_rate:.1%} ({flip_count} flips), "
                    f"mean confidence delta of {mean_conf_delta:.4f}, amplification of {mean_amplification:.2f}."
                ),
                created_at=now,
                sample_reference=model_file.name,
                observed_value=round(flip_rate, 4),
                threshold_value=flip_threshold,
                metadata={
                    "epsilon": epsilon,
                    "flip_count": flip_count,
                    "flip_rate": flip_rate,
                    "mean_confidence_delta": mean_conf_delta,
                    "mean_logit_amplification": mean_amplification,
                    "threshold": flip_threshold,
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
                "epsilon": epsilon,
                "flip_rate": flip_rate,
                "mean_logit_amplification": mean_amplification,
                "threshold": flip_threshold,
            },
        )
