"""Backdoor indicator detector for PyTorch computer vision models.

Evaluates whether localized geometric or blended candidate trigger patterns cause
diverse reference probe predictions to collapse onto a single target class.
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


class BackdoorIndicatorDetector(BaseDetector):
    """Evaluates candidate trigger patterns for targeted class hijacking and backdoor indicators."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.backdoor_indicator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_model_checkpoint=True)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Evaluates target-class concentration and trigger vulnerability across candidate patterns."""
        dispatch = get_or_run_behavioral_report(self.detector_id, self.version, context)
        if isinstance(dispatch, DetectorResult):
            return dispatch

        model_file, architecture_id, report = dispatch
        now = datetime.now(UTC)
        assessment_id = context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []

        backdoor = report.backdoor
        triggers = backdoor.get("triggers", {})

        conc_thresh = float(context.config.get("backdoor_concentration_threshold", 0.85))
        delta_thresh = float(context.config.get("backdoor_delta_threshold", 0.50))
        conf_thresh = float(context.config.get("backdoor_confidence_threshold", 0.80))

        flagged_triggers: list[str] = []

        for t_name, t_data in triggers.items():
            conc = float(t_data.get("concentration_ratio", 0.0))
            delta = float(t_data.get("concentration_delta", 0.0))
            avg_conf = float(t_data.get("avg_confidence", 0.0))
            target_class = t_data.get("target_class")
            sample_count = t_data.get("target_sample_count", 0)
            clean_conc = float(t_data.get("clean_concentration", 0.0))

            if conc >= conc_thresh and delta >= delta_thresh and avg_conf >= conf_thresh:
                flagged_triggers.append(t_name)
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_backdoor_{uid}"
                ev_id = f"ev_backdoor_{uid}"

                severity = SeverityLevel.CRITICAL if conc >= 0.95 else SeverityLevel.HIGH
                confidence_score = min(0.95, round(0.50 + delta * 0.45, 2))

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                    finding_id=fnd_id,
                    signal_name="backdoor_trigger_class_concentration",
                    rationale=(
                        f"Candidate trigger '{t_name}' caused {conc:.1%} ({sample_count}/{report.probes_count}) "
                        f"of diverse probes to collapse onto target class {target_class} "
                        f"(baseline clean concentration: {clean_conc:.1%}, delta: +{delta:.1%}, avg confidence: {avg_conf:.2f})."
                    ),
                    created_at=now,
                    sample_reference=model_file.name,
                    observed_value=round(conc, 4),
                    threshold_value=conc_thresh,
                    metadata={
                        "trigger_name": t_name,
                        "target_class": target_class,
                        "concentration_ratio": conc,
                        "clean_concentration": clean_conc,
                        "concentration_delta": delta,
                        "avg_confidence": avg_conf,
                        "sample_count": sample_count,
                        "total_probes": report.probes_count,
                    },
                )
                evidence_list.append(ev)

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=assessment_id,
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.SUSPECTED_POISON,
                    severity=severity,
                    title=f"Potential Backdoor Indicator: Targeted Class Concentration under Trigger '{t_name}'",
                    description=(
                        f"Model displayed high target-class concentration under candidate trigger '{t_name}': "
                        f"{conc:.1%} of probes ({sample_count}/{report.probes_count}) collapsed onto class {target_class} "
                        f"with average confidence {avg_conf:.2f} (clean baseline concentration was {clean_conc:.1%}). "
                        f"Targeted class hijacking across independent samples strongly indicates an implanted trigger backdoor. "
                        f"Limitations: Trigger analysis evaluates candidate patterns; absence of triggers does not prove the model is clean."
                    ),
                    confidence=confidence_score,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={
                            "trigger_name": t_name,
                            "concentration_threshold": conc_thresh,
                            "delta_threshold": delta_thresh,
                            "confidence_threshold": conf_thresh,
                            "architecture_id": architecture_id,
                        },
                    ),
                    created_at=now,
                    suggested_disposition=SuggestedDisposition.QUARANTINE,
                    anomaly_score=round(conc, 4),
                    affected_assets=[
                        AffectedAssetReference(
                            asset_id=context.snapshot.asset_id,
                            sample_path=model_file.name,
                        )
                    ],
                    evidence_ids=[ev_id],
                    details={
                        "trigger_name": t_name,
                        "target_class": target_class,
                        "concentration_ratio": conc,
                        "concentration_delta": delta,
                        "avg_confidence": avg_conf,
                    },
                )
                findings.append(fnd)

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "evaluated_triggers": list(triggers.keys()),
                "flagged_triggers": flagged_triggers,
                "highest_concentration": backdoor.get("highest_concentration", 0.0),
                "caveat": "Absence of detected backdoor indicators does not prove absence of backdoors.",
            },
        )
