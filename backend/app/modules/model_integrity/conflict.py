"""ADR-012 Multi-Detector Visible Conflict Reporting Engine.

Evaluates cross-detector contradictions across structural, cryptographic, and
behavioral evaluation outputs without averaging, blending, or suppressing
original detector findings. Enforces radical epistemic honesty:
- Preserves all original detector findings, severities, and metadata unaltered.
- Preserves capability gating: detectors reporting UNAVAILABLE never trigger false conflicts.
- Emits explicit visible conflict findings under FindingCategory.CONFLICT.
- Distinguishes verified structural facts from behavioral observations and heuristics.
- Reports INCONCLUSIVE when reference baselines or evidence are insufficient.
- Fully auditable bidirectional cross-references across detector runs and findings.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

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

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{3,128}$")


class ConflictType(StrEnum):
    """Formal classification of multi-detector contradictory findings."""

    STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE = "STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE"
    BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER = "BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER"
    HASH_MISMATCH_STATE_DICT_IDENTICAL = "HASH_MISMATCH_STATE_DICT_IDENTICAL"
    INSUFFICIENT_REFERENCE_EVIDENCE = "INSUFFICIENT_REFERENCE_EVIDENCE"


class ModelConflictEngine:
    """Evaluates multi-detector execution results for visible contradictions.

    In accordance with ADR-012, this engine detects epistemological conflicts
    between structural, cryptographic, and behavioral integrity detectors without
    silencing contradictory signals or fabricating mathematical consensus.
    """

    DETECTOR_ID = "model_integrity.conflict"
    VERSION = "1.0.0"

    def evaluate_conflicts(
        self,
        assessment_id: str,
        snapshot_id: str,
        asset_id: str,
        detector_runs: list[dict[str, Any]],
        findings: list[UniversalFinding],
        evidence_records: list[EvidenceRecord],
        context_config: dict[str, Any] | None = None,
        sample_path: str | None = None,
    ) -> tuple[list[UniversalFinding], list[EvidenceRecord]]:
        """Evaluates detector runs and findings for cross-detector contradictions.

        Args:
            assessment_id: Unique identifier of the assessment.
            snapshot_id: Snapshot identifier for affected asset references.
            asset_id: Asset identifier for affected asset references.
            detector_runs: Complete list of detector execution run records.
            findings: Unmodified list of findings produced by all detectors.
            evidence_records: Supporting forensic evidence produced by detectors.
            context_config: Assessment configuration dictionary.
            sample_path: Optional relative path of the candidate model artifact.

        Returns:
            Tuple of (conflict_findings, conflict_evidence_records).
            Does NOT mutate or suppress any of the input findings or evidence.
        """
        cfg = context_config or {}
        runs_by_detector = {run["detector_id"]: run for run in detector_runs}
        findings_by_detector: dict[str, list[UniversalFinding]] = {}
        for f in findings:
            findings_by_detector.setdefault(f.detector_id, []).append(f)

        now = datetime.now(UTC)
        conflict_findings: list[UniversalFinding] = []
        conflict_evidence: list[EvidenceRecord] = []

        # ---------------------------------------------------------------------
        # RULE 1: HASH_MISMATCH_STATE_DICT_IDENTICAL
        # Condition: Artifact file hash mismatch vs reference, but state-dict
        # tensor parameters match authoritative reference.
        # ---------------------------------------------------------------------
        self._evaluate_hash_vs_state_dict_conflict(
            assessment_id=assessment_id,
            asset_id=asset_id,
            now=now,
            runs_by_detector=runs_by_detector,
            findings_by_detector=findings_by_detector,
            sample_path=sample_path,
            conflict_findings=conflict_findings,
            conflict_evidence=conflict_evidence,
        )

        # ---------------------------------------------------------------------
        # RULE 2: STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
        # Condition: State-dict matches reference, but runtime behavioral
        # evaluation diverges on probe suite.
        # ---------------------------------------------------------------------
        self._evaluate_structural_vs_behavioral_conflict(
            assessment_id=assessment_id,
            asset_id=asset_id,
            now=now,
            runs_by_detector=runs_by_detector,
            findings_by_detector=findings_by_detector,
            sample_path=sample_path,
            conflict_findings=conflict_findings,
            conflict_evidence=conflict_evidence,
        )

        # ---------------------------------------------------------------------
        # RULE 3: BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        # Condition: Clean probe predictions show high stability/determinism,
        # but backdoor indicator reveals trigger-specific anomalies.
        # ---------------------------------------------------------------------
        self._evaluate_stability_vs_backdoor_conflict(
            assessment_id=assessment_id,
            asset_id=asset_id,
            now=now,
            runs_by_detector=runs_by_detector,
            findings_by_detector=findings_by_detector,
            sample_path=sample_path,
            conflict_findings=conflict_findings,
            conflict_evidence=conflict_evidence,
        )

        # ---------------------------------------------------------------------
        # RULE 4: INSUFFICIENT_REFERENCE_EVIDENCE -> INCONCLUSIVE
        # Condition: Reference comparison requested or configured, but
        # reference artifacts/metadata are absent or probe confidence is lacking.
        # ---------------------------------------------------------------------
        self._evaluate_inconclusive_reference_evidence(
            assessment_id=assessment_id,
            asset_id=asset_id,
            now=now,
            config=cfg,
            runs_by_detector=runs_by_detector,
            sample_path=sample_path,
            conflict_findings=conflict_findings,
            conflict_evidence=conflict_evidence,
        )

        # Deterministic ordering by title and conflict type
        conflict_findings.sort(key=lambda f: (f.title, f.finding_id))
        conflict_evidence.sort(key=lambda e: (e.signal_name, e.evidence_id))

        return conflict_findings, conflict_evidence

    # =========================================================================
    # Rule Evaluation Internals
    # =========================================================================

    def _evaluate_hash_vs_state_dict_conflict(
        self,
        assessment_id: str,
        asset_id: str,
        now: datetime,
        runs_by_detector: dict[str, dict[str, Any]],
        findings_by_detector: dict[str, list[UniversalFinding]],
        sample_path: str | None,
        conflict_findings: list[UniversalFinding],
        conflict_evidence: list[EvidenceRecord],
    ) -> None:
        """Evaluates contradiction between artifact file digest mismatch and state-dict match."""
        hash_run = runs_by_detector.get("model_integrity.artifact_hash")
        sd_run = runs_by_detector.get("model_integrity.state_dict_integrity")

        if not hash_run or not sd_run:
            return

        # Capability gating: Never emit conflict if either detector is UNAVAILABLE or failed
        if (
            hash_run.get("status") != CapabilityStatus.COMPLETED
            or sd_run.get("status") != CapabilityStatus.COMPLETED
        ):
            return

        hash_metrics = hash_run.get("metrics") or {}
        sd_metrics = sd_run.get("metrics") or {}

        # Condition 1: Artifact hash detector found a mismatch
        hash_mismatch = hash_metrics.get("matched") is False
        hash_findings = findings_by_detector.get("model_integrity.artifact_hash", [])
        has_hash_finding = any(
            f.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
            and "mismatch" in f.title.lower()
            for f in hash_findings
        )
        is_hash_divergent = hash_mismatch or has_hash_finding

        # Condition 2: State-dict integrity verified a match
        sd_match = (
            sd_metrics.get("fingerprint_verified") is True
            or sd_metrics.get("fingerprint_comparison") == "VERIFIED_MATCH"
        )
        sd_findings = findings_by_detector.get("model_integrity.state_dict_integrity", [])
        has_sd_fp_mismatch = any(
            f.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
            and "fingerprint" in f.title.lower()
            for f in sd_findings
        )
        is_state_dict_identical = sd_match and not has_sd_fp_mismatch

        if is_hash_divergent and is_state_dict_identical:
            conflict_type = ConflictType.HASH_MISMATCH_STATE_DICT_IDENTICAL
            stable_seed = f"{assessment_id}:{conflict_type.value}:hash_sd"
            uid = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
            finding_id = f"fnd_conflict_{uid}"
            evidence_id = f"ev_conflict_{uid}"

            conflicting_runs = [hash_run["run_id"], sd_run["run_id"]]
            conflicting_finding_ids = [f.finding_id for f in hash_findings]

            desc = (
                "Contradictory integrity signals detected: The artifact file SHA-256 digest differs "
                "from the approved reference digest, but extracted state-dictionary tensor parameters "
                "and mathematical structure match the authoritative state-dict fingerprint. A file-level "
                "hash mismatch alongside identical tensor parameters does NOT prove malicious tampering—it "
                "often arises from benign checkpoint repackaging (such as zip container compression differences, "
                "metadata reordering, non-weight header metadata changes, or PyTorch serialization version differences). "
                "Human auditor inspection is recommended rather than automatic rejection."
            )

            details: dict[str, Any] = {
                "conflict_type": conflict_type.value,
                "conflicting_detectors": [
                    "model_integrity.artifact_hash",
                    "model_integrity.state_dict_integrity",
                ],
                "conflicting_run_ids": conflicting_runs,
                "conflicting_finding_ids": conflicting_finding_ids,
                "file_hash_metrics": {
                    "expected_sha256": hash_metrics.get("expected_sha256"),
                    "observed_sha256": hash_metrics.get("observed_sha256"),
                },
                "state_dict_metrics": {
                    "state_dict_fingerprint": sd_metrics.get("state_dict_fingerprint"),
                    "fingerprint_verified": sd_metrics.get("fingerprint_verified"),
                    "fingerprint_comparison": sd_metrics.get("fingerprint_comparison"),
                },
            }

            ev = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.METADATA_MISMATCH,
                finding_id=finding_id,
                signal_name="hash_vs_state_dict_contradiction",
                rationale=(
                    f"Artifact file digest '{hash_metrics.get('observed_sha256', '')[:16]}...' mismatched reference, "
                    f"while state-dict fingerprint '{sd_metrics.get('state_dict_fingerprint', '')[:16]}...' verified identical."
                ),
                created_at=now,
                sample_reference=sample_path,
                observed_value=1.0,
                threshold_value=0.0,
                metadata=details,
            )

            fnd = UniversalFinding(
                finding_id=finding_id,
                assessment_id=assessment_id,
                detector_id=self.DETECTOR_ID,
                detector_version=self.VERSION,
                category=FindingCategory.CONFLICT,
                severity=SeverityLevel.MEDIUM,
                title="Conflict: Artifact File Digest Mismatch with Identical State-Dict Fingerprint",
                description=desc,
                confidence=0.95,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"conflict_engine": self.VERSION},
                    detector_config=details,
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                status=CapabilityStatus.CONFLICTED,
                anomaly_score=0.50,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=asset_id,
                        sample_path=sample_path,
                    )
                ],
                evidence_ids=[evidence_id],
                details=details,
            )

            conflict_findings.append(fnd)
            conflict_evidence.append(ev)

    def _evaluate_structural_vs_behavioral_conflict(
        self,
        assessment_id: str,
        asset_id: str,
        now: datetime,
        runs_by_detector: dict[str, dict[str, Any]],
        findings_by_detector: dict[str, list[UniversalFinding]],
        sample_path: str | None,
        conflict_findings: list[UniversalFinding],
        conflict_evidence: list[EvidenceRecord],
    ) -> None:
        """Evaluates contradiction between verified structural tensor match and behavioral divergence."""
        sd_run = runs_by_detector.get("model_integrity.state_dict_integrity")
        if not sd_run or sd_run.get("status") != CapabilityStatus.COMPLETED:
            return

        sd_metrics = sd_run.get("metrics") or {}
        sd_match = (
            sd_metrics.get("fingerprint_verified") is True
            or sd_metrics.get("fingerprint_comparison") == "VERIFIED_MATCH"
        )
        if not sd_match:
            return

        # Check behavioral detectors for divergence
        # 1. reference_comparison
        ref_run = runs_by_detector.get("model_integrity.reference_comparison")
        ref_diverged = False
        ref_divergence_metrics: dict[str, Any] = {}
        conflicting_behavioral_runs: list[str] = []
        conflicting_behavioral_findings: list[UniversalFinding] = []
        divergent_detector_ids: list[str] = []

        if ref_run and ref_run.get("status") == CapabilityStatus.COMPLETED:
            ref_findings = findings_by_detector.get("model_integrity.reference_comparison", [])
            for f in ref_findings:
                if (
                    f.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
                    or "divergence" in f.title.lower()
                ):
                    ref_diverged = True
                    conflicting_behavioral_findings.append(f)

            ref_metrics = ref_run.get("metrics") or {}
            fp_match = ref_metrics.get("fingerprints_match")
            if fp_match is False or ref_diverged:
                ref_diverged = True
                conflicting_behavioral_runs.append(ref_run["run_id"])
                divergent_detector_ids.append("model_integrity.reference_comparison")
                ref_divergence_metrics["reference_comparison"] = {
                    "top1_agreement_rate": ref_metrics.get("top1_agreement_rate"),
                    "max_logit_divergence": ref_metrics.get("max_logit_divergence"),
                    "reference_model_id": ref_metrics.get("reference_model_id"),
                }

        # 2. behavioral_fingerprint
        bfp_run = runs_by_detector.get("model_integrity.behavioral_fingerprint")
        bfp_diverged = False
        if bfp_run and bfp_run.get("status") == CapabilityStatus.COMPLETED:
            bfp_metrics = bfp_run.get("metrics") or {}
            if (
                bfp_metrics.get("comparison_status") == "VERIFIED_MISMATCH"
                or bfp_metrics.get("fingerprint_matches") is False
            ):
                bfp_diverged = True
                conflicting_behavioral_runs.append(bfp_run["run_id"])
                divergent_detector_ids.append("model_integrity.behavioral_fingerprint")
                bfp_findings = findings_by_detector.get(
                    "model_integrity.behavioral_fingerprint", []
                )
                conflicting_behavioral_findings.extend(bfp_findings)
                ref_divergence_metrics["behavioral_fingerprint"] = {
                    "behavioral_fingerprint": bfp_metrics.get("behavioral_fingerprint"),
                    "comparison_status": bfp_metrics.get("comparison_status"),
                }

        # 3. prediction_consistency
        cons_run = runs_by_detector.get("model_integrity.prediction_consistency")
        cons_diverged = False
        if cons_run and cons_run.get("status") == CapabilityStatus.COMPLETED:
            cons_findings = findings_by_detector.get("model_integrity.prediction_consistency", [])
            cons_metrics = cons_run.get("metrics") or {}
            if (
                cons_metrics.get("is_deterministic") is False
                or cons_metrics.get("flip_count", 0) > 0
                or any(
                    f.category == FindingCategory.STRUCTURAL_DEFECT
                    and "inconsistent" in f.title.lower()
                    for f in cons_findings
                )
            ):
                cons_diverged = True
                conflicting_behavioral_runs.append(cons_run["run_id"])
                divergent_detector_ids.append("model_integrity.prediction_consistency")
                conflicting_behavioral_findings.extend(cons_findings)
                ref_divergence_metrics["prediction_consistency"] = {
                    "is_deterministic": cons_metrics.get("is_deterministic"),
                    "flip_count": cons_metrics.get("flip_count"),
                    "max_logit_drift": cons_metrics.get("max_logit_drift"),
                }

        if ref_diverged or bfp_diverged or cons_diverged:
            conflict_type = ConflictType.STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
            det_sorted = sorted(divergent_detector_ids)
            stable_seed = f"{assessment_id}:{conflict_type.value}:{':'.join(det_sorted)}"
            uid = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
            finding_id = f"fnd_conflict_{uid}"
            evidence_id = f"ev_conflict_{uid}"

            desc = (
                "Contradictory integrity signals detected: Checkpoint state-dictionary tensor parameters "
                "and mathematical structure match the authoritative reference, but runtime behavioral evaluation "
                "diverged across reference probes. Structural tensor equivalence is verified while runtime behavior "
                "diverges. This discrepancy may indicate non-weight code execution differences, custom execution layer "
                "logic, runtime environment divergences, or hardware precision differences. Further forensic review is "
                "required; this finding does not prove malicious code injection without dynamic code execution analysis."
            )

            details: dict[str, Any] = {
                "conflict_type": conflict_type.value,
                "conflicting_detectors": [
                    "model_integrity.state_dict_integrity",
                    *det_sorted,
                ],
                "conflicting_run_ids": [sd_run["run_id"], *conflicting_behavioral_runs],
                "conflicting_finding_ids": [f.finding_id for f in conflicting_behavioral_findings],
                "state_dict_fingerprint": sd_metrics.get("state_dict_fingerprint"),
                "divergence_metrics": ref_divergence_metrics,
            }

            ev = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.STATISTICAL_TEST,
                finding_id=finding_id,
                signal_name="structural_match_behavioral_divergence",
                rationale=(
                    f"State-dict fingerprint '{sd_metrics.get('state_dict_fingerprint', '')[:16]}...' verified match, "
                    f"while behavioral probes diverged across detectors: {', '.join(det_sorted)}."
                ),
                created_at=now,
                sample_reference=sample_path,
                observed_value=1.0,
                threshold_value=0.0,
                metadata=details,
            )

            fnd = UniversalFinding(
                finding_id=finding_id,
                assessment_id=assessment_id,
                detector_id=self.DETECTOR_ID,
                detector_version=self.VERSION,
                category=FindingCategory.CONFLICT,
                severity=SeverityLevel.HIGH,
                title="Conflict: Structural State-Dict Match with Runtime Behavioral Divergence",
                description=desc,
                confidence=0.92,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"conflict_engine": self.VERSION},
                    detector_config=details,
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                status=CapabilityStatus.CONFLICTED,
                anomaly_score=0.80,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=asset_id,
                        sample_path=sample_path,
                    )
                ],
                evidence_ids=[evidence_id],
                details=details,
            )

            conflict_findings.append(fnd)
            conflict_evidence.append(ev)

    def _evaluate_stability_vs_backdoor_conflict(
        self,
        assessment_id: str,
        asset_id: str,
        now: datetime,
        runs_by_detector: dict[str, dict[str, Any]],
        findings_by_detector: dict[str, list[UniversalFinding]],
        sample_path: str | None,
        conflict_findings: list[UniversalFinding],
        conflict_evidence: list[EvidenceRecord],
    ) -> None:
        """Evaluates contradiction between clean prediction stability and backdoor trigger detection."""
        cons_run = runs_by_detector.get("model_integrity.prediction_consistency")

        if not cons_run:
            return

        # Capability gating: Never emit conflict if prediction consistency is not COMPLETED
        if cons_run.get("status") != CapabilityStatus.COMPLETED:
            return

        cons_metrics = cons_run.get("metrics") or {}

        # Condition 1: High prediction stability on clean probes (100% determinism, 0 flips, no defects)
        cons_findings = findings_by_detector.get("model_integrity.prediction_consistency", [])
        is_clean_stable = (
            cons_metrics.get("is_deterministic", False) is True
            and cons_metrics.get("flip_count", 0) == 0
            and len(cons_findings) == 0
        )
        if not is_clean_stable:
            return

        # Condition 2: Backdoor / poisoning detectors that may flag active SUSPECTED_POISON
        candidate_backdoor_detectors = [
            "model_integrity.backdoor_indicator",
            "model_integrity.spectral_signatures",
            "model_integrity.activation_clustering",
        ]

        active_poison_detectors: list[str] = []
        conflicting_poison_runs: list[str] = []
        conflicting_poison_findings: list[UniversalFinding] = []

        for det_id in candidate_backdoor_detectors:
            run = runs_by_detector.get(det_id)
            if not run:
                continue

            # Epistemic guardrail: Never trigger false conflicts if detector failed,
            # was skipped, or is UNAVAILABLE / INCONCLUSIVE.
            if run.get("status") != CapabilityStatus.COMPLETED:
                continue

            metrics = run.get("metrics") or {}
            det_findings = findings_by_detector.get(det_id, [])
            poison_findings = [
                f for f in det_findings if f.category == FindingCategory.SUSPECTED_POISON
            ]

            is_poison_flagged = bool(
                poison_findings
                or (
                    det_id == "model_integrity.backdoor_indicator"
                    and (
                        metrics.get("has_backdoor_indicators") is True
                        or bool(metrics.get("flagged_triggers"))
                    )
                )
                or (
                    det_id == "model_integrity.spectral_signatures"
                    and (
                        metrics.get("has_spectral_outliers") is True
                        or bool(metrics.get("flagged_classes"))
                    )
                )
                or (
                    det_id == "model_integrity.activation_clustering"
                    and (
                        metrics.get("has_cluster_anomalies") is True
                        or bool(metrics.get("flagged_classes"))
                    )
                )
            )

            if is_poison_flagged:
                active_poison_detectors.append(det_id)
                conflicting_poison_runs.append(run["run_id"])
                conflicting_poison_findings.extend(poison_findings)

        if not active_poison_detectors:
            return

        conflict_type = ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        if active_poison_detectors == ["model_integrity.backdoor_indicator"]:
            det_key = "cons_bd"
        else:
            short_names = sorted(d.split(".")[-1] for d in active_poison_detectors)
            det_key = f"cons_{'_'.join(short_names)}"

        stable_seed = f"{assessment_id}:{conflict_type.value}:{det_key}"
        uid = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
        finding_id = f"fnd_conflict_{uid}"
        evidence_id = f"ev_conflict_{uid}"

        conflicting_runs = [cons_run["run_id"], *conflicting_poison_runs]
        conflicting_finding_ids = [f.finding_id for f in conflicting_poison_findings]

        det_names = ", ".join(sorted(active_poison_detectors))
        desc = (
            "Contradictory integrity signals detected: The model exhibits high inference determinism and "
            "prediction stability across clean reference probes, but backdoor indicator and poisoning evaluation "
            f"({det_names}) revealed suspicious target-class concentration under candidate trigger patterns, "
            "spectral signature outliers, or representation clusters. High statistical prediction "
            "stability on clean inputs does NOT preclude dormant backdoor triggers, nor does candidate trigger "
            "concentration heuristic alone prove an implanted exploit without threat-model validation. Both findings "
            "are preserved without suppression for forensic triage."
        )

        bd_run = runs_by_detector.get("model_integrity.backdoor_indicator")
        bd_metrics = bd_run.get("metrics") or {} if bd_run else {}

        details: dict[str, Any] = {
            "conflict_type": conflict_type.value,
            "conflicting_detectors": [
                "model_integrity.prediction_consistency",
                *sorted(active_poison_detectors),
            ],
            "conflicting_run_ids": conflicting_runs,
            "conflicting_finding_ids": conflicting_finding_ids,
            "clean_stability_metrics": {
                "is_deterministic": cons_metrics.get("is_deterministic"),
                "flip_count": cons_metrics.get("flip_count"),
                "max_logit_drift": cons_metrics.get("max_logit_drift"),
            },
            "backdoor_trigger_metrics": {
                "flagged_triggers": bd_metrics.get("flagged_triggers"),
                "triggers_count": bd_metrics.get("triggers_count"),
            },
            "backdoor_detectors_metrics": {
                det_id: (runs_by_detector[det_id].get("metrics") or {})
                for det_id in sorted(active_poison_detectors)
            },
        }

        # Format rationale and observed value
        if "model_integrity.backdoor_indicator" in active_poison_detectors and bd_metrics.get(
            "flagged_triggers"
        ):
            rationale = (
                f"Clean evaluation showed 100% determinism (0 flips), but trigger patterns "
                f"flagged by backdoor detector: {bd_metrics.get('flagged_triggers')}."
            )
            observed_value = float(len(bd_metrics.get("flagged_triggers", [])))
        else:
            rationale = (
                f"Clean evaluation showed 100% determinism (0 flips), but backdoor/poisoning anomalies "
                f"were flagged by: {det_names}."
            )
            observed_value = float(len(conflicting_poison_findings) or len(active_poison_detectors))

        ev = EvidenceRecord(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.METRIC_DISTRIBUTION,
            finding_id=finding_id,
            signal_name="clean_stability_vs_backdoor_trigger",
            rationale=rationale,
            created_at=now,
            sample_reference=sample_path,
            observed_value=observed_value,
            threshold_value=0.0,
            metadata=details,
        )

        fnd = UniversalFinding(
            finding_id=finding_id,
            assessment_id=assessment_id,
            detector_id=self.DETECTOR_ID,
            detector_version=self.VERSION,
            category=FindingCategory.CONFLICT,
            severity=SeverityLevel.HIGH,
            title="Conflict: Clean Probe Prediction Stability Contradicts Backdoor Trigger Indicators",
            description=desc,
            confidence=0.90,
            reproducibility=ReproducibilityMetadata(
                execution_timestamp=now,
                software_versions={"conflict_engine": self.VERSION},
                detector_config=details,
            ),
            created_at=now,
            suggested_disposition=SuggestedDisposition.REVIEW,
            status=CapabilityStatus.CONFLICTED,
            anomaly_score=0.75,
            affected_assets=[
                AffectedAssetReference(
                    asset_id=asset_id,
                    sample_path=sample_path,
                )
            ],
            evidence_ids=[evidence_id],
            details=details,
        )

        conflict_findings.append(fnd)
        conflict_evidence.append(ev)

    def _evaluate_inconclusive_reference_evidence(
        self,
        assessment_id: str,
        asset_id: str,
        now: datetime,
        config: dict[str, Any],
        runs_by_detector: dict[str, dict[str, Any]],
        sample_path: str | None,
        conflict_findings: list[UniversalFinding],
        conflict_evidence: list[EvidenceRecord],
    ) -> None:
        """Evaluates whether reference baseline evidence is insufficient, producing INCONCLUSIVE outcome."""
        # Check if reference comparison was requested or expected
        requested_detectors = config.get("detectors") or []
        ref_requested = (
            "model_integrity.reference_comparison" in requested_detectors
            or config.get("require_reference_validation", False) is True
        )
        ref_run = runs_by_detector.get("model_integrity.reference_comparison")

        # Condition 1: Reference comparison explicitly skipped due to missing reference model
        is_missing_reference = False
        reason = ""

        if ref_run and ref_run.get("status") == CapabilityStatus.SKIPPED:
            status_reason = str(ref_run.get("status_reason", "")).lower()
            if "no trusted reference" in status_reason or "no reference" in status_reason:
                is_missing_reference = True
                reason = ref_run.get("status_reason", "No reference model provided.")
        elif ref_requested and not ref_run:
            is_missing_reference = True
            reason = "Reference comparison was requested but no reference model was provisioned."

        # Condition 2: Insufficient probe count or statistical sample size
        probe_count = int(config.get("probe_count", config.get("probes_count", 0)) or 0)
        min_probes = int(config.get("min_statistically_sound_probes", 10))
        is_underpowered = probe_count > 0 and probe_count < min_probes

        if is_missing_reference and ref_requested:
            conflict_type = ConflictType.INSUFFICIENT_REFERENCE_EVIDENCE
            stable_seed = f"{assessment_id}:{conflict_type.value}:missing_ref"
            uid = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
            finding_id = f"fnd_inconclusive_{uid}"
            evidence_id = f"ev_inconclusive_{uid}"

            desc = (
                f"Comparative conflict evaluation cannot reach a conclusive finding: {reason} "
                "In accordance with ADR-012, missing reference baselines are explicitly reported as "
                "INCONCLUSIVE rather than fabricating a passing verdict or false conflict."
            )

            details: dict[str, Any] = {
                "conflict_type": conflict_type.value,
                "reason": reason,
                "detector_id": "model_integrity.reference_comparison",
            }

            ev = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.METADATA_MISMATCH,
                finding_id=finding_id,
                signal_name="missing_authoritative_reference_baseline",
                rationale=reason,
                created_at=now,
                sample_reference=sample_path,
                observed_value=0.0,
                threshold_value=1.0,
                metadata=details,
            )

            fnd = UniversalFinding(
                finding_id=finding_id,
                assessment_id=assessment_id,
                detector_id=self.DETECTOR_ID,
                detector_version=self.VERSION,
                category=FindingCategory.CONFLICT,
                severity=SeverityLevel.INFORMATIONAL,
                title="Inconclusive Assessment: Missing Authoritative Reference Model",
                description=desc,
                confidence=0.50,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"conflict_engine": self.VERSION},
                    detector_config=details,
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                status=CapabilityStatus.INCONCLUSIVE,
                anomaly_score=None,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=asset_id,
                        sample_path=sample_path,
                    )
                ],
                evidence_ids=[evidence_id],
                details=details,
            )

            conflict_findings.append(fnd)
            conflict_evidence.append(ev)

        elif is_underpowered and ref_requested:
            conflict_type = ConflictType.INSUFFICIENT_REFERENCE_EVIDENCE
            stable_seed = f"{assessment_id}:{conflict_type.value}:underpowered"
            uid = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
            finding_id = f"fnd_inconclusive_{uid}"
            evidence_id = f"ev_inconclusive_{uid}"

            desc = (
                f"Statistical sample size underpowered: Probes count ({probe_count}) is below minimum "
                f"required threshold ({min_probes}) for statistical confidence. Outcome is reported as "
                "INCONCLUSIVE to prevent premature conclusions."
            )

            details = {
                "conflict_type": conflict_type.value,
                "probe_count": probe_count,
                "min_probes": min_probes,
            }

            ev = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.STATISTICAL_TEST,
                finding_id=finding_id,
                signal_name="insufficient_probe_sample_size",
                rationale=f"Evaluated only {probe_count} probes; minimum threshold is {min_probes}.",
                created_at=now,
                sample_reference=sample_path,
                observed_value=float(probe_count),
                threshold_value=float(min_probes),
                metadata=details,
            )

            fnd = UniversalFinding(
                finding_id=finding_id,
                assessment_id=assessment_id,
                detector_id=self.DETECTOR_ID,
                detector_version=self.VERSION,
                category=FindingCategory.CONFLICT,
                severity=SeverityLevel.INFORMATIONAL,
                title="Inconclusive Assessment: Insufficient Probe Sample Size for Statistical Confidence",
                description=desc,
                confidence=0.40,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"conflict_engine": self.VERSION},
                    detector_config=details,
                ),
                created_at=now,
                suggested_disposition=SuggestedDisposition.REVIEW,
                status=CapabilityStatus.INCONCLUSIVE,
                anomaly_score=None,
                affected_assets=[
                    AffectedAssetReference(
                        asset_id=asset_id,
                        sample_path=sample_path,
                    )
                ],
                evidence_ids=[evidence_id],
                details=details,
            )

            conflict_findings.append(fnd)
            conflict_evidence.append(ev)


__all__ = [
    "ConflictType",
    "ModelConflictEngine",
]
