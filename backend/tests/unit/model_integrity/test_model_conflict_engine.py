"""Unit tests for ADR-012 Multi-Detector Visible Conflict Reporting Engine.

Verifies:
1. HASH_MISMATCH_STATE_DICT_IDENTICAL detection and neutral framing.
2. STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE detection across reference comparison,
   behavioral fingerprint, and prediction consistency.
3. BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER detection and distinction between
   clean determinism and trigger-specific anomalies.
4. Simultaneous multiple conflict detection and matrix combinations.
5. Strict preservation of original findings (immutability, zero suppression).
6. Capability gating: UNAVAILABLE and FAILED detectors never trigger false conflicts.
7. INSUFFICIENT_REFERENCE_EVIDENCE producing INCONCLUSIVE findings.
8. Epistemic guardrail: never infer unauthorized substitution from prediction stability alone.
9. Deterministic finding IDs, sorting, and idempotency.
"""

from datetime import UTC, datetime
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
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.model_integrity.conflict import (
    ConflictType,
    ModelConflictEngine,
)
from backend.app.modules.model_integrity.detectors import (
    register_model_integrity_detectors,
)

_VALID_SHA256_A = "a" * 64
_VALID_SHA256_B = "b" * 64
_VALID_SHA256_FP = "c" * 64


def _make_finding(
    finding_id: str,
    detector_id: str,
    category: FindingCategory,
    severity: SeverityLevel = SeverityLevel.HIGH,
    title: str = "Test Finding",
    description: str = "Test description",
    confidence: float = 0.95,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> UniversalFinding:
    now = datetime.now(UTC)
    return UniversalFinding(
        finding_id=finding_id,
        assessment_id="assess_test_001",
        detector_id=detector_id,
        detector_version="1.0.0",
        category=category,
        severity=severity,
        title=title,
        description=description,
        confidence=confidence,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now,
            seed=42,
            software_versions={"torch": "2.15.0"},
            detector_config=details or {},
        ),
        created_at=now,
        run_id=run_id,
        suggested_disposition=SuggestedDisposition.REVIEW,
        status=CapabilityStatus.COMPLETED,
        affected_assets=[
            AffectedAssetReference(
                asset_id="model_asset_001",
                sample_path="model.pt",
                sha256=_VALID_SHA256_A,
            )
        ],
        details=details or {},
    )


def _make_evidence(
    evidence_id: str,
    finding_id: str,
    evidence_type: EvidenceType = EvidenceType.METRIC_DISTRIBUTION,
    signal_name: str = "test_signal",
    rationale: str = "test rationale",
) -> EvidenceRecord:
    now = datetime.now(UTC)
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        finding_id=finding_id,
        signal_name=signal_name,
        rationale=rationale,
        created_at=now,
        sample_reference="model.pt",
        observed_value=1.0,
        threshold_value=0.0,
    )


class TestModelConflictEngine:
    """Comprehensive test suite for ModelConflictEngine."""

    def setup_method(self) -> None:
        self.engine = ModelConflictEngine()
        self.assessment_id = "assess_test_001"
        self.snapshot_id = "snap_test_001"
        self.asset_id = "model_asset_001"

    # -------------------------------------------------------------------------
    # 1. HASH_MISMATCH_STATE_DICT_IDENTICAL
    # -------------------------------------------------------------------------
    def test_hash_mismatch_state_dict_identical_conflict(self) -> None:
        """Verifies detection of artifact digest mismatch alongside identical state-dict."""
        hash_fnd = _make_finding(
            finding_id="fnd_hash_001",
            detector_id="model_integrity.artifact_hash",
            category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            severity=SeverityLevel.CRITICAL,
            title="Model Artifact Digest Mismatch: model.pt",
            description="Digest mismatch vs approved reference.",
            run_id="run_hash_001",
        )
        detector_runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "matched": False,
                    "expected_sha256": _VALID_SHA256_A,
                    "observed_sha256": _VALID_SHA256_B,
                },
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "state_dict_fingerprint": _VALID_SHA256_FP,
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                    "finite_weights_verified": True,
                },
            },
        ]
        orig_findings = [hash_fnd]
        orig_evidence = [_make_evidence("ev_hash_001", hash_fnd.finding_id)]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=orig_findings,
            evidence_records=orig_evidence,
            sample_path="model.pt",
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.CONFLICTED
        assert c.severity == SeverityLevel.MEDIUM
        assert c.suggested_disposition == SuggestedDisposition.REVIEW
        assert c.details["conflict_type"] == ConflictType.HASH_MISMATCH_STATE_DICT_IDENTICAL
        assert "repackaging" in c.description.lower()
        assert "not prove malicious" in c.description.lower()
        assert "model_integrity.artifact_hash" in c.details["conflicting_detectors"]
        assert "model_integrity.state_dict_integrity" in c.details["conflicting_detectors"]
        assert "run_hash_001" in c.details["conflicting_run_ids"]
        assert "run_sd_001" in c.details["conflicting_run_ids"]
        assert "fnd_hash_001" in c.details["conflicting_finding_ids"]

        assert len(ev_records) == 1
        assert ev_records[0].finding_id == c.finding_id
        assert ev_records[0].evidence_type == EvidenceType.METADATA_MISMATCH

        # Check immutability of input findings
        assert len(orig_findings) == 1
        assert orig_findings[0].category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE

    # -------------------------------------------------------------------------
    # 2. STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
    # -------------------------------------------------------------------------
    def test_structural_match_behavioral_divergence_reference_comparison(self) -> None:
        """Verifies state-dict match + reference behavioral comparison divergence conflict."""
        ref_fnd = _make_finding(
            finding_id="fnd_ref_001",
            detector_id="model_integrity.reference_comparison",
            category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Reference Model Behavioral Divergence",
            description="Top-1 agreement dropped below threshold.",
            run_id="run_ref_001",
        )
        detector_runs = [
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "state_dict_fingerprint": _VALID_SHA256_FP,
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "top1_agreement_rate": 0.40,
                    "max_logit_divergence": 2.50,
                    "fingerprints_match": False,
                    "reference_model_id": "model_ref_001",
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[ref_fnd],
            evidence_records=[],
            sample_path="model.pt",
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.CONFLICTED
        assert c.severity == SeverityLevel.HIGH
        assert c.details["conflict_type"] == ConflictType.STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
        assert "structural tensor equivalence is verified" in c.description.lower()
        assert "does not prove malicious code injection" in c.description.lower()
        assert "model_integrity.reference_comparison" in c.details["conflicting_detectors"]
        assert len(ev_records) == 1
        assert ev_records[0].finding_id == c.finding_id

    def test_structural_match_behavioral_divergence_behavioral_fingerprint(self) -> None:
        """Verifies state-dict match + behavioral fingerprint mismatch conflict."""
        bfp_fnd = _make_finding(
            finding_id="fnd_bfp_001",
            detector_id="model_integrity.behavioral_fingerprint",
            category=FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Behavioral Fingerprint Mismatch",
            run_id="run_bfp_001",
        )
        detector_runs = [
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "state_dict_fingerprint": _VALID_SHA256_FP,
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_bfp_001",
                "detector_id": "model_integrity.behavioral_fingerprint",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "comparison_status": "VERIFIED_MISMATCH",
                    "fingerprint_matches": False,
                    "behavioral_fingerprint": _VALID_SHA256_B,
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[bfp_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        assert (
            conflicts[0].details["conflict_type"]
            == ConflictType.STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
        )
        assert (
            "model_integrity.behavioral_fingerprint"
            in conflicts[0].details["conflicting_detectors"]
        )

    def test_structural_match_behavioral_divergence_prediction_consistency(self) -> None:
        """Verifies state-dict match + non-deterministic prediction consistency conflict."""
        cons_fnd = _make_finding(
            finding_id="fnd_cons_001",
            detector_id="model_integrity.prediction_consistency",
            category=FindingCategory.STRUCTURAL_DEFECT,
            title="Inconsistent Inference Output Detected",
            run_id="run_cons_001",
        )
        detector_runs = [
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "state_dict_fingerprint": _VALID_SHA256_FP,
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "is_deterministic": False,
                    "flip_count": 3,
                    "max_logit_drift": 0.05,
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[cons_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        assert (
            conflicts[0].details["conflict_type"]
            == ConflictType.STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE
        )
        assert (
            "model_integrity.prediction_consistency"
            in conflicts[0].details["conflicting_detectors"]
        )

    # -------------------------------------------------------------------------
    # 3. BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
    # -------------------------------------------------------------------------
    def test_behavioral_stability_backdoor_trigger_conflict(self) -> None:
        """Verifies clean determinism contradicts trigger backdoor indicator."""
        bd_fnd = _make_finding(
            finding_id="fnd_bd_001",
            detector_id="model_integrity.backdoor_indicator",
            category=FindingCategory.SUSPECTED_POISON,
            severity=SeverityLevel.HIGH,
            title="Potential Backdoor Indicator: Targeted Class Concentration under Trigger 'pattern_a'",
            run_id="run_bd_001",
        )
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "is_deterministic": True,
                    "flip_count": 0,
                    "max_logit_drift": 0.0,
                },
            },
            {
                "run_id": "run_bd_001",
                "detector_id": "model_integrity.backdoor_indicator",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "has_backdoor_indicators": True,
                    "flagged_triggers": ["pattern_a"],
                    "triggers_count": 4,
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[bd_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.CONFLICTED
        assert c.severity == SeverityLevel.HIGH
        assert c.details["conflict_type"] == ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        assert "clean reference probes" in c.description.lower()
        assert "candidate trigger patterns" in c.description.lower()
        assert "run_cons_001" in c.details["conflicting_run_ids"]
        assert "run_bd_001" in c.details["conflicting_run_ids"]
        assert "fnd_bd_001" in c.details["conflicting_finding_ids"]
        assert len(ev_records) == 1

    # -------------------------------------------------------------------------
    # 4. Simultaneous Multiple Conflicts
    # -------------------------------------------------------------------------
    def test_multi_detector_simultaneous_conflicts(self) -> None:
        """Verifies multiple independent conflicts emitted simultaneously in one assessment."""
        hash_fnd = _make_finding(
            "fnd_hash_001",
            "model_integrity.artifact_hash",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Model Artifact Digest Mismatch: model.pt",
            run_id="run_hash_001",
        )
        ref_fnd = _make_finding(
            "fnd_ref_001",
            "model_integrity.reference_comparison",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Reference Model Behavioral Divergence",
            run_id="run_ref_001",
        )
        bd_fnd = _make_finding(
            "fnd_bd_001",
            "model_integrity.backdoor_indicator",
            FindingCategory.SUSPECTED_POISON,
            title="Potential Backdoor Indicator: Trigger 'pattern_a'",
            run_id="run_bd_001",
        )
        detector_runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "matched": False,
                    "expected_sha256": _VALID_SHA256_A,
                    "observed_sha256": _VALID_SHA256_B,
                },
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "state_dict_fingerprint": _VALID_SHA256_FP,
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"top1_agreement_rate": 0.50, "fingerprints_match": False},
            },
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"is_deterministic": True, "flip_count": 0},
            },
            {
                "run_id": "run_bd_001",
                "detector_id": "model_integrity.backdoor_indicator",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"has_backdoor_indicators": True, "flagged_triggers": ["pattern_a"]},
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[hash_fnd, ref_fnd, bd_fnd],
            evidence_records=[],
        )

        # All 3 distinct conflict types must be detected
        assert len(conflicts) == 3
        conflict_types = {c.details["conflict_type"] for c in conflicts}
        assert ConflictType.HASH_MISMATCH_STATE_DICT_IDENTICAL in conflict_types
        assert ConflictType.STRUCTURAL_MATCH_BEHAVIORAL_DIVERGENCE in conflict_types
        assert ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER in conflict_types
        assert len(ev_records) == 3

    # -------------------------------------------------------------------------
    # 5. Agreement & No Conflict Scenarios
    # -------------------------------------------------------------------------
    def test_no_conflict_when_all_detectors_agree_clean(self) -> None:
        """Verifies zero conflict findings when all detectors agree the model is clean."""
        detector_runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"matched": True, "sha256": _VALID_SHA256_A},
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"top1_agreement_rate": 1.0, "fingerprints_match": True},
            },
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"is_deterministic": True, "flip_count": 0},
            },
            {
                "run_id": "run_bd_001",
                "detector_id": "model_integrity.backdoor_indicator",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"has_backdoor_indicators": False, "flagged_triggers": []},
            },
        ]
        conflicts, ev = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, detector_runs, [], []
        )
        assert len(conflicts) == 0
        assert len(ev) == 0

    def test_no_conflict_when_all_detectors_agree_tampered(self) -> None:
        """Verifies zero conflict findings when both hash and state-dict agree the model is substituted."""
        hash_fnd = _make_finding(
            "fnd_hash_001",
            "model_integrity.artifact_hash",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Model Artifact Digest Mismatch",
        )
        sd_fnd = _make_finding(
            "fnd_sd_001",
            "model_integrity.state_dict_integrity",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="State-Dict Fingerprint Mismatch",
        )
        detector_runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"matched": False},
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "fingerprint_verified": False,
                    "fingerprint_comparison": "VERIFIED_MISMATCH",
                },
            },
        ]
        conflicts, ev = self.engine.evaluate_conflicts(
            self.assessment_id,
            self.snapshot_id,
            self.asset_id,
            detector_runs,
            [hash_fnd, sd_fnd],
            [],
        )
        # Both agree it is mismatched -> No contradiction
        assert len(conflicts) == 0
        assert len(ev) == 0

    # -------------------------------------------------------------------------
    # 6. Capability Gating: UNAVAILABLE & FAILED Detectors
    # -------------------------------------------------------------------------
    def test_capability_gating_unavailable_state_dict_never_triggers_false_conflict(self) -> None:
        """Verifies that an UNAVAILABLE detector (e.g. unhardened host sandbox) never triggers conflict."""
        hash_fnd = _make_finding(
            "fnd_hash_001",
            "model_integrity.artifact_hash",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Model Artifact Digest Mismatch",
        )
        detector_runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"matched": False},
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "status": CapabilityStatus.UNAVAILABLE,
                "status_reason": "Host platform lacks required sandbox isolation.",
                "metrics": {},
            },
        ]

        conflicts, ev = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, detector_runs, [hash_fnd], []
        )
        assert len(conflicts) == 0
        assert len(ev) == 0

    def test_capability_gating_failed_behavioral_detector_never_triggers_false_conflict(
        self,
    ) -> None:
        """Verifies that a crashed/FAILED behavioral detector never triggers a false conflict."""
        detector_runs = [
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "status": CapabilityStatus.FAILED,
                "status_reason": "Process crashed with OutOfMemoryError",
                "metrics": {},
            },
        ]

        conflicts, ev = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, detector_runs, [], []
        )
        assert len(conflicts) == 0
        assert len(ev) == 0

    # -------------------------------------------------------------------------
    # 7. INSUFFICIENT_REFERENCE_EVIDENCE -> INCONCLUSIVE
    # -------------------------------------------------------------------------
    def test_missing_reference_when_requested_emits_inconclusive(self) -> None:
        """Verifies that missing reference model emits INCONCLUSIVE finding under FindingCategory.CONFLICT."""
        detector_runs = [
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "status": CapabilityStatus.SKIPPED,
                "status_reason": "No trusted reference model specified in assessment configuration.",
                "metrics": {},
            }
        ]
        context_config = {"detectors": ["model_integrity.reference_comparison"]}

        conflicts, ev = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[],
            evidence_records=[],
            context_config=context_config,
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.INCONCLUSIVE
        assert c.severity == SeverityLevel.INFORMATIONAL
        assert c.details["conflict_type"] == ConflictType.INSUFFICIENT_REFERENCE_EVIDENCE
        assert "inconclusive" in c.title.lower()

    def test_underpowered_probe_count_emits_inconclusive(self) -> None:
        """Verifies that an underpowered probe sample size emits an INCONCLUSIVE outcome."""
        detector_runs = [
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"top1_agreement_rate": 1.0},
            }
        ]
        context_config = {
            "require_reference_validation": True,
            "probe_count": 3,
            "min_statistically_sound_probes": 10,
        }

        conflicts, ev = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[],
            evidence_records=[],
            context_config=context_config,
        )

        assert len(conflicts) == 1
        assert conflicts[0].status == CapabilityStatus.INCONCLUSIVE
        assert "underpowered" in conflicts[0].description.lower()

    # -------------------------------------------------------------------------
    # 8. Epistemic Guardrail: Never Infer Substitution From Stability Alone
    # -------------------------------------------------------------------------
    def test_never_infer_model_substitution_from_stability_alone(self) -> None:
        """Verifies engine never claims model substitution or tampering based on prediction stability."""
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"is_deterministic": True, "flip_count": 0},
            }
        ]
        conflicts, ev = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, detector_runs, [], []
        )
        assert len(conflicts) == 0
        for c in conflicts:
            assert "unauthorized model substitution" not in c.description.lower()

    # -------------------------------------------------------------------------
    # 9. Determinism, Ordering, and Idempotency
    # -------------------------------------------------------------------------
    def test_deterministic_ordering_and_idempotency(self) -> None:
        """Verifies repeated evaluation on identical inputs produces deterministic findings and IDs."""
        hash_fnd = _make_finding(
            "fnd_hash_001",
            "model_integrity.artifact_hash",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Model Artifact Digest Mismatch",
            run_id="run_hash_001",
        )
        ref_fnd = _make_finding(
            "fnd_ref_001",
            "model_integrity.reference_comparison",
            FindingCategory.UNAUTHORIZED_MODEL_CHANGE,
            title="Reference Model Behavioral Divergence",
            run_id="run_ref_001",
        )
        runs = [
            {
                "run_id": "run_hash_001",
                "detector_id": "model_integrity.artifact_hash",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"matched": False},
            },
            {
                "run_id": "run_sd_001",
                "detector_id": "model_integrity.state_dict_integrity",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "fingerprint_verified": True,
                    "fingerprint_comparison": "VERIFIED_MATCH",
                },
            },
            {
                "run_id": "run_ref_001",
                "detector_id": "model_integrity.reference_comparison",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"top1_agreement_rate": 0.3, "fingerprints_match": False},
            },
        ]

        conflicts1, ev1 = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, runs, [hash_fnd, ref_fnd], []
        )
        conflicts2, ev2 = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, runs, [hash_fnd, ref_fnd], []
        )

        assert len(conflicts1) == len(conflicts2) == 2
        assert [c.finding_id for c in conflicts1] == [c.finding_id for c in conflicts2]
        assert [c.title for c in conflicts1] == [c.title for c in conflicts2]
        assert [e.evidence_id for e in ev1] == [e.evidence_id for e in ev2]

    # -------------------------------------------------------------------------
    # 10. Phase 4 Task 4.5 Wire-Up Tests (Backdoor Analyzers & Registry)
    # -------------------------------------------------------------------------
    def test_behavioral_stability_spectral_signatures_backdoor_conflict(self) -> None:
        """Verifies clean prediction stability vs spectral signatures poison finding emits conflict."""
        ss_fnd = _make_finding(
            finding_id="fnd_ss_001",
            detector_id="model_integrity.spectral_signatures",
            category=FindingCategory.SUSPECTED_POISON,
            severity=SeverityLevel.HIGH,
            title="Spectral Signatures: Suspected Backdoor Poisoning Detected in Class 1",
            run_id="run_ss_001",
        )
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "is_deterministic": True,
                    "flip_count": 0,
                    "max_logit_drift": 0.0,
                },
            },
            {
                "run_id": "run_ss_001",
                "detector_id": "model_integrity.spectral_signatures",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "has_spectral_outliers": True,
                    "flagged_classes": [1],
                    "classes_analyzed": [0, 1],
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[ss_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.CONFLICTED
        assert c.severity == SeverityLevel.HIGH
        assert c.details["conflict_type"] == ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        assert "model_integrity.prediction_consistency" in c.details["conflicting_detectors"]
        assert "model_integrity.spectral_signatures" in c.details["conflicting_detectors"]
        assert "run_cons_001" in c.details["conflicting_run_ids"]
        assert "run_ss_001" in c.details["conflicting_run_ids"]
        assert "fnd_ss_001" in c.details["conflicting_finding_ids"]
        assert len(ev_records) == 1
        assert ev_records[0].signal_name == "clean_stability_vs_backdoor_trigger"
        assert ev_records[0].finding_id == c.finding_id

    def test_behavioral_stability_activation_clustering_backdoor_conflict(self) -> None:
        """Verifies clean prediction stability vs activation clustering poison finding emits conflict."""
        ac_fnd = _make_finding(
            finding_id="fnd_ac_001",
            detector_id="model_integrity.activation_clustering",
            category=FindingCategory.SUSPECTED_POISON,
            severity=SeverityLevel.HIGH,
            title="Activation Clustering: Suspected Backdoor Poisoning Detected in Class 2",
            run_id="run_ac_001",
        )
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "is_deterministic": True,
                    "flip_count": 0,
                    "max_logit_drift": 0.0,
                },
            },
            {
                "run_id": "run_ac_001",
                "detector_id": "model_integrity.activation_clustering",
                "detector_version": "1.0.0",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {
                    "has_cluster_anomalies": True,
                    "flagged_classes": [2],
                    "classes_analyzed": [0, 1, 2],
                },
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[ac_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.category == FindingCategory.CONFLICT
        assert c.status == CapabilityStatus.CONFLICTED
        assert c.details["conflict_type"] == ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        assert "model_integrity.prediction_consistency" in c.details["conflicting_detectors"]
        assert "model_integrity.activation_clustering" in c.details["conflicting_detectors"]
        assert "run_cons_001" in c.details["conflicting_run_ids"]
        assert "run_ac_001" in c.details["conflicting_run_ids"]
        assert "fnd_ac_001" in c.details["conflicting_finding_ids"]
        assert len(ev_records) == 1

    def test_behavioral_stability_multi_backdoor_analyzers_simultaneous_conflict(self) -> None:
        """Verifies clean prediction stability with both spectral signatures and clustering flags consolidated conflict."""
        ss_fnd = _make_finding(
            finding_id="fnd_ss_001",
            detector_id="model_integrity.spectral_signatures",
            category=FindingCategory.SUSPECTED_POISON,
            run_id="run_ss_001",
        )
        ac_fnd = _make_finding(
            finding_id="fnd_ac_001",
            detector_id="model_integrity.activation_clustering",
            category=FindingCategory.SUSPECTED_POISON,
            run_id="run_ac_001",
        )
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"is_deterministic": True, "flip_count": 0},
            },
            {
                "run_id": "run_ss_001",
                "detector_id": "model_integrity.spectral_signatures",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"has_spectral_outliers": True, "flagged_classes": [1]},
            },
            {
                "run_id": "run_ac_001",
                "detector_id": "model_integrity.activation_clustering",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"has_cluster_anomalies": True, "flagged_classes": [1]},
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            assessment_id=self.assessment_id,
            snapshot_id=self.snapshot_id,
            asset_id=self.asset_id,
            detector_runs=detector_runs,
            findings=[ss_fnd, ac_fnd],
            evidence_records=[],
        )

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.details["conflict_type"] == ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        assert "model_integrity.prediction_consistency" in c.details["conflicting_detectors"]
        assert "model_integrity.spectral_signatures" in c.details["conflicting_detectors"]
        assert "model_integrity.activation_clustering" in c.details["conflicting_detectors"]
        assert "run_cons_001" in c.details["conflicting_run_ids"]
        assert "run_ss_001" in c.details["conflicting_run_ids"]
        assert "run_ac_001" in c.details["conflicting_run_ids"]
        assert "fnd_ss_001" in c.details["conflicting_finding_ids"]
        assert "fnd_ac_001" in c.details["conflicting_finding_ids"]

    def test_capability_gating_inconclusive_or_unavailable_backdoor_never_triggers_conflict(
        self,
    ) -> None:
        """Verifies that INCONCLUSIVE or UNAVAILABLE backdoor analyzers never trigger false conflicts."""
        detector_runs = [
            {
                "run_id": "run_cons_001",
                "detector_id": "model_integrity.prediction_consistency",
                "status": CapabilityStatus.COMPLETED,
                "metrics": {"is_deterministic": True, "flip_count": 0},
            },
            {
                "run_id": "run_ss_001",
                "detector_id": "model_integrity.spectral_signatures",
                "status": CapabilityStatus.INCONCLUSIVE,
                "status_reason": "Per-class sample density < 10 samples.",
                "metrics": {},
            },
            {
                "run_id": "run_ac_001",
                "detector_id": "model_integrity.activation_clustering",
                "status": CapabilityStatus.UNAVAILABLE,
                "status_reason": "Host platform lacks required sandbox isolation.",
                "metrics": {},
            },
        ]

        conflicts, ev_records = self.engine.evaluate_conflicts(
            self.assessment_id, self.snapshot_id, self.asset_id, detector_runs, [], []
        )

        assert len(conflicts) == 0
        assert len(ev_records) == 0

    def test_central_detector_registry_registers_all_eleven_detectors(self) -> None:
        """Verifies that register_model_integrity_detectors registers all 11 detectors."""
        registry = DetectorRegistry()
        register_model_integrity_detectors(registry)

        expected_detector_ids = {
            "model_integrity.artifact_hash",
            "model_integrity.checkpoint_structure",
            "model_integrity.state_dict_integrity",
            "model_integrity.architecture_compatibility",
            "model_integrity.behavioral_fingerprint",
            "model_integrity.prediction_consistency",
            "model_integrity.input_sensitivity",
            "model_integrity.backdoor_indicator",
            "model_integrity.reference_comparison",
            "model_integrity.spectral_signatures",
            "model_integrity.activation_clustering",
        }

        all_registered = registry.list_all()
        assert len(all_registered) == 11
        registered_ids = {d.detector_id for d in all_registered}
        assert registered_ids == expected_detector_ids
