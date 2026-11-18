"""Unit tests for the 5 Slice 3B Model Behavioral and Backdoor Detectors."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    FindingCategory,
    IngestionStatus,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.behavioral.common import seed_behavioral_cache_entry
from backend.app.modules.model_integrity.behavioral.worker import BehavioralReport
from backend.app.modules.model_integrity.detectors import (
    BackdoorIndicatorDetector,
    BehavioralFingerprintDetector,
    InputSensitivityDetector,
    PredictionConsistencyDetector,
    ReferenceComparisonDetector,
    register_model_integrity_detectors,
)
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


@pytest.fixture(autouse=True)
def mock_verified_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates verified container isolation for functional detector evaluation tests."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )


@pytest.fixture
def clean_model_snapshot(tmp_path: Path) -> tuple[AssetSnapshot, Path]:
    snap_dir = tmp_path / "snapshots" / "snap_behavioral_clean"
    snap_dir.mkdir(parents=True, exist_ok=True)
    model_file = snap_dir / "model.pt"

    model = SimpleCNN()
    torch.save(model.state_dict(), model_file)

    hasher = hashlib.sha256()
    with open(model_file, "rb") as f:
        hasher.update(f.read())
    file_sha256 = hasher.hexdigest()

    manifest_json = snap_dir / "manifest.json"
    manifest_json.write_text("{}", encoding="utf-8")

    snapshot = AssetSnapshot(
        snapshot_id="snap_behavioral_model_001",
        asset_id="mod_behavioral_clean",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="a" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=model_file.stat().st_size,
        metadata={
            "storage_path": str(model_file),
            "snapshot_path": str(snap_dir),
            "filename": "model.pt",
            "expected_sha256": file_sha256,
            "architecture_id": "simple_cnn",
        },
    )
    return snapshot, model_file


def test_detector_registry_registration_all_nine() -> None:
    """Verifies that all 9 model detectors register cleanly in the registry."""
    reg = DetectorRegistry()
    register_model_integrity_detectors(reg)

    expected_ids = [
        "model_integrity.artifact_hash",
        "model_integrity.checkpoint_structure",
        "model_integrity.state_dict_integrity",
        "model_integrity.architecture_compatibility",
        "model_integrity.behavioral_fingerprint",
        "model_integrity.prediction_consistency",
        "model_integrity.input_sensitivity",
        "model_integrity.backdoor_indicator",
        "model_integrity.reference_comparison",
    ]
    for eid in expected_ids:
        assert reg.has(eid)
        d = reg.get(eid)
        assert d.version == "1.0.0"
        assert d.requirements.requires_model_checkpoint is True


# =========================================================================
# 1. BehavioralFingerprintDetector
# =========================================================================


def test_behavioral_fingerprint_baseline_execution(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that BehavioralFingerprintDetector computes fingerprint and completes cleanly."""
    snapshot, model_file = clean_model_snapshot
    detector = BehavioralFingerprintDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent), "probes_count": 10},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert len(res.evidence) == 0
    assert len(res.metadata["behavioral_fingerprint"]) == 64
    assert res.metadata["comparison_status"] == "SKIPPED_NO_REFERENCE"


def test_behavioral_fingerprint_expected_match(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that matching expected fingerprint produces zero mismatch findings."""
    snapshot, model_file = clean_model_snapshot
    detector = BehavioralFingerprintDetector()

    # Pre-compute fingerprint
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent), "probes_count": 10},
    )
    first_res = detector.evaluate(context)
    observed_fp = first_res.metadata["behavioral_fingerprint"]

    # Evaluate again with matching expected fingerprint
    context_match = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_behavioral_fingerprint": observed_fp,
            "probes_count": 10,
        },
    )
    res = detector.evaluate(context_match)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata["comparison_status"] == "VERIFIED_MATCH"
    assert res.metadata["fingerprint_matches"] is True


def test_behavioral_fingerprint_expected_mismatch(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that mismatched expected fingerprint produces UNAUTHORIZED_MODEL_CHANGE finding."""
    snapshot, model_file = clean_model_snapshot
    detector = BehavioralFingerprintDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_behavioral_fingerprint": "0" * 64,
            "probes_count": 10,
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    assert fnd.severity == SeverityLevel.HIGH
    assert fnd.suggested_disposition == SuggestedDisposition.QUARANTINE
    assert res.metadata["comparison_status"] == "VERIFIED_MISMATCH"
    assert res.metadata["fingerprint_matches"] is False


def test_behavioral_fingerprint_malformed_expected(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that a non-hex or invalid-length expected fingerprint fails closed."""
    snapshot, model_file = clean_model_snapshot
    detector = BehavioralFingerprintDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_behavioral_fingerprint": "not_a_valid_hex_hash",
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "malformed" in (res.status_reason or "").lower()


# =========================================================================
# 2. PredictionConsistencyDetector
# =========================================================================


def test_prediction_consistency_clean_run(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that deterministic evaluation yields no structural defect findings."""
    snapshot, model_file = clean_model_snapshot
    detector = PredictionConsistencyDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent), "probes_count": 10},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata["is_deterministic"] is True
    assert res.metadata["flip_count"] == 0


def test_prediction_consistency_nondeterministic_detected(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that nondeterministic inference behavior triggers STRUCTURAL_DEFECT finding."""
    snapshot, model_file = clean_model_snapshot
    detector = PredictionConsistencyDetector()

    mock_report = BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=10,
        behavioral_fingerprint="a" * 64,
        consistency={
            "is_deterministic": False,
            "flip_count": 4,
            "max_logit_drift": 0.35,
            "repetitions": 3,
        },
        capability_status=CapabilityStatus.COMPLETED,
    )

    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
        },
    )
    seed_behavioral_cache_entry(context, mock_report, candidate_model_file=model_file)

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.STRUCTURAL_DEFECT
    assert fnd.severity == SeverityLevel.HIGH
    assert fnd.details["flip_count"] == 4


# =========================================================================
# 3. InputSensitivityDetector
# =========================================================================


def test_input_sensitivity_clean_run(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that bounded perturbations within tolerance produce zero findings."""
    snapshot, model_file = clean_model_snapshot
    detector = InputSensitivityDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent), "probes_count": 10},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata["flip_rate"] <= 0.40


def test_input_sensitivity_excessive_instability(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that excessive prediction flips under epsilon perturbation trigger STRUCTURAL_DEFECT."""
    snapshot, model_file = clean_model_snapshot
    detector = InputSensitivityDetector()

    mock_report = BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=20,
        behavioral_fingerprint="b" * 64,
        sensitivity={
            "epsilon": 0.02,
            "flip_count": 14,
            "flip_rate": 0.70,
            "mean_confidence_delta": 0.42,
            "mean_logit_amplification": 18.2,
        },
        capability_status=CapabilityStatus.COMPLETED,
    )

    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "sensitivity_flip_threshold": 0.40,
        },
    )
    seed_behavioral_cache_entry(context, mock_report, candidate_model_file=model_file)

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.STRUCTURAL_DEFECT
    assert fnd.severity == SeverityLevel.MEDIUM
    assert fnd.details["flip_rate"] == 0.70


# =========================================================================
# 4. BackdoorIndicatorDetector
# =========================================================================


def test_backdoor_indicator_clean_model(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies clean candidate trigger evaluation logs evidence with epistemic caveat."""
    snapshot, model_file = clean_model_snapshot
    detector = BackdoorIndicatorDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent), "probes_count": 10},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert len(res.evidence) == 0
    assert "caveat" in res.metadata
    assert "does not prove" in res.metadata["caveat"].lower()


def test_backdoor_indicator_triggered_hijacking(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies candidate trigger causing high class concentration triggers SUSPECTED_POISON."""
    snapshot, model_file = clean_model_snapshot
    detector = BackdoorIndicatorDetector()

    mock_report = BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=20,
        behavioral_fingerprint="c" * 64,
        backdoor={
            "evaluated_triggers_count": 4,
            "highest_concentration": 0.95,
            "clean_concentration": 0.15,
            "max_concentration_delta": 0.80,
            "most_suspicious_trigger": "corner_checkerboard",
            "target_class": 5,
            "triggers": {
                "corner_checkerboard": {
                    "target_class": 5,
                    "concentration_ratio": 0.95,
                    "clean_concentration": 0.15,
                    "concentration_delta": 0.80,
                    "avg_confidence": 0.94,
                    "target_sample_count": 19,
                },
                "corner_square": {
                    "target_class": 1,
                    "concentration_ratio": 0.20,
                    "clean_concentration": 0.15,
                    "concentration_delta": 0.05,
                    "avg_confidence": 0.25,
                    "target_sample_count": 4,
                },
            },
        },
        capability_status=CapabilityStatus.COMPLETED,
    )

    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
        },
    )
    seed_behavioral_cache_entry(context, mock_report, candidate_model_file=model_file)

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.SUSPECTED_POISON
    assert fnd.severity == SeverityLevel.CRITICAL  # >= 0.95 triggers CRITICAL
    assert fnd.details["trigger_name"] == "corner_checkerboard"
    assert fnd.details["target_class"] == 5
    assert fnd.suggested_disposition == SuggestedDisposition.QUARANTINE


# =========================================================================
# 5. ReferenceComparisonDetector
# =========================================================================


def test_reference_comparison_skipped_when_no_reference(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ReferenceComparisonDetector returns SKIPPED when no reference model is supplied."""
    snapshot, model_file = clean_model_snapshot
    detector = ReferenceComparisonDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.SKIPPED
    assert "reference" in (res.status_reason or "").lower()


def test_reference_comparison_failed_when_reference_path_missing(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ReferenceComparisonDetector returns FAILED when specified reference file does not exist."""
    snapshot, model_file = clean_model_snapshot
    detector = ReferenceComparisonDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(model_file.parent / "nonexistent_ref.pt"),
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "does not exist" in (res.status_reason or "").lower()


class MockModelRecord:
    def __init__(
        self,
        model_id: str,
        storage_path: str,
        sha256: str,
        status: IngestionStatus = IngestionStatus.ACCEPTED,
    ) -> None:
        self.model_id = model_id
        self.storage_path = storage_path
        self.sha256 = sha256
        self.status = status


class MockModelRepo:
    def __init__(self, models: list[MockModelRecord]) -> None:
        self._models = {m.model_id: m for m in models}

    def get_model(self, model_id: str) -> MockModelRecord | None:
        return self._models.get(model_id)

    def get_model_by_storage_path(self, path: str) -> MockModelRecord | None:
        for m in self._models.values():
            if m.storage_path == path:
                return m
        return None

    def get_model_by_sha256(self, sha: str) -> MockModelRecord | None:
        for m in self._models.values():
            if m.sha256 == sha:
                return m
        return None


def test_reference_comparison_identical_reference(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies independently registered reference model achieves 1.0 agreement with zero findings."""
    snapshot, model_file = clean_model_snapshot
    detector = ReferenceComparisonDetector()

    ref_file = model_file.parent / "ref_identical.pt"
    raw_weights = torch.load(model_file, weights_only=True)
    torch.save({"state_dict": raw_weights, "meta": "test_ref_run"}, ref_file)
    ref_sha256 = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    rec = MockModelRecord("mod_ref_test_001", str(ref_file.resolve()), ref_sha256)
    mock_repo = MockModelRepo([rec])

    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
            "probes_count": 10,
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert len(res.evidence) == 0
    assert res.metadata["top1_agreement_rate"] == 1.0
    assert res.metadata["fingerprints_match"] is True


def test_reference_comparison_divergent_reference(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies behavioral divergence from reference model triggers UNAUTHORIZED_MODEL_CHANGE."""
    snapshot, model_file = clean_model_snapshot
    detector = ReferenceComparisonDetector()

    ref_file = model_file.parent / "ref_divergent.pt"
    ref_file.write_bytes(b"divergent_bytes_content_different_from_model_file")
    ref_sha256 = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    rec = MockModelRecord("mod_ref_diverge_001", str(ref_file.resolve()), ref_sha256)
    mock_repo = MockModelRepo([rec])

    mock_report = BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=20,
        behavioral_fingerprint="d" * 64,
        reference_comparison={
            "reference_checkpoint": str(ref_file),
            "reference_behavioral_fingerprint": "e" * 64,
            "top1_agreement_rate": 0.40,
            "max_logit_divergence": 3.45,
            "mean_logit_divergence": 1.25,
            "mean_confidence_difference": 0.55,
            "fingerprints_match": False,
        },
        capability_status=CapabilityStatus.COMPLETED,
    )

    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
        },
    )
    seed_behavioral_cache_entry(
        context,
        mock_report,
        candidate_model_file=model_file,
        reference_model_file=ref_file,
        reference_sha256=ref_sha256,
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    assert fnd.severity == SeverityLevel.HIGH
    assert fnd.suggested_disposition == SuggestedDisposition.QUARANTINE
    assert fnd.details["top1_agreement_rate"] == 0.40
