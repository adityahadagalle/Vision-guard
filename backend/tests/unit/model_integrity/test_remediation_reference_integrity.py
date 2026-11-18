"""Unit tests verifying remediation of FIND-3B-01: Trusted Reference Integrity & Provenance.

Adversarial test cases:
1. Candidate path supplied as its own reference -> FAILED (self-reference path).
2. Same file reached through a different path -> FAILED (self-reference path).
3. Same content copied to a different path -> FAILED (self-reference content digest).
4. Reference file modified after repository record created -> FAILED (digest mismatch).
5. Missing repository record -> FAILED (unregistered reference).
6. Missing reference -> SKIPPED.
7. Valid independently registered reference -> COMPLETED.
8. Reference digest mismatch -> FAILED.
9. Reference record in REJECTED status -> FAILED.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    IngestionStatus,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.behavioral.worker import BehavioralReport
from backend.app.modules.model_integrity.detectors.reference_comparison import (
    ReferenceComparisonDetector,
)
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


@pytest.fixture(autouse=True)
def mock_verified_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        ),
    )


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


@pytest.fixture
def clean_model_and_snapshot(tmp_path: Path) -> tuple[AssetSnapshot, Path]:
    snap_dir = tmp_path / "model_snap"
    snap_dir.mkdir(parents=True)
    model_file = snap_dir / "candidate_model.pt"

    cnn = SimpleCNN()
    torch.save(cnn.state_dict(), model_file)
    cand_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()

    snap = AssetSnapshot(
        snapshot_id="snap_cand_001",
        asset_id="mod_cand_001",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="c" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=model_file.stat().st_size,
        metadata={"storage_path": str(model_file), "sha256": cand_sha256},
    )
    return snap, model_file


def test_missing_reference_returns_skipped(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that missing reference model configuration produces CapabilityStatus.SKIPPED."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()
    context = AssessmentContext(snapshot=snap, config={"snapshot_root": str(model_file.parent)})
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.SKIPPED
    assert "No trusted reference model specified" in (res.status_reason or "")


def test_candidate_path_supplied_as_its_own_reference_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that passing candidate's exact path as reference is rejected."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()
    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(model_file),
            "_model_repo": MockModelRepo([]),
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "cannot serve as its own reference model" in (res.status_reason or "")
    assert res.metadata.get("error_type") == "SELF_REFERENCE_PATH_DETECTED"


def test_same_file_reached_through_different_path_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies that relative path resolution resolving to candidate file is rejected."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    # Create redundant relative path pointing to same canonical file
    redundant_path = model_file.parent / "." / model_file.name

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(redundant_path),
            "_model_repo": MockModelRepo([]),
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert res.metadata.get("error_type") == "SELF_REFERENCE_PATH_DETECTED"


def test_same_content_copied_to_different_path_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
    tmp_path: Path,
) -> None:
    """Verifies that copying candidate bytes to another path is rejected by content-hash self-reference guard."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    # Exact byte duplicate in different directory
    copy_path = tmp_path / "copied_candidate.pt"
    copy_path.write_bytes(model_file.read_bytes())
    copy_sha = hashlib.sha256(copy_path.read_bytes()).hexdigest()

    rec = MockModelRecord("mod_copy_ref", str(copy_path.resolve()), copy_sha)
    mock_repo = MockModelRepo([rec])

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(copy_path),
            "_model_repo": mock_repo,
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "reference content SHA-256 digest is identical to candidate checkpoint" in (
        res.status_reason or ""
    )
    assert res.metadata.get("error_type") == "SELF_REFERENCE_DIGEST_DETECTED"


def test_missing_repository_record_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
    tmp_path: Path,
) -> None:
    """Verifies that reference model lacking an authoritative ModelRepository record fails closed."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    unregistered_ref = tmp_path / "unregistered_ref.pt"
    torch.save({"fc.weight": torch.randn(2, 2)}, unregistered_ref)

    mock_repo = MockModelRepo([])  # Empty repo

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(unregistered_ref),
            "_model_repo": mock_repo,
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "has no authoritative record in ModelRepository" in (res.status_reason or "")
    assert res.metadata.get("error_type") == "UNREGISTERED_REFERENCE_MODEL"


def test_reference_file_modified_after_registration_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
    tmp_path: Path,
) -> None:
    """Verifies that reference file tampered on disk after registration is detected and rejected."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    ref_file = tmp_path / "golden_ref.pt"
    torch.save({"data": 123}, ref_file)
    original_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    rec = MockModelRecord("mod_golden_001", str(ref_file.resolve()), original_sha)
    mock_repo = MockModelRepo([rec])

    # Tamper the file on disk
    ref_file.write_bytes(b"tampered_bytes_after_registration")

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "does not match authoritative ModelRepository digest" in (res.status_reason or "")
    assert res.metadata.get("error_type") == "REFERENCE_DIGEST_MISMATCH"


def test_reference_model_not_accepted_status_fails_closed(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
    tmp_path: Path,
) -> None:
    """Verifies that reference model with status != ACCEPTED (e.g. REJECTED) fails closed."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    ref_file = tmp_path / "rejected_ref.pt"
    torch.save({"data": 456}, ref_file)
    ref_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    rec = MockModelRecord(
        "mod_rejected_001",
        str(ref_file.resolve()),
        ref_sha,
        status=IngestionStatus.REJECTED,
    )
    mock_repo = MockModelRepo([rec])

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "not in ACCEPTED state" in (res.status_reason or "")
    assert res.metadata.get("error_type") == "REFERENCE_NOT_ACCEPTED"


def test_valid_independently_registered_reference_succeeds(
    clean_model_and_snapshot: tuple[AssetSnapshot, Path],
    tmp_path: Path,
) -> None:
    """Verifies that an independently registered reference model with valid digest executes comparison."""
    snap, model_file = clean_model_and_snapshot
    detector = ReferenceComparisonDetector()

    ref_file = tmp_path / "valid_authorized_ref.pt"
    # Independent weights with differing file bytes
    cand_weights = torch.load(model_file, weights_only=True)
    torch.save({"state_dict": cand_weights, "extra_meta": "authorized_run"}, ref_file)
    ref_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()

    rec = MockModelRecord("mod_authorized_001", str(ref_file.resolve()), ref_sha)
    mock_repo = MockModelRepo([rec])

    mock_report = BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=30,
        behavioral_fingerprint="f" * 64,
        reference_comparison={
            "reference_checkpoint": str(ref_file),
            "reference_behavioral_fingerprint": "f" * 64,
            "top1_agreement_rate": 1.0,
            "max_logit_divergence": 0.0,
            "mean_logit_divergence": 0.0,
            "mean_confidence_difference": 0.0,
            "fingerprints_match": True,
        },
        capability_status=CapabilityStatus.COMPLETED,
    )

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(model_file.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
            "_cached_behavioral_report": mock_report,
        },
    )
    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata["reference_model_id"] == "mod_authorized_001"
    assert res.metadata["reference_verified_sha256"] == ref_sha
    assert res.metadata["top1_agreement_rate"] == 1.0
