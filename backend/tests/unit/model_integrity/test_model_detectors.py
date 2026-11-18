"""Unit tests for the 4 initial Model Integrity Detectors."""

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
    SeverityLevel,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.detectors import (
    ArchitectureCompatibilityDetector,
    CheckpointStructureDetector,
    ModelArtifactHashDetector,
    StateDictIntegrityDetector,
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
    snap_dir = tmp_path / "snapshots" / "snap_clean"
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
        snapshot_id="snap_clean_model_123456",
        asset_id="mod_test_clean",
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
        },
    )
    return snapshot, model_file


def test_detector_registry_registration() -> None:
    """Verifies that all 4 model detectors register cleanly without conflict."""
    reg = DetectorRegistry()
    register_model_integrity_detectors(reg)

    expected_ids = [
        "model_integrity.artifact_hash",
        "model_integrity.checkpoint_structure",
        "model_integrity.state_dict_integrity",
        "model_integrity.architecture_compatibility",
    ]
    for eid in expected_ids:
        assert reg.has(eid)
        d = reg.get(eid)
        assert d.version == "1.0.0"
        assert d.requirements.requires_model_checkpoint is True


def test_artifact_hash_detector_match(clean_model_snapshot: tuple[AssetSnapshot, Path]) -> None:
    """Verifies ModelArtifactHashDetector passes when observed matches expected."""
    snapshot, model_file = clean_model_snapshot
    detector = ModelArtifactHashDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata.get("matched") is True


def test_artifact_hash_detector_mismatch(clean_model_snapshot: tuple[AssetSnapshot, Path]) -> None:
    """Verifies ModelArtifactHashDetector emits CRITICAL finding when hash mismatches."""
    snapshot, model_file = clean_model_snapshot
    detector = ModelArtifactHashDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_sha256": "f" * 64,  # Intentionally mismatched
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    assert fnd.severity == SeverityLevel.CRITICAL
    assert len(res.evidence) == 1


def test_artifact_hash_detector_skipped_when_no_expected(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ModelArtifactHashDetector reports SKIPPED when expected hash is omitted."""
    snapshot, model_file = clean_model_snapshot
    detector = ModelArtifactHashDetector()
    # Remove expected_sha256
    modified_meta = dict(snapshot.metadata)
    del modified_meta["expected_sha256"]
    snap_no_expected = snapshot.model_copy(update={"metadata": modified_meta})

    context = AssessmentContext(
        snapshot=snap_no_expected,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.SKIPPED
    assert "No expected SHA-256" in (res.status_reason or "")


def test_checkpoint_structure_detector_clean(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies CheckpointStructureDetector passes on authentic state-dict."""
    snapshot, model_file = clean_model_snapshot
    detector = CheckpointStructureDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata.get("weights_only_verified") is True


def test_checkpoint_structure_detector_corrupt(tmp_path: Path) -> None:
    """Verifies CheckpointStructureDetector emits CRITICAL finding on corrupted file."""
    snap_dir = tmp_path / "corrupt_snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    model_file = snap_dir / "model.pt"
    model_file.write_bytes(b"garbage not a checkpoint")

    snapshot = AssetSnapshot(
        snapshot_id="snap_corrupt_12345678",
        asset_id="mod_corrupt",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="c" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=len(model_file.read_bytes()),
        metadata={"storage_path": str(model_file), "snapshot_path": str(snap_dir)},
    )
    detector = CheckpointStructureDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    assert res.findings[0].category == FindingCategory.STRUCTURAL_DEFECT
    assert res.findings[0].severity == SeverityLevel.CRITICAL


def test_state_dict_integrity_detector_clean(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies StateDictIntegrityDetector passes on finite weights."""
    snapshot, model_file = clean_model_snapshot
    detector = StateDictIntegrityDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata.get("finite_weights_verified") is True


def test_state_dict_integrity_detector_nan_weights(tmp_path: Path) -> None:
    """Verifies StateDictIntegrityDetector emits HIGH finding when NaN weights are present."""
    snap_dir = tmp_path / "nan_snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    model_file = snap_dir / "model.pt"

    t = torch.randn(4, 4)
    t[0, 0] = float("nan")
    torch.save({"layer.weight": t}, model_file)

    snapshot = AssetSnapshot(
        snapshot_id="snap_nan_123456789012",
        asset_id="mod_nan",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="d" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=model_file.stat().st_size,
        metadata={"storage_path": str(model_file), "snapshot_path": str(snap_dir)},
    )
    detector = StateDictIntegrityDetector()
    context = AssessmentContext(snapshot=snapshot, config={"snapshot_root": str(snap_dir)})

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 1
    fnd = res.findings[0]
    assert fnd.severity == SeverityLevel.HIGH
    assert "Non-Finite" in fnd.title


def test_architecture_compatibility_detector_match(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ArchitectureCompatibilityDetector passes when candidate matches SimpleCNN."""
    snapshot, model_file = clean_model_snapshot
    detector = ArchitectureCompatibilityDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_architecture": "simple_cnn",
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.COMPLETED
    assert len(res.findings) == 0
    assert res.metadata.get("compatible") is True


def test_architecture_compatibility_detector_skipped_when_no_arch(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ArchitectureCompatibilityDetector reports SKIPPED when expected_architecture is omitted."""
    snapshot, model_file = clean_model_snapshot
    detector = ArchitectureCompatibilityDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={"snapshot_root": str(model_file.parent)},
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.SKIPPED
    assert "No expected architecture" in (res.status_reason or "")


def test_architecture_compatibility_detector_unknown_arch(
    clean_model_snapshot: tuple[AssetSnapshot, Path],
) -> None:
    """Verifies ArchitectureCompatibilityDetector reports FAILED when unknown architecture is given."""
    snapshot, model_file = clean_model_snapshot
    detector = ArchitectureCompatibilityDetector()
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_architecture": "custom_transformer_gpt99",
        },
    )

    res = detector.evaluate(context)
    assert res.status == CapabilityStatus.FAILED
    assert "Unknown architecture" in (res.status_reason or "")


def test_model_detectors_propagate_unavailable_on_unhardened_host(
    clean_model_snapshot: tuple[AssetSnapshot, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly verifies all 3 inspection detectors return UNAVAILABLE when host is unhardened."""
    monkeypatch.setattr(
        HostIsolationAssessor,
        "assess",
        lambda self: (
            IsolationStatus.UNHARDENED_HOST_SUBPROCESS,
            {"platform": "Windows"},
            "Unhardened host without container isolation.",
        ),
    )
    snapshot, model_file = clean_model_snapshot
    context = AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(model_file.parent),
            "expected_architecture": "simple_cnn",
        },
    )

    assert CheckpointStructureDetector().evaluate(context).status == CapabilityStatus.UNAVAILABLE
    assert StateDictIntegrityDetector().evaluate(context).status == CapabilityStatus.UNAVAILABLE
    assert (
        ArchitectureCompatibilityDetector().evaluate(context).status == CapabilityStatus.UNAVAILABLE
    )
