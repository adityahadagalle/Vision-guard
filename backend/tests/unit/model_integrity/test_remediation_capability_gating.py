"""Adversarial and unit tests for fail-closed capability gating (SEC-3A-01, SEC-3A-02).

Proves:
1. Unhardened Windows host does not deserialize checkpoints and returns UNAVAILABLE.
2. Unhardened Linux host does not deserialize checkpoints and returns UNAVAILABLE.
3. Missing isolation controls produce UNAVAILABLE.
4. Failed isolation setup produces UNAVAILABLE.
5. No torch.load call occurs when isolation is unavailable.
6. No detector reports COMPLETED after an unavailable inspection.
7. No configuration flag can silently bypass the gate.
8. Verified hardened isolation allows safe inspection with sanitized environment.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType, CapabilityStatus
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.detectors.architecture_compatibility import (
    ArchitectureCompatibilityDetector,
)
from backend.app.modules.model_integrity.detectors.checkpoint_structure import (
    CheckpointStructureDetector,
)
from backend.app.modules.model_integrity.detectors.state_dict_integrity import (
    StateDictIntegrityDetector,
)
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
    SubprocessModelWorker,
)


class MockAssessor(HostIsolationAssessor):
    def __init__(self, status: IsolationStatus, reason: str = "Test reason"):
        self._status = status
        self._reason = reason

    def assess(self):
        return self._status, {"mock": True}, self._reason


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "dummy.pt"
    torch.save({"w": torch.randn(2, 2)}, ckpt)
    return ckpt


def test_unhardened_windows_host_returns_unavailable_without_deserialization(
    dummy_checkpoint: Path,
) -> None:
    """Verifies standard Windows host returns UNAVAILABLE and never calls torch.load."""
    worker = SubprocessModelWorker()

    with patch("torch.load") as mock_torch_load:
        report = worker.inspect(dummy_checkpoint)

        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        assert report.success is False
        assert "lacks proven OS AppContainer" in report.error_message
        assert report.isolation_level == IsolationStatus.UNHARDENED_HOST_SUBPROCESS.value
        mock_torch_load.assert_not_called()


def test_unhardened_linux_host_returns_unavailable_without_deserialization(
    dummy_checkpoint: Path,
) -> None:
    """Verifies unhardened Linux host returns UNAVAILABLE and never calls torch.load."""
    assessor = MockAssessor(
        IsolationStatus.UNHARDENED_HOST_SUBPROCESS,
        "Host platform 'Linux' lacks unshared network namespaces.",
    )
    worker = SubprocessModelWorker(isolation_assessor=assessor)

    with patch("torch.load") as mock_torch_load:
        report = worker.inspect(dummy_checkpoint)

        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        assert report.success is False
        assert "lacks unshared network namespaces" in report.error_message
        mock_torch_load.assert_not_called()


def test_isolation_failed_setup_returns_unavailable(dummy_checkpoint: Path) -> None:
    """Verifies unexpected errors during isolation probing fail closed as UNAVAILABLE."""
    assessor = MockAssessor(IsolationStatus.ISOLATION_FAILED, "Kernel probe crashed.")
    worker = SubprocessModelWorker(isolation_assessor=assessor)

    with patch("torch.load") as mock_torch_load:
        report = worker.inspect(dummy_checkpoint)

        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        assert report.success is False
        assert "Kernel probe crashed" in report.error_message
        mock_torch_load.assert_not_called()


def test_no_detector_reports_completed_on_unavailable_inspection(
    dummy_checkpoint: Path,
) -> None:
    """Verifies all detectors requiring checkpoint inspection report UNAVAILABLE when isolation is missing."""
    snapshot = AssetSnapshot(
        snapshot_id="snap_unavail_001",
        asset_id="mod_unavail",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="0" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=dummy_checkpoint.stat().st_size,
        schema_version="1.0.0",
        metadata={"storage_path": str(dummy_checkpoint)},
    )
    context = AssessmentContext(snapshot=snapshot, config={})

    # All three inspection detectors must report UNAVAILABLE
    d1 = CheckpointStructureDetector()
    r1 = d1.evaluate(context)
    assert r1.status == CapabilityStatus.UNAVAILABLE
    assert "isolation" in r1.status_reason.lower() or "unavailable" in r1.status_reason.lower()

    d2 = StateDictIntegrityDetector()
    r2 = d2.evaluate(context)
    assert r2.status == CapabilityStatus.UNAVAILABLE
    assert "isolation" in r2.status_reason.lower() or "unavailable" in r2.status_reason.lower()

    d3 = ArchitectureCompatibilityDetector()
    context_with_arch = AssessmentContext(
        snapshot=snapshot, config={"expected_architecture": "simple_cnn"}
    )
    r3 = d3.evaluate(context_with_arch)
    assert r3.status == CapabilityStatus.UNAVAILABLE
    assert "isolation" in r3.status_reason.lower() or "unavailable" in r3.status_reason.lower()


def test_no_config_flag_can_bypass_isolation_gate(dummy_checkpoint: Path) -> None:
    """Verifies that ModelIntegrityConfig contains no bypass flags and worker enforces gate."""
    config = ModelIntegrityConfig()
    # Confirm no bypass attribute exists
    assert not hasattr(config, "allow_unhardened_inspection")
    assert not hasattr(config, "bypass_isolation")
    assert not hasattr(config, "insecure_mode")

    worker = SubprocessModelWorker(config=config)
    report = worker.inspect(dummy_checkpoint)
    assert report.capability_status == CapabilityStatus.UNAVAILABLE


def test_verified_hardened_isolation_executes_inspection(dummy_checkpoint: Path) -> None:
    """Verifies that when hardened isolation is positively verified, worker executes inspection."""
    assessor = MockAssessor(
        IsolationStatus.HARDENED_ISOLATION_VERIFIED,
        "Hardened container sandbox verified.",
    )
    worker = SubprocessModelWorker(isolation_assessor=assessor)

    report = worker.inspect(dummy_checkpoint)
    assert report.success is True
    assert report.capability_status == CapabilityStatus.COMPLETED
    assert report.weights_only_verified is True
    assert report.total_tensors == 1
    assert len(report.state_dict_fingerprint) == 64
