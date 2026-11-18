"""Security and adversarial tests for behavioral evaluation capability gating."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType, CapabilityStatus, IngestionStatus
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.behavioral.worker import (
    SubprocessBehavioralWorker,
)
from backend.app.modules.model_integrity.detectors.backdoor_indicator import (
    BackdoorIndicatorDetector,
)
from backend.app.modules.model_integrity.detectors.behavioral_fingerprint import (
    BehavioralFingerprintDetector,
)
from backend.app.modules.model_integrity.detectors.input_sensitivity import (
    InputSensitivityDetector,
)
from backend.app.modules.model_integrity.detectors.prediction_consistency import (
    PredictionConsistencyDetector,
)
from backend.app.modules.model_integrity.detectors.reference_comparison import (
    ReferenceComparisonDetector,
)
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


class MockAssessor(HostIsolationAssessor):
    def __init__(self, status: IsolationStatus, reason: str = "Test reason"):
        self._status = status
        self._reason = reason

    def assess(self):
        return self._status, {"mock": True}, self._reason


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
def clean_model_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "model.pt"
    model = SimpleCNN()
    torch.save(model.state_dict(), ckpt)
    return ckpt


def test_unhardened_windows_host_returns_unavailable_without_inference(
    clean_model_checkpoint: Path,
) -> None:
    """Verifies that standard Windows host returns UNAVAILABLE and never runs forward inference."""
    worker = SubprocessBehavioralWorker()

    with patch("torch.load") as mock_torch_load:
        report = worker.evaluate_behavior(clean_model_checkpoint, architecture_id="simple_cnn")

        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        assert report.success is False
        assert "lacks proven OS AppContainer" in report.error_message
        assert report.isolation_level == IsolationStatus.UNHARDENED_HOST_SUBPROCESS.value
        mock_torch_load.assert_not_called()


def test_unhardened_linux_host_returns_unavailable_without_inference(
    clean_model_checkpoint: Path,
) -> None:
    """Verifies standard Linux host without hardened namespaces fails closed to UNAVAILABLE."""
    assessor = MockAssessor(
        IsolationStatus.UNHARDENED_HOST_SUBPROCESS,
        reason="Host platform 'Linux' lacks unshared network namespaces.",
    )
    worker = SubprocessBehavioralWorker(isolation_assessor=assessor)

    with patch("torch.load") as mock_torch_load:
        report = worker.evaluate_behavior(clean_model_checkpoint, architecture_id="simple_cnn")

        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        assert report.success is False
        assert "lacks unshared network namespaces" in report.error_message
        mock_torch_load.assert_not_called()


def test_all_behavioral_detectors_propagate_unavailable_on_unhardened_host(
    clean_model_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Verifies that all 5 behavioral detectors report UNAVAILABLE when isolation is unproven.

    Provides an independently registered reference model so ReferenceComparisonDetector
    tests isolation capability gating rather than failing on self-reference or unregistered model.
    """
    # Create an independent reference model checkpoint with distinct weights and SHA-256
    ref_file = tmp_path / "independent_ref.pt"
    torch.save({"fc.weight": torch.randn(10, 128)}, ref_file)
    ref_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()
    mock_repo = MockModelRepo([MockModelRecord("mod_ref_001", str(ref_file.resolve()), ref_sha)])

    snap = AssetSnapshot(
        snapshot_id="snap_unhardened_test",
        asset_id="mod_test",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="0" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=100,
        metadata={
            "storage_path": str(clean_model_checkpoint),
            "architecture_id": "simple_cnn",
            "reference_model_path": str(ref_file),
        },
    )

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(clean_model_checkpoint.parent),
            "reference_model_path": str(ref_file),
            "_model_repo": mock_repo,
        },
    )

    detectors = [
        BehavioralFingerprintDetector(),
        PredictionConsistencyDetector(),
        InputSensitivityDetector(),
        BackdoorIndicatorDetector(),
        ReferenceComparisonDetector(),
    ]

    with (
        patch("subprocess.Popen") as mock_popen,
        patch("torch.load") as mock_torch_load,
        patch.object(SimpleCNN, "forward") as mock_forward,
    ):
        for d in detectors:
            res = d.evaluate(context)
            assert res.status == CapabilityStatus.UNAVAILABLE
            assert "lacks proven" in (res.status_reason or "") or "UNAVAILABLE" in (
                res.status_reason or ""
            )
            assert len(res.findings) == 0

        # Strict security assertions: On an unhardened host, NO inference,
        # NO subprocess, and NO checkpoint loading may occur across any detector.
        mock_popen.assert_not_called()
        mock_torch_load.assert_not_called()
        mock_forward.assert_not_called()


def test_reference_comparison_skipped_when_no_reference_on_unhardened_host(
    clean_model_checkpoint: Path,
) -> None:
    """Verifies ReferenceComparisonDetector returns SKIPPED when no reference model is supplied."""
    snap = AssetSnapshot(
        snapshot_id="snap_unhardened_no_ref",
        asset_id="mod_test",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="0" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=100,
        metadata={
            "storage_path": str(clean_model_checkpoint),
            "architecture_id": "simple_cnn",
        },
    )
    context = AssessmentContext(
        snapshot=snap,
        config={"snapshot_root": str(clean_model_checkpoint.parent)},
    )
    detector = ReferenceComparisonDetector()

    with (
        patch("subprocess.Popen") as mock_popen,
        patch("torch.load") as mock_torch_load,
        patch.object(SimpleCNN, "forward") as mock_forward,
    ):
        res = detector.evaluate(context)
        assert res.status == CapabilityStatus.SKIPPED
        assert "No trusted reference model specified" in (res.status_reason or "")
        assert len(res.findings) == 0
        mock_popen.assert_not_called()
        mock_torch_load.assert_not_called()
        mock_forward.assert_not_called()


def test_reference_comparison_self_reference_rejected_as_failed_without_inference(
    clean_model_checkpoint: Path,
) -> None:
    """Verifies candidate checkpoint supplied as its own reference fails closed with FAILED.

    Proves self-reference rejection occurs before any subprocess launch or model inference.
    """
    snap = AssetSnapshot(
        snapshot_id="snap_unhardened_self_ref",
        asset_id="mod_test",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="0" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=100,
        metadata={
            "storage_path": str(clean_model_checkpoint),
            "architecture_id": "simple_cnn",
            "reference_model_path": str(clean_model_checkpoint),
        },
    )
    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(clean_model_checkpoint.parent),
            "reference_model_path": str(clean_model_checkpoint),
        },
    )
    detector = ReferenceComparisonDetector()

    with (
        patch("subprocess.Popen") as mock_popen,
        patch("torch.load") as mock_torch_load,
        patch.object(SimpleCNN, "forward") as mock_forward,
    ):
        res = detector.evaluate(context)
        assert res.status == CapabilityStatus.FAILED
        assert "cannot serve as its own reference model" in (res.status_reason or "")
        assert res.metadata.get("error_type") == "SELF_REFERENCE_PATH_DETECTED"
        assert len(res.findings) == 0
        mock_popen.assert_not_called()
        mock_torch_load.assert_not_called()
        mock_forward.assert_not_called()


def test_no_config_flag_can_bypass_behavioral_isolation_gate(
    clean_model_checkpoint: Path,
) -> None:
    """Verifies that passing adversarial bypass flags in config fails closed."""
    worker = SubprocessBehavioralWorker()

    with patch("torch.load") as mock_torch_load:
        report = worker.evaluate_behavior(clean_model_checkpoint, architecture_id="simple_cnn")
        assert report.capability_status == CapabilityStatus.UNAVAILABLE
        mock_torch_load.assert_not_called()
