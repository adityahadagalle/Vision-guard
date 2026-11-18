"""Unit tests for SubprocessBehavioralWorker and probe CLI execution."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.model_integrity.architecture.registry import (
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.behavioral.worker import (
    IsolationStatus,
    SubprocessBehavioralWorker,
)
from backend.app.modules.model_integrity.exceptions import ModelWorkerTimeoutError


class MockVerifiedAssessor:
    """Mock isolation assessor simulating a verified container sandbox."""

    def assess(self) -> tuple[IsolationStatus, dict[str, Any], str]:
        return (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        )


@pytest.fixture
def behavioral_worker() -> SubprocessBehavioralWorker:
    config = BehavioralConfig(worker_timeout_seconds=30.0, probe_count=10)
    return SubprocessBehavioralWorker(config, isolation_assessor=MockVerifiedAssessor())


def _create_simple_cnn_weights() -> dict[str, torch.Tensor]:
    reg = TrustedArchitectureRegistry()
    model = reg.create_instance("simple_cnn")
    return model.state_dict()


def test_behavioral_worker_successful_evaluation(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies end-to-end execution of probe_cli subprocess on a valid SimpleCNN checkpoint."""
    ckpt_path = tmp_path / "simple_cnn.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    report = behavioral_worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
    )

    assert report.success is True
    assert report.capability_status == CapabilityStatus.COMPLETED
    assert report.architecture_id == "simple_cnn"
    assert report.probes_count == 10
    assert len(report.sample_records) == 5
    assert len(report.behavioral_fingerprint) == 64

    # Consistency
    assert report.consistency is not None
    assert report.consistency["repetitions"] == 3
    assert report.consistency["flip_count"] == 0

    # Sensitivity
    assert report.sensitivity is not None
    assert report.sensitivity["epsilon"] == 0.02
    assert report.sensitivity["flip_count"] >= 0

    # Backdoor triggers
    assert report.backdoor is not None
    assert len(report.backdoor["triggers"]) == 4

    # No reference provided
    assert report.reference_comparison is None


def test_behavioral_worker_with_reference_model(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that providing a reference model executes reference comparison inside the worker."""
    weights = _create_simple_cnn_weights()
    ckpt_path = tmp_path / "model.pt"
    ref_path = tmp_path / "ref.pt"

    torch.save(weights, ckpt_path)
    torch.save(weights, ref_path)

    report = behavioral_worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
        reference_checkpoint_path=ref_path,
    )

    assert report.success is True
    assert report.reference_comparison is not None
    assert report.reference_comparison["top1_agreement_rate"] == 1.0
    assert report.reference_comparison["max_logit_divergence"] < 1e-4
    assert report.reference_comparison["fingerprints_match"] is True


def test_behavioral_worker_timeout_enforcement(tmp_path: Path) -> None:
    """Verifies that exceeding the worker timeout kills the subprocess and raises ModelWorkerTimeoutError."""
    tight_config = BehavioralConfig(worker_timeout_seconds=0.0001, probe_count=10)
    timeout_worker = SubprocessBehavioralWorker(
        tight_config, isolation_assessor=MockVerifiedAssessor()
    )

    ckpt_path = tmp_path / "model.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    with pytest.raises(ModelWorkerTimeoutError) as exc_info:
        timeout_worker.evaluate_behavior(
            checkpoint_path=ckpt_path,
            architecture_id="simple_cnn",
        )
    assert "timeout" in exc_info.value.message.lower()


def test_behavioral_worker_corrupted_checkpoint(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that corrupted checkpoint files fail closed gracefully without crashing host."""
    ckpt_path = tmp_path / "corrupted.pt"
    ckpt_path.write_bytes(b"NOT_A_VALID_PYTORCH_CHECKPOINT_STREAM_OR_ZIP")

    report = behavioral_worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
    )

    assert report.success is False
    assert report.error_message is not None
    assert any(
        term in report.error_message.lower()
        for term in ["invalid", "corrupt", "failed", "unsupported", "pickle", "load"]
    )


def test_behavioral_worker_shape_mismatch(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that state_dict with mismatched tensor dimensions fails closed gracefully."""
    ckpt_path = tmp_path / "mismatch.pt"
    mismatched = {
        "conv1.weight": torch.randn(99, 99, 1, 1),
    }
    torch.save(mismatched, ckpt_path)

    report = behavioral_worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
    )

    assert report.success is False
    assert report.error_message is not None
    assert any(
        term in report.error_message.lower()
        for term in ["mismatch", "missing", "unexpected", "shape", "size", "error"]
    )


def test_behavioral_worker_extracts_and_transports_latent_representations(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that worker subprocess extracts, serializes, and decodes latent representations."""
    ckpt_path = tmp_path / "model_feats.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    report = behavioral_worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
    )

    assert report.success is True
    assert isinstance(report.latent_representations, dict)
    assert len(report.latent_representations) == 10

    for c in range(10):
        c_str = str(c)
        assert c_str in report.latent_representations
        arr = report.latent_representations[c_str]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 64)
        assert np.isfinite(arr).all()


def test_behavioral_worker_scaled_probe_suite_transport(tmp_path: Path) -> None:
    """Verifies that M=100 probe suite yields 10 balanced classes each with 10 samples (10, 64)."""
    config = BehavioralConfig(worker_timeout_seconds=30.0, probe_count=100)
    worker = SubprocessBehavioralWorker(config, isolation_assessor=MockVerifiedAssessor())

    ckpt_path = tmp_path / "model_scaled.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    report = worker.evaluate_behavior(
        checkpoint_path=ckpt_path,
        architecture_id="simple_cnn",
    )

    assert report.success is True
    assert report.probes_count == 100
    assert len(report.latent_representations) == 10

    total_samples = 0
    for c in range(10):
        arr = report.latent_representations[str(c)]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 64)
        total_samples += arr.shape[0]

    assert total_samples == 100


def test_behavioral_worker_malformed_b64_payload_graceful_handling(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that malformed Base64 data in CLI output is skipped safely without crashing worker."""
    ckpt_path = tmp_path / "model.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    malformed_payload = {
        "success": True,
        "architecture_id": "simple_cnn",
        "probes_count": 10,
        "latent_representations": {
            "0": {
                "shape": [1, 64],
                "dtype": "float32",
                "data_b64": "!!!NOT_VALID_BASE64_BYTES!!!",
            }
        },
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = (json.dumps(malformed_payload), "")

    with patch("subprocess.Popen", return_value=mock_proc):
        report = behavioral_worker.evaluate_behavior(
            checkpoint_path=ckpt_path,
            architecture_id="simple_cnn",
        )

    assert report.success is True
    assert "0" not in report.latent_representations


def test_behavioral_worker_oversized_payload_byte_length_mismatch(
    behavioral_worker: SubprocessBehavioralWorker, tmp_path: Path
) -> None:
    """Verifies that byte length mismatch between declared shape and decoded buffer is skipped safely."""
    ckpt_path = tmp_path / "model.pt"
    torch.save(_create_simple_cnn_weights(), ckpt_path)

    import base64

    # 10 bytes encoded, but declared shape demands 1000 * 1000 * 4 = 4,000,000 bytes
    ten_bytes_b64 = base64.b64encode(b"0123456789").decode("ascii")
    mismatch_payload = {
        "success": True,
        "architecture_id": "simple_cnn",
        "probes_count": 10,
        "latent_representations": {
            "0": {
                "shape": [1000, 1000],
                "dtype": "float32",
                "data_b64": ten_bytes_b64,
            }
        },
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = (json.dumps(mismatch_payload), "")

    with patch("subprocess.Popen", return_value=mock_proc):
        report = behavioral_worker.evaluate_behavior(
            checkpoint_path=ckpt_path,
            architecture_id="simple_cnn",
        )

    assert report.success is True
    assert "0" not in report.latent_representations
