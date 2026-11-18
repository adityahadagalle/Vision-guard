"""Unit tests for isolated PyTorch checkpoint inspection and SubprocessModelWorker."""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.exceptions import (
    ModelWorkerTimeoutError,
)
from backend.app.modules.model_integrity.inspection.worker import (
    IsolationStatus,
    SubprocessModelWorker,
)


class MockVerifiedAssessor:
    """Mock assessor simulating a verified container isolation sandbox."""

    def assess(self) -> tuple[IsolationStatus, dict[str, Any], str]:
        return (
            IsolationStatus.HARDENED_ISOLATION_VERIFIED,
            {"platform": "Linux", "container": True},
            "Verified isolation sandbox.",
        )


@pytest.fixture
def worker() -> SubprocessModelWorker:
    config = ModelIntegrityConfig(worker_timeout_seconds=10)
    return SubprocessModelWorker(config, isolation_assessor=MockVerifiedAssessor())


def test_inspect_clean_raw_state_dict(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies safe inspection of standard state-dict checkpoint under weights_only=True."""
    ckpt_path = tmp_path / "clean_state_dict.pt"
    state_dict = {
        "conv1.weight": torch.randn(16, 3, 3, 3),
        "conv1.bias": torch.zeros(16),
        "fc.weight": torch.randn(10, 16),
        "fc.bias": torch.zeros(10),
    }
    torch.save(state_dict, ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert report.weights_only_verified is True
    assert report.format == "raw_state_dict"
    assert report.total_tensors == 4
    assert report.total_parameters == (16 * 3 * 3 * 3 + 16 + 10 * 16 + 10)
    assert len(report.state_dict_fingerprint) == 64
    assert report.has_nan is False
    assert report.has_inf is False
    assert len(report.defects) == 0


def test_inspect_nested_checkpoint(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies safe inspection of nested checkpoint dictionary containing model_state_dict."""
    ckpt_path = tmp_path / "nested_ckpt.pth"
    checkpoint = {
        "epoch": 10,
        "best_acc": 94.5,
        "model_state_dict": {
            "layer1.weight": torch.ones(4, 4),
            "layer1.bias": torch.zeros(4),
        },
    }
    torch.save(checkpoint, ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert report.format == "nested_model_state_dict"
    assert report.total_tensors == 2
    assert "layer1.weight" in report.tensors


def test_reject_non_dict_root_checkpoint(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that non-dict top-level checkpoint objects are rejected."""
    ckpt_path = tmp_path / "list_root.pt"
    torch.save([1, 2, 3, 4], ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is False
    assert "dict" in (report.error_message or "").lower()


def test_reject_malicious_pickle_exploit(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that pickle __reduce__ command execution payloads are safely blocked by weights_only=True."""
    ckpt_path = tmp_path / "malicious_payload.pt"

    class ExploitCommand:
        def __reduce__(self):
            import os

            return (os.system, ("echo EXPLOIT_EXECUTED_SHOULD_BE_BLOCKED",))

    # Serialize malicious payload with standard pickle
    with open(ckpt_path, "wb") as f:
        pickle.dump(ExploitCommand(), f)

    report = worker.inspect(ckpt_path)

    # Must fail closed: weights_only does not instantiate arbitrary classes/functions
    assert report.success is False
    assert report.error_message is not None
    assert any(
        err_term in report.error_message.lower()
        for err_term in [
            "weights only",
            "unpickler",
            "unsupported",
            "weightsunpickler",
            "failed to load",
        ]
    )


def test_inspect_corrupted_file(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that truncated or damaged checkpoint files fail closed safely."""
    ckpt_path = tmp_path / "corrupted.pt"
    ckpt_path.write_bytes(
        b"PK\x03\x04random garbage corrupted data that is not a valid zip or pickle"
    )

    report = worker.inspect(ckpt_path)
    assert report.success is False
    assert report.error_message is not None


def test_worker_timeout_enforcement(tmp_path: Path) -> None:
    """Verifies that worker subprocess exceeding timeout is killed and raises ModelWorkerTimeoutError."""
    # Set an impossible timeout of 0.0001 seconds
    tight_config = ModelIntegrityConfig(worker_timeout_seconds=0.0001)
    timeout_worker = SubprocessModelWorker(tight_config, isolation_assessor=MockVerifiedAssessor())

    ckpt_path = tmp_path / "test.pt"
    torch.save({"w": torch.randn(10, 10)}, ckpt_path)

    with pytest.raises(ModelWorkerTimeoutError) as exc_info:
        timeout_worker.inspect(ckpt_path)
    assert "timeout" in exc_info.value.message.lower()


def test_hardened_isolation_gate_on_unsupported_host(tmp_path: Path) -> None:
    """Verifies capability gating returns UNAVAILABLE when running without proven isolation."""
    # Unmocked default worker assesses the real host (Windows/unconfined Linux)
    default_worker = SubprocessModelWorker(ModelIntegrityConfig())
    ckpt_path = tmp_path / "test.pt"
    torch.save({"w": torch.randn(2, 2)}, ckpt_path)

    report = default_worker.inspect(ckpt_path)
    assert report.capability_status == CapabilityStatus.UNAVAILABLE
    assert "isolation" in (report.error_message or "").lower()
