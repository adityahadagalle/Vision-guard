"""Unit tests for state-dict mathematical integrity and non-finite weight detection."""

from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.modules.model_integrity.config import ModelIntegrityConfig
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
    config = ModelIntegrityConfig(worker_timeout_seconds=60)
    return SubprocessModelWorker(config, isolation_assessor=MockVerifiedAssessor())


def test_detect_nan_weights(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that tensors containing NaN values are identified and quantified."""
    ckpt_path = tmp_path / "nan_weights.pt"
    t = torch.randn(10, 10)
    t[2, 3] = float("nan")
    t[5, 5] = float("nan")
    torch.save({"corrupted_layer.weight": t}, ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert report.has_nan is True
    assert report.total_nan == 2
    assert "corrupted_layer.weight" in report.affected_non_finite_tensors


def test_detect_inf_weights(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that tensors containing +Inf and -Inf values are identified."""
    ckpt_path = tmp_path / "inf_weights.pt"
    t = torch.randn(8, 8)
    t[0, 0] = float("inf")
    t[1, 1] = float("-inf")
    torch.save({"exploding_layer.weight": t}, ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert report.has_inf is True
    assert report.total_inf == 2
    assert "exploding_layer.weight" in report.affected_non_finite_tensors


def test_all_finite_weights_pass(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that completely finite tensors produce clean reports with 0 NaNs/Infs."""
    ckpt_path = tmp_path / "finite_weights.pt"
    state_dict = {
        "l1.weight": torch.randn(4, 4),
        "l1.bias": torch.zeros(4),
        "l2.weight": torch.ones(2, 4),
    }
    torch.save(state_dict, ckpt_path)

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert report.has_nan is False
    assert report.has_inf is False
    assert report.total_nan == 0
    assert report.total_inf == 0
    assert len(report.affected_non_finite_tensors) == 0


def test_state_dict_fingerprint_determinism(worker: SubprocessModelWorker, tmp_path: Path) -> None:
    """Verifies that state_dict_fingerprint is 100% deterministic and sensitive to single weight changes."""
    ckpt1 = tmp_path / "model1.pt"
    ckpt2 = tmp_path / "model2.pt"
    ckpt3_modified = tmp_path / "model3_mod.pt"

    torch.manual_seed(42)
    state_dict_a = {
        "fc1.weight": torch.randn(10, 5),
        "fc1.bias": torch.randn(10),
    }

    # Clone exactly for model 2
    state_dict_b = {k: v.clone() for k, v in state_dict_a.items()}

    # Slightly modify one weight for model 3
    state_dict_c = {k: v.clone() for k, v in state_dict_a.items()}
    state_dict_c["fc1.weight"][0, 0] += 1e-4

    torch.save(state_dict_a, ckpt1)
    torch.save(state_dict_b, ckpt2)
    torch.save(state_dict_c, ckpt3_modified)

    rep1 = worker.inspect(ckpt1)
    rep2 = worker.inspect(ckpt2)
    rep3 = worker.inspect(ckpt3_modified)

    # Identical models must produce identical fingerprints
    assert rep1.state_dict_fingerprint == rep2.state_dict_fingerprint

    # Modified model must produce different fingerprint
    assert rep1.state_dict_fingerprint != rep3.state_dict_fingerprint


def test_parameter_count_limit_defect(tmp_path: Path) -> None:
    """Verifies that exceeding max_parameter_count flags a defect in the report."""
    tight_config = ModelIntegrityConfig(max_parameter_count=50, worker_timeout_seconds=60)
    worker = SubprocessModelWorker(tight_config, isolation_assessor=MockVerifiedAssessor())

    ckpt_path = tmp_path / "too_many_params.pt"
    torch.save({"big.weight": torch.randn(10, 10)}, ckpt_path)  # 100 params > 50

    report = worker.inspect(ckpt_path)

    assert report.success is True
    assert any("parameter count" in d.lower() for d in report.defects)
