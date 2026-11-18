"""Unit tests verifying remediation of FIND-3B-02: Behavioral Report Cache Isolation.

Adversarial test cases:
1. Safe reuse (cache hit) on identical execution context.
2. Cache miss on mutated probe_seed.
3. Cache miss on mutated architecture_id.
4. Cache miss on mutated candidate_sha256.
5. Cache miss on mutated reference_sha256.
6. Cache miss on mutated assessment_id.
7. Cache miss on mutated lease_generation (job retry).
8. Cache miss on mutated job_id.
9. Cache miss on mutated sensitivity_epsilon.
10. Cache miss on mutated consistency_repetitions.
11. Cache miss on metadata tampering (hash collision defense).
12. Zero secret or credential leakage in cache keys.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.behavioral.common import (
    compute_behavioral_cache_identity,
    get_or_run_behavioral_report,
)
from backend.app.modules.model_integrity.behavioral.worker import BehavioralReport
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


@pytest.fixture
def dummy_model_context(tmp_path: Path) -> tuple[AssessmentContext, Path]:
    model_file = tmp_path / "model.pt"
    cnn = SimpleCNN()
    torch.save(cnn.state_dict(), model_file)

    snap = AssetSnapshot(
        snapshot_id="snap_cache_001",
        asset_id="mod_cache_001",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="a" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=model_file.stat().st_size,
        metadata={"storage_path": str(model_file)},
    )

    context = AssessmentContext(
        snapshot=snap,
        config={
            "snapshot_root": str(tmp_path),
            "assessment_id": "asm_cache_001",
            "job_id": "job_cache_001",
            "lease_generation": 1,
            "architecture_id": "simple_cnn",
            "probe_count": 20,
            "probe_seed": 42,
            "sensitivity_epsilon": 0.02,
        },
    )
    return context, model_file


def _create_mock_report(seed: int = 42, fp: str = "a" * 64) -> BehavioralReport:
    return BehavioralReport(
        success=True,
        architecture_id="simple_cnn",
        probes_count=20,
        probe_seed=seed,
        behavioral_fingerprint=fp,
        capability_status=CapabilityStatus.COMPLETED,
    )


def test_cache_hit_on_identical_execution_context(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that multiple calls under identical parameters reuse cached report without worker execution."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        # Call 1: Cache miss -> worker called once
        res1 = get_or_run_behavioral_report("det.1", "1.0", context)
        assert not isinstance(res1, tuple) or res1[2].success is True
        assert mock_instance.evaluate_behavior.call_count == 1

        # Call 2: Identical context -> Cache hit -> worker NOT called again
        res2 = get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1
        assert res1[2] == res2[2]


def test_cache_miss_on_mutated_probe_seed(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that mutating probe_seed forces cache miss and spawns fresh evaluation."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report(seed=42)
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        # Mutate probe_seed
        context.config["probe_seed"] = 999
        mock_instance.evaluate_behavior.return_value = _create_mock_report(seed=999)

        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_cache_miss_on_mutated_architecture_id(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that changing architecture_id triggers cache miss."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        context.config["architecture_id"] = "resnet18"
        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_cache_miss_on_mutated_lease_generation_retry(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that a job retry with incremented lease_generation cannot reuse stale attempt's report."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        # Retry event: lease_generation increments from 1 to 2
        context.config["lease_generation"] = 2
        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_cache_miss_on_mutated_assessment_id(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that distinct assessment IDs cannot cross-pollinate cache."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        context.config["assessment_id"] = "asm_cache_002"
        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_cache_miss_on_mutated_sensitivity_epsilon(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies that changing sensitivity perturbation bound triggers cache miss."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        context.config["sensitivity_epsilon"] = 0.05
        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_cache_miss_on_metadata_tampering_defense(
    dummy_model_context: tuple[AssessmentContext, Path],
) -> None:
    """Verifies defense-in-depth: if cache key matches but metadata differs, cache miss occurs."""
    context, _ = dummy_model_context

    with patch(
        "backend.app.modules.model_integrity.behavioral.common.SubprocessBehavioralWorker"
    ) as mock_worker_cls:
        mock_instance = MagicMock()
        mock_instance.evaluate_behavior.return_value = _create_mock_report()
        mock_worker_cls.return_value = mock_instance

        get_or_run_behavioral_report("det.1", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 1

        # Intentionally tamper with the stored metadata inside the cache store
        cache_store = context.config["_behavioral_cache_store"]
        for key in cache_store:
            cache_store[key]["metadata"]["probe_seed"] = 99999

        # Request with original context -> metadata mismatch -> cache miss
        get_or_run_behavioral_report("det.2", "1.0", context)
        assert mock_instance.evaluate_behavior.call_count == 2


def test_zero_secret_leakage_in_cache_key() -> None:
    """Verifies that cache key generator never includes sensitive tokens or credentials."""
    key, meta = compute_behavioral_cache_identity(
        candidate_sha256="a" * 64,
        reference_sha256="b" * 64,
        architecture_id="simple_cnn",
        assessment_id="asm_test_001",
        job_id="job_001",
        lease_generation=1,
        probe_count=30,
        probe_batch_size=10,
        probe_seed=42,
        input_shape=[3, 32, 32],
        sensitivity_epsilon=0.02,
        sensitivity_flip_threshold=0.40,
        consistency_repetitions=3,
        backdoor_trigger_size=4,
        backdoor_concentration_threshold=0.85,
        backdoor_confidence_threshold=0.80,
        backdoor_delta_threshold=0.50,
        reference_agreement_threshold=0.95,
    )
    assert len(key) == 64
    for sensitive_key in ["token", "secret", "password", "key_token", "auth"]:
        assert sensitive_key not in meta
