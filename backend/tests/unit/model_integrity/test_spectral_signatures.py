"""Unit tests for SpectralSignaturesDetector and epistemic capability gating."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import (
    AssetType,
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.detectors.spectral_signatures import (
    SpectralSignaturesDetector,
)


@pytest.fixture
def dummy_snapshot() -> AssetSnapshot:
    """Provides a valid test AssetSnapshot."""
    return AssetSnapshot(
        snapshot_id="snap_test_model_01",
        asset_id="model_candidate_v1",
        asset_type=AssetType.MODEL_CHECKPOINT,
        manifest_sha256="a" * 64,
        created_at=datetime.now(UTC),
        file_count=1,
        total_size_bytes=1024,
    )


@pytest.fixture
def detector() -> SpectralSignaturesDetector:
    return SpectralSignaturesDetector()


def test_detector_identity_and_requirements(detector: SpectralSignaturesDetector) -> None:
    """Verifies detector contract, identification, and prerequisites."""
    assert detector.detector_id == "model_integrity.spectral_signatures"
    assert detector.version == "1.0.0"
    assert detector.requirements.requires_model_checkpoint is True
    assert detector.requirements.requires_embeddings is True


def test_capability_gating_unhardened_isolation_attribute(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies immediate UNAVAILABLE return when context indicates unhardened isolation via attribute."""
    # Context with isolation_status = False mock attribute
    mock_context = MagicMock(spec=AssessmentContext)
    mock_context.snapshot = dummy_snapshot
    mock_context.config = {}
    mock_context.isolation_status = False

    result = detector.evaluate(mock_context)

    assert result.status == CapabilityStatus.UNAVAILABLE
    assert result.status_reason is not None
    assert "Host isolation unhardened" in result.status_reason
    assert len(result.findings) == 0


def test_capability_gating_unhardened_isolation_config(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies immediate UNAVAILABLE return when context.config explicitly marks isolation false."""
    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={"isolation_hardened": False},
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.UNAVAILABLE
    assert result.status_reason is not None
    assert "Host isolation unhardened" in result.status_reason
    assert len(result.findings) == 0


def test_inconclusive_when_no_latents_provided(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies INCONCLUSIVE capability status when latent representations are omitted."""
    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={},
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.INCONCLUSIVE
    assert result.status_reason is not None
    assert "No latent representations provided" in result.status_reason


def test_inconclusive_when_sample_density_is_underpowered(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies INCONCLUSIVE status when any class has fewer than 10 samples (e.g. 5 samples)."""
    np.random.seed(42)
    underpowered_latents = {
        0: np.random.randn(5, 64).tolist(),  # Only 5 samples
    }

    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={"class_latents": underpowered_latents},
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.INCONCLUSIVE
    assert result.status_reason is not None
    assert "Insufficient sample density" in result.status_reason
    assert "5 samples" in result.status_reason
    assert "minimum threshold for spectral SVD projection is 10" in result.status_reason


def test_clean_distribution_reports_completed_with_caveat(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies clean unimodal Gaussian latent distribution yields 0 findings and disclaimer."""
    np.random.seed(42)
    # 2 classes with 100 clean Gaussian samples each
    clean_latents = {
        0: np.random.randn(100, 64),
        1: np.random.randn(100, 64),
    }

    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={"class_latents": clean_latents},
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert len(result.evidence) == 0
    assert result.metadata["flagged_classes"] == []
    # Epistemic honesty: MUST contain the exact absence disclaimer
    assert (
        result.metadata["caveat"]
        == "Absence of detected backdoor indicators does not prove absence of backdoors."
    )


def test_poisoned_distribution_detects_spectral_outlier_cluster(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies poisoned distribution with trigger shift is detected, flagged, and emits SUSPECTED_POISON."""
    np.random.seed(42)
    n_samples = 100
    dim = 64

    # 80 clean samples + 20 poisoned samples with heavy shift along trigger directions
    poisoned_class_samples = np.random.randn(n_samples, dim)
    trigger_perturbation = np.zeros(dim)
    trigger_perturbation[:6] = 5.0
    poisoned_class_samples[:20] += trigger_perturbation

    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={
            "class_latents": {4: poisoned_class_samples},
            "outlier_ratio_threshold": 0.15,
        },
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 1
    assert len(result.evidence) == 1
    assert result.metadata["flagged_classes"] == ["4"]

    finding = result.findings[0]
    assert finding.category == FindingCategory.SUSPECTED_POISON
    assert finding.suggested_disposition == SuggestedDisposition.QUARANTINE
    assert finding.anomaly_score is not None
    assert finding.anomaly_score >= 0.15
    assert finding.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
    assert "Suspected Backdoor Poisoning: Spectral Outlier Cluster in Class '4'" in finding.title
    assert "Tran et al., 2018" in finding.description
    assert (
        "Absence of detected spectral signatures does not prove absence of backdoors."
        in finding.description
    )

    evidence = result.evidence[0]
    assert evidence.evidence_type == EvidenceType.STATISTICAL_TEST
    assert evidence.signal_name == "spectral_signature_svd_outlier"
    assert evidence.observed_value == finding.anomaly_score
    assert evidence.threshold_value == 0.15
    assert evidence.metadata["class_id"] == "4"
    assert evidence.metadata["max_outlier_multiplier"] > 5.0
    assert evidence.metadata["top_singular_value"] > 0.0


def test_multiclass_selective_poisoning_isolation(
    detector: SpectralSignaturesDetector,
    dummy_snapshot: AssetSnapshot,
) -> None:
    """Verifies that in a multi-class setup, only the poisoned class is flagged while clean class passes."""
    np.random.seed(1337)
    # Class 0: 100 clean samples
    class_0_clean = np.random.randn(100, 64)

    # Class 1: 75 clean + 25 poisoned samples
    class_1_poisoned = np.random.randn(100, 64)
    trigger = np.zeros(64)
    trigger[:8] = 4.5
    class_1_poisoned[:25] += trigger

    context = AssessmentContext(
        snapshot=dummy_snapshot,
        config={
            "class_latents": {
                "class_0": class_0_clean,
                "class_1": class_1_poisoned,
            },
            "outlier_ratio_threshold": 0.15,
        },
    )

    result = detector.evaluate(context)

    assert result.status == CapabilityStatus.COMPLETED
    assert result.metadata["flagged_classes"] == ["class_1"]
    assert len(result.findings) == 1
    assert result.findings[0].details["class_id"] == "class_1"
    assert "class_0" not in result.metadata["flagged_classes"]
