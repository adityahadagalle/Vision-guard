"""Unit tests for covariate shift detectors and domain entities (Phase 6 Task 6.1).

Verifies:
1. Identical distributions yield is_drifting = False.
2. Distinct distributions (mean shift, variance shift, different families) yield is_drifting = True.
3. Sample density gate (N < 30) returns CapabilityStatus.INCONCLUSIVE without exception.
4. Non-finite value filtering (NaN, Inf) and density threshold interaction.
5. Domain schemas (DriftMetric, DriftReport) enforce extra="forbid" and proper serialization.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.domain.entities.drift import DriftMetric, DriftReport
from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.distribution_shift.detectors.ks_drift import (
    KolmogorovSmirnovDriftDetector,
    detect_ks_drift,
)
from backend.app.modules.distribution_shift.detectors.wasserstein_drift import (
    WassersteinDistanceDriftDetector,
    detect_wasserstein_drift,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible test distributions."""
    return np.random.default_rng(seed=42)


# =============================================================================
# 1. Kolmogorov-Smirnov (KS) Drift Detector Tests
# =============================================================================


def test_ks_identical_distributions_no_drift(rng: np.random.Generator) -> None:
    """Verifies that samples from identical distributions do not trigger KS drift."""
    ref = rng.normal(loc=0.0, scale=1.0, size=200)
    prod = rng.normal(loc=0.0, scale=1.0, size=200)

    metric = KolmogorovSmirnovDriftDetector.evaluate(ref, prod, alpha=0.05)

    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.is_drifting is False
    assert metric.drift_score is not None
    assert metric.p_value is not None
    assert metric.p_value > 0.05
    assert metric.reference_sample_count == 200
    assert metric.production_sample_count == 200
    assert metric.status_reason is None


def test_ks_shifted_distribution_triggers_drift(rng: np.random.Generator) -> None:
    """Verifies that a pronounced distribution shift (mean offset) triggers KS drift."""
    ref = rng.normal(loc=0.0, scale=1.0, size=150)
    # Production distribution shifted by +1.5 standard deviations
    prod = rng.normal(loc=1.5, scale=1.0, size=150)

    metric = detect_ks_drift(ref, prod, alpha=0.05)

    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.is_drifting is True
    assert metric.drift_score is not None
    assert metric.p_value is not None
    assert metric.p_value < 1e-4
    assert metric.reference_sample_count == 150
    assert metric.production_sample_count == 150


def test_ks_different_distribution_families(rng: np.random.Generator) -> None:
    """Verifies that comparing Normal vs Uniform distributions triggers KS drift."""
    ref = rng.normal(loc=0.5, scale=0.15, size=100)
    prod = rng.uniform(low=0.0, high=1.0, size=100)

    metric = KolmogorovSmirnovDriftDetector.evaluate(ref, prod)

    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.is_drifting is True
    assert metric.p_value is not None
    assert metric.p_value < 0.01


def test_ks_sample_density_gate_inconclusive(rng: np.random.Generator) -> None:
    """Verifies ADR-012 fail-closed gate: N < 30 returns INCONCLUSIVE without error."""
    # Reference has 20 samples (< 30)
    ref_small = rng.normal(size=20)
    prod_full = rng.normal(size=100)

    metric = KolmogorovSmirnovDriftDetector.evaluate(ref_small, prod_full, min_samples=30)

    assert metric.capability_status == CapabilityStatus.INCONCLUSIVE
    assert metric.is_drifting is False
    assert metric.drift_score is None
    assert metric.p_value is None
    assert metric.reference_sample_count == 20
    assert metric.production_sample_count == 100
    assert "Insufficient sample density" in str(metric.status_reason)


def test_ks_production_window_small_inconclusive(rng: np.random.Generator) -> None:
    """Verifies ADR-012 gate when production window has < 30 samples."""
    ref_full = rng.normal(size=100)
    prod_small = rng.normal(size=15)

    metric = detect_ks_drift(ref_full, prod_small, min_samples=30)

    assert metric.capability_status == CapabilityStatus.INCONCLUSIVE
    assert metric.is_drifting is False
    assert metric.drift_score is None
    assert metric.p_value is None
    assert "production has 15 finite samples" in str(metric.status_reason)


def test_ks_nan_filtering_and_density(rng: np.random.Generator) -> None:
    """Verifies that non-finite values are filtered out before sample density check."""
    ref = rng.normal(size=40).tolist()
    # Add NaNs and Infs to production
    prod = rng.normal(size=35).tolist() + [float("nan"), float("inf"), float("-inf")]

    # 35 finite samples >= 30 -> should complete
    metric = KolmogorovSmirnovDriftDetector.evaluate(ref, prod, min_samples=30)
    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.production_sample_count == 35

    # If NaNs reduce finite count below threshold:
    prod_sparse = [1.0] * 25 + [float("nan")] * 20
    metric_inconclusive = KolmogorovSmirnovDriftDetector.evaluate(ref, prod_sparse, min_samples=30)
    assert metric_inconclusive.capability_status == CapabilityStatus.INCONCLUSIVE
    assert metric_inconclusive.production_sample_count == 25


# =============================================================================
# 2. Wasserstein Distance Drift Detector Tests
# =============================================================================


def test_wasserstein_identical_distributions_no_drift(rng: np.random.Generator) -> None:
    """Verifies that samples from identical distributions have low Wasserstein distance."""
    ref = rng.normal(loc=0.0, scale=1.0, size=300)
    prod = rng.normal(loc=0.0, scale=1.0, size=300)

    metric = WassersteinDistanceDriftDetector.evaluate(ref, prod, threshold=0.15)

    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.is_drifting is False
    assert metric.drift_score is not None
    assert metric.drift_score < 0.15
    assert metric.p_value is None  # Pure distance metric
    assert metric.reference_sample_count == 300
    assert metric.production_sample_count == 300


def test_wasserstein_shifted_distribution_triggers_drift(
    rng: np.random.Generator,
) -> None:
    """Verifies that an offset distribution triggers Wasserstein drift."""
    ref = rng.normal(loc=0.0, scale=1.0, size=100)
    # Shifted by 0.5 units
    prod = rng.normal(loc=0.5, scale=1.0, size=100)

    metric = detect_wasserstein_drift(ref, prod, threshold=0.10)

    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.is_drifting is True
    assert metric.drift_score is not None
    assert metric.drift_score > 0.10
    assert metric.p_value is None


def test_wasserstein_sample_density_gate_inconclusive(rng: np.random.Generator) -> None:
    """Verifies ADR-012 fail-closed gate: N < 30 returns INCONCLUSIVE for Wasserstein."""
    ref_small = rng.normal(size=12)
    prod_full = rng.normal(size=50)

    metric = WassersteinDistanceDriftDetector.evaluate(ref_small, prod_full, min_samples=30)

    assert metric.capability_status == CapabilityStatus.INCONCLUSIVE
    assert metric.is_drifting is False
    assert metric.drift_score is None
    assert metric.p_value is None
    assert "Insufficient sample density" in str(metric.status_reason)


def test_wasserstein_nan_filtering(rng: np.random.Generator) -> None:
    """Verifies that non-finite values are filtered prior to EMD calculation."""
    ref = [0.0] * 50
    prod = [0.2] * 40 + [float("nan"), float("inf")]

    metric = detect_wasserstein_drift(ref, prod, threshold=0.10, min_samples=30)
    assert metric.capability_status == CapabilityStatus.COMPLETED
    assert metric.production_sample_count == 40
    assert metric.drift_score == pytest.approx(0.2, abs=1e-3)
    assert metric.is_drifting is True


# =============================================================================
# 3. Domain Entity Schema Tests (drift.py)
# =============================================================================


def test_drift_metric_extra_forbid() -> None:
    """Confirms that DriftMetric strictly forbids unknown attributes."""
    with pytest.raises(ValidationError):
        DriftMetric(
            metric_name="test_metric",
            drift_score=0.25,
            threshold=0.10,
            is_drifting=True,
            unauthorized_field="malicious_payload",  # type: ignore[call-arg]
        )


def test_drift_report_aggregation() -> None:
    """Verifies constructing and aggregating a DriftReport with multiple metrics."""
    m1 = DriftMetric(
        metric_name="kolmogorov_smirnov",
        drift_score=0.45,
        threshold=0.05,
        p_value=0.001,
        is_drifting=True,
        capability_status=CapabilityStatus.COMPLETED,
        reference_sample_count=100,
        production_sample_count=100,
    )
    m2 = DriftMetric(
        metric_name="wasserstein_distance",
        drift_score=0.35,
        threshold=0.10,
        p_value=None,
        is_drifting=True,
        capability_status=CapabilityStatus.COMPLETED,
        reference_sample_count=100,
        production_sample_count=100,
    )

    report = DriftReport(
        dataset_or_model_id="model_resnet18_v1",
        metrics=[m1, m2],
        is_drifting=True,
        overall_status=CapabilityStatus.COMPLETED,
        reference_sample_count=100,
        production_sample_count=100,
        details={"evaluated_features": "confidence_scores"},
    )

    assert report.is_drifting is True
    assert len(report.metrics) == 2
    assert report.overall_status == CapabilityStatus.COMPLETED
    assert report.report_id.startswith("drf_")


def test_drift_report_extra_forbid() -> None:
    """Confirms that DriftReport strictly forbids unknown attributes."""
    with pytest.raises(ValidationError):
        DriftReport(
            is_drifting=False,
            unexpected_key="invalid",  # type: ignore[call-arg]
        )
