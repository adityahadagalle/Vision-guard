"""Unit tests for UniversalFinding domain model."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.domain.entities.enums import (
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)

_VALID_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _make_valid_finding(**overrides) -> UniversalFinding:
    now = datetime.now(UTC)
    base = {
        "finding_id": "find_1001",
        "assessment_id": "assess_2001",
        "detector_id": "data_integrity.corrupt_image",
        "detector_version": "1.0.0",
        "category": FindingCategory.CORRUPTED_FILE,
        "severity": SeverityLevel.HIGH,
        "title": "Truncated JPEG Stream",
        "description": "Image file ended before encountering expected EOI marker (0xFFD9).",
        "confidence": 0.99,
        "anomaly_score": 0.85,
        "suggested_disposition": SuggestedDisposition.QUARANTINE,
        "status": CapabilityStatus.COMPLETED,
        "affected_assets": [
            AffectedAssetReference(
                asset_id="asset_cifar10",
                sample_path="images/bad_01.jpg",
                sha256=_VALID_SHA256,
            )
        ],
        "evidence_ids": ["ev_3001"],
        "reproducibility": ReproducibilityMetadata(
            execution_timestamp=now,
            seed=42,
            software_versions={"torch": "2.15.0+cpu"},
            detector_config={"strict_eoi": True},
        ),
        "created_at": now,
        "details": {"missing_bytes": 142},
    }
    base.update(overrides)
    return UniversalFinding(**base)


def test_valid_universal_finding() -> None:
    """Verifies that a valid UniversalFinding instantiates with all properties preserved."""
    finding = _make_valid_finding()
    assert finding.finding_id == "find_1001"
    assert finding.detector_id == "data_integrity.corrupt_image"
    assert finding.severity == SeverityLevel.HIGH
    assert finding.confidence == 0.99
    assert finding.anomaly_score == 0.85
    assert finding.category == FindingCategory.CORRUPTED_FILE
    assert len(finding.affected_assets) == 1
    assert finding.affected_assets[0].sample_path == "images/bad_01.jpg"
    assert finding.details["missing_bytes"] == 142


def test_finding_distinct_concepts() -> None:
    """Verifies that anomaly_score, confidence, and severity remain distinct and uncoupled."""
    # Low anomaly score, high confidence, low severity
    f1 = _make_valid_finding(
        anomaly_score=0.12,
        confidence=0.98,
        severity=SeverityLevel.LOW,
    )
    assert f1.anomaly_score == 0.12
    assert f1.confidence == 0.98
    assert f1.severity == SeverityLevel.LOW

    # High anomaly score, low confidence, critical severity
    f2 = _make_valid_finding(
        anomaly_score=0.95,
        confidence=0.35,
        severity=SeverityLevel.CRITICAL,
    )
    assert f2.anomaly_score == 0.95
    assert f2.confidence == 0.35
    assert f2.severity == SeverityLevel.CRITICAL


def test_finding_invalid_confidence_range() -> None:
    """Verifies rejection of confidence outside [0.0, 1.0]."""
    with pytest.raises(ValidationError, match="confidence"):
        _make_valid_finding(confidence=1.05)

    with pytest.raises(ValidationError, match="confidence"):
        _make_valid_finding(confidence=-0.01)


def test_finding_nan_and_inf_rejection() -> None:
    """Verifies that NaN and infinity values are strictly rejected."""
    for bad_num in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValidationError):
            _make_valid_finding(confidence=bad_num)

        with pytest.raises(ValidationError):
            _make_valid_finding(anomaly_score=bad_num)


def test_finding_invalid_severity() -> None:
    """Verifies rejection of invalid severity values."""
    with pytest.raises(ValidationError):
        _make_valid_finding(severity="URGENT_CRITICAL")


def test_finding_missing_required_fields() -> None:
    """Verifies rejection when mandatory fields are omitted."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        UniversalFinding(  # type: ignore[call-arg]
            finding_id="find_1",
            # missing assessment_id, detector_id, category, severity, etc.
            created_at=now,
        )


def test_finding_naive_timestamp_rejection() -> None:
    """Verifies rejection of naive datetime objects."""
    naive_dt = datetime.now()  # noqa: DTZ005
    with pytest.raises(ValidationError, match="Datetime must be timezone-aware"):
        _make_valid_finding(created_at=naive_dt)


def test_finding_serialization_and_deserialization() -> None:
    """Verifies round-trip Pydantic serialization to dict and JSON."""
    finding = _make_valid_finding()
    data_dict = finding.model_dump()
    assert isinstance(data_dict, dict)
    assert data_dict["finding_id"] == "find_1001"

    data_json = finding.model_dump_json()
    parsed = json.loads(data_json)
    assert parsed["detector_id"] == "data_integrity.corrupt_image"
    assert parsed["confidence"] == 0.99

    # Reconstitute from JSON
    reconstituted = UniversalFinding.model_validate_json(data_json)
    assert reconstituted == finding
