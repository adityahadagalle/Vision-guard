"""Unit tests for EvidenceRecord domain model."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.domain.entities.enums import EvidenceType
from backend.app.domain.entities.evidence import EvidenceRecord

_VALID_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_valid_evidence_record() -> None:
    """Verifies that a valid EvidenceRecord instantiates and preserves values."""
    now = datetime.now(UTC)
    ev = EvidenceRecord(
        evidence_id="ev_101",
        evidence_type=EvidenceType.IMAGE_CORRUPTION,
        finding_id="find_501",
        signal_name="jpeg_eoi_marker",
        rationale="Missing expected 0xFFD9 marker at end of byte stream.",
        created_at=now,
        sample_reference="images/damaged.jpg",
        evidence_digest=_VALID_SHA256,
        observed_value=0.0,
        threshold_value=1.0,
        metadata={"scanned_bytes": 1048},
    )
    assert ev.evidence_id == "ev_101"
    assert ev.evidence_type == EvidenceType.IMAGE_CORRUPTION
    assert ev.finding_id == "find_501"
    assert ev.signal_name == "jpeg_eoi_marker"
    assert ev.sample_reference == "images/damaged.jpg"
    assert ev.evidence_digest == _VALID_SHA256
    assert ev.observed_value == 0.0
    assert ev.threshold_value == 1.0
    assert ev.metadata["scanned_bytes"] == 1048


def test_evidence_invalid_digest() -> None:
    """Verifies rejection of invalid SHA-256 digest on evidence records."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        EvidenceRecord(
            evidence_id="ev_101",
            evidence_type=EvidenceType.IMAGE_CORRUPTION,
            finding_id="find_501",
            signal_name="jpeg_eoi_marker",
            rationale="Test rationale",
            created_at=now,
            evidence_digest="bad_digest",
        )


def test_evidence_invalid_sample_reference() -> None:
    """Verifies rejection of absolute paths or directory traversals in sample references."""
    now = datetime.now(UTC)
    for bad_ref in ["/etc/passwd", "C:\\image.jpg", "../secret.png", "img.jpg:stream"]:
        with pytest.raises(ValidationError):
            EvidenceRecord(
                evidence_id="ev_101",
                evidence_type=EvidenceType.IMAGE_CORRUPTION,
                finding_id="find_501",
                signal_name="signal",
                rationale="Test rationale",
                created_at=now,
                sample_reference=bad_ref,
            )


def test_evidence_nan_and_inf_rejection() -> None:
    """Verifies rejection of NaN and infinity in numerical signals."""
    now = datetime.now(UTC)
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValidationError, match="must be a finite numerical value"):
            EvidenceRecord(
                evidence_id="ev_101",
                evidence_type=EvidenceType.METRIC_DISTRIBUTION,
                finding_id="find_501",
                signal_name="distance",
                rationale="Test rationale",
                created_at=now,
                observed_value=bad_val,
            )


def test_evidence_naive_timestamp_rejection() -> None:
    """Verifies rejection of naive datetime on evidence records."""
    naive_dt = datetime.now()  # noqa: DTZ005
    with pytest.raises(ValidationError, match="Datetime must be timezone-aware"):
        EvidenceRecord(
            evidence_id="ev_101",
            evidence_type=EvidenceType.METRIC_DISTRIBUTION,
            finding_id="find_501",
            signal_name="distance",
            rationale="Test rationale",
            created_at=naive_dt,
        )


def test_evidence_immutability() -> None:
    """Verifies that EvidenceRecord is frozen and immutable."""
    now = datetime.now(UTC)
    ev = EvidenceRecord(
        evidence_id="ev_101",
        evidence_type=EvidenceType.IMAGE_CORRUPTION,
        finding_id="find_501",
        signal_name="signal",
        rationale="Test rationale",
        created_at=now,
    )
    with pytest.raises(ValidationError):
        ev.signal_name = "new_signal"  # type: ignore[misc]
