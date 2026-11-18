"""Unit tests for SEC-3A-05: Expected state-dict fingerprint verification semantics.

Tests:
1. Trusted fingerprint match -> VERIFIED_MATCH, fingerprint_verified=True, no finding.
2. Trusted fingerprint mismatch -> VERIFIED_MISMATCH, fingerprint_verified=False, CRITICAL finding, QUARANTINE disposition.
3. Malformed expected fingerprint -> CapabilityStatus.FAILED.
4. Omitted expected fingerprint -> SKIPPED_NO_REFERENCE, fingerprint_verified=None, no claim of authenticity.
5. Case-insensitivity and whitespace stripping.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.app.domain.entities.enums import (
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.modules.model_integrity.detectors.state_dict_integrity import (
    StateDictIntegrityDetector,
)
from backend.app.modules.model_integrity.inspection.worker import (
    CheckpointInspectionReport,
)


@pytest.fixture
def mock_clean_report() -> CheckpointInspectionReport:
    """Returns a valid inspection report with a known fingerprint."""
    return CheckpointInspectionReport(
        capability_status=CapabilityStatus.COMPLETED,
        success=True,
        state_dict_fingerprint="a" * 64,
        total_tensors=10,
        total_parameters=1000,
        has_nan=False,
        has_inf=False,
        total_nan=0,
        total_inf=0,
        affected_non_finite_tensors=[],
        isolation_level="HARDENED_ISOLATION_VERIFIED",
    )


@pytest.fixture
def base_context(
    tmp_path: Path, mock_clean_report: CheckpointInspectionReport
) -> AssessmentContext:
    """Constructs a basic AssessmentContext with cached inspection report and dummy model file."""
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"dummy checkpoint data")

    snapshot = MagicMock()
    snapshot.snapshot_id = "snap-fp-test"
    snapshot.asset_id = "asset-fp-test"
    snapshot.metadata = {"storage_path": str(model_file)}

    return AssessmentContext(
        snapshot=snapshot,
        config={
            "snapshot_root": str(tmp_path),
            "_cached_inspection_report": mock_clean_report,
        },
    )


def test_fingerprint_omitted_returns_skipped_no_reference(base_context: AssessmentContext) -> None:
    """When expected_state_dict_fingerprint is omitted, detector completes without claiming verified authenticity."""
    detector = StateDictIntegrityDetector()
    result = detector.evaluate(base_context)

    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert result.metadata["fingerprint_verified"] is None
    assert result.metadata["fingerprint_comparison"] == "SKIPPED_NO_REFERENCE"
    assert result.metadata["state_dict_fingerprint"] == "a" * 64


def test_fingerprint_matching_verified(base_context: AssessmentContext) -> None:
    """When expected_state_dict_fingerprint matches observed, status is COMPLETED and match is confirmed."""
    base_context.config["expected_state_dict_fingerprint"] = "a" * 64

    detector = StateDictIntegrityDetector()
    result = detector.evaluate(base_context)

    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert result.metadata["fingerprint_verified"] is True
    assert result.metadata["fingerprint_comparison"] == "VERIFIED_MATCH"


def test_fingerprint_case_and_whitespace_insensitivity(base_context: AssessmentContext) -> None:
    """Verification normalizes uppercase hex and leading/trailing whitespace."""
    base_context.config["expected_state_dict_fingerprint"] = "   " + ("A" * 64) + " \n"

    detector = StateDictIntegrityDetector()
    result = detector.evaluate(base_context)

    assert result.status == CapabilityStatus.COMPLETED
    assert len(result.findings) == 0
    assert result.metadata["fingerprint_verified"] is True
    assert result.metadata["fingerprint_comparison"] == "VERIFIED_MATCH"


def test_fingerprint_mismatch_raises_critical_finding(base_context: AssessmentContext) -> None:
    """When expected fingerprint differs from observed, emit CRITICAL finding with QUARANTINE disposition."""
    expected_fp = "b" * 64
    base_context.config["expected_state_dict_fingerprint"] = expected_fp

    detector = StateDictIntegrityDetector()
    result = detector.evaluate(base_context)

    assert result.status == CapabilityStatus.COMPLETED
    assert result.metadata["fingerprint_verified"] is False
    assert result.metadata["fingerprint_comparison"] == "VERIFIED_MISMATCH"

    assert len(result.findings) == 1
    fnd = result.findings[0]
    assert fnd.category == FindingCategory.UNAUTHORIZED_MODEL_CHANGE
    assert fnd.severity == SeverityLevel.CRITICAL
    assert fnd.suggested_disposition == SuggestedDisposition.QUARANTINE
    assert "State-Dict Fingerprint Mismatch" in fnd.title
    assert fnd.details["expected_fingerprint"] == expected_fp
    assert fnd.details["observed_fingerprint"] == "a" * 64

    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.signal_name == "state_dict_fingerprint_mismatch"
    assert ev.metadata["expected_fingerprint"] == expected_fp
    assert ev.metadata["observed_fingerprint"] == "a" * 64


@pytest.mark.parametrize(
    "bad_fp",
    [
        "not-a-hash",
        "a" * 63,  # too short
        "a" * 65,  # too long
        "g" * 64,  # non-hex character
        12345,  # non-string
        "",  # empty
        None,  # Handled as omitted if None, but tested here when non-valid string
    ],
)
def test_malformed_fingerprint_fails_closed(
    base_context: AssessmentContext, bad_fp: object
) -> None:
    """Malformed or invalid hex string yields CapabilityStatus.FAILED rather than false match or silent pass."""
    if bad_fp is None:
        # None is omitted, tested separately
        return

    base_context.config["expected_state_dict_fingerprint"] = bad_fp

    detector = StateDictIntegrityDetector()
    result = detector.evaluate(base_context)

    assert result.status == CapabilityStatus.FAILED
    assert "Malformed expected_state_dict_fingerprint" in (result.status_reason or "")
    assert len(result.findings) == 0
