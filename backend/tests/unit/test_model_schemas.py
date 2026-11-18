"""Unit tests for Pydantic v2 Model API schemas.

Verifies strict extra="forbid" behavior, frozen immutability, field constraints,
and serialization/deserialization for model integrity API schemas.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas.model import (
    CreateModelAssessmentRequest,
    ModelDetailResponse,
    ModelListResponse,
    ModelUploadResponse,
)


def test_model_upload_response_valid():
    """Verifies valid ModelUploadResponse instantiation and immutability."""
    now = datetime.now(UTC)
    resp = ModelUploadResponse(
        model_id="model_12345678",
        name="test_model.pt",
        format="pytorch",
        status="RECEIVED",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        size_bytes=1024,
        created_at=now,
        updated_at=now,
        metadata={"key": "val"},
        snapshot_id=None,
        job_id="job_abc123",
    )
    assert resp.model_id == "model_12345678"
    assert resp.format == "pytorch"
    assert resp.status == "RECEIVED"
    assert resp.job_id == "job_abc123"

    # Frozen immutability verification
    with pytest.raises(ValidationError):
        resp.status = "ACCEPTED"


def test_model_upload_response_extra_forbidden():
    """Verifies that extra unexpected attributes raise ValidationError."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError) as exc_info:
        ModelUploadResponse(
            model_id="model_123",
            name="test.pt",
            format="pytorch",
            status="RECEIVED",
            sha256="abc",
            size_bytes=100,
            created_at=now,
            updated_at=now,
            unauthorized_extra_field="malicious",
        )
    assert "extra_forbidden" in str(exc_info.value)


def test_model_detail_response_inheritance():
    """Verifies ModelDetailResponse fields and constraints."""
    now = datetime.now(UTC)
    detail = ModelDetailResponse(
        model_id="model_999",
        name="resnet18.pt",
        format="pytorch",
        status="ACCEPTED",
        sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        size_bytes=45000000,
        created_at=now,
        updated_at=now,
        snapshot_id="snap_123",
        snapshot_manifest_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        snapshot_file_count=1,
        assessments_count=2,
    )
    assert detail.snapshot_file_count == 1
    assert detail.assessments_count == 2
    assert detail.status == "ACCEPTED"


def test_model_list_response():
    """Verifies ModelListResponse pagination validation."""
    now = datetime.now(UTC)
    item = ModelUploadResponse(
        model_id="model_1",
        name="m1.pt",
        format="pytorch",
        status="RECEIVED",
        sha256="abc",
        size_bytes=10,
        created_at=now,
        updated_at=now,
    )
    list_resp = ModelListResponse(
        page=1,
        page_size=10,
        total_items=1,
        total_pages=1,
        items=[item],
    )
    assert list_resp.page == 1
    assert len(list_resp.items) == 1

    # Invalid negative page
    with pytest.raises(ValidationError):
        ModelListResponse(
            page=0,
            page_size=10,
            total_items=0,
            total_pages=0,
            items=[],
        )


def test_create_model_assessment_request():
    """Verifies CreateModelAssessmentRequest default values and strictness."""
    req = CreateModelAssessmentRequest()
    assert req.name == "Automated Model Assessment"
    assert req.detectors is None
    assert req.config == {}

    custom = CreateModelAssessmentRequest(
        name="Custom Backdoor Check",
        detectors=["model_integrity.backdoor_indicator"],
        config={"eval_mode": "eval"},
    )
    assert custom.detectors == ["model_integrity.backdoor_indicator"]

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        CreateModelAssessmentRequest(unsupported_config_field=True)
