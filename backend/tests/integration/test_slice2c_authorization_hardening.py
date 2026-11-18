"""Adversarial tests for Slice 2C: Publication Authorization Hardening and Quarantine Metadata.

Verifies:
- F-01: SQLite json_extract() eliminates JSON LIKE bypass across compact, pretty,
        escaped, malformed, missing, and type-mismatched payloads.
- F-02: Strict mandatory lease credential enforcement (fail closed on missing credentials,
        stale states, expired leases, or unauthorized callers).
- F-03: Explicit QuarantineMetadataError on audit record failure while preserving quarantined contents,
        and graceful recovery in create_snapshot.
- F-04: Explicit docstring specifications for POSIX advisory vs Windows mandatory locking.
"""

import inspect
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import (
    Base,
    JobModel,
)
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.repositories.job_repository import (
    InvalidJobStateTransitionError,
    JobLeaseExpiredError,
    JobNotClaimedError,
)
from backend.app.db.session import _set_sqlite_pragmas, now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    JobStatus,
    SeverityLevel,
)
from backend.app.domain.entities.finding import (
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    QuarantineMetadataError,
)
from backend.app.modules.data_integrity.ingestion.formats.base import NormalizedSample
from backend.app.modules.data_integrity.ingestion.normalizer import NormalizedDataset
from backend.app.modules.data_integrity.ingestion.snapshot import SnapshotManager


def _build_test_db(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return engine, session_factory


def _setup_assessment_with_job(
    session_factory,
    payload_raw: str | dict,
    worker_id: str = "worker_alpha",
    lease_seconds: int = 300,
):
    session = session_factory()
    try:
        ass_id = "asm_slice2c_test"
        snap_id = "snap_slice2c_dummy"
        ds_id = "ds_slice2c_dummy"

        ds_repo = SqliteDatasetRepository(session)
        ds_repo.create_dataset(
            dataset_id=ds_id,
            name="Slice2C Dataset",
            dataset_format="COCO",
            storage_path="/tmp/slice2c",
            sha256="0" * 64,
            size_bytes=1000,
        )
        ds_repo.create_snapshot(
            snapshot_id=snap_id,
            dataset_id=ds_id,
            manifest_sha256="0" * 64,
            storage_path="/tmp/slice2c",
            file_count=1,
            total_size_bytes=1000,
        )

        ass_repo = SqliteAssessmentRepository(session)
        ass_repo.create_assessment(
            assessment_id=ass_id,
            name="Slice2C Assessment",
            snapshot_id=snap_id,
            config={},
            dataset_id=ds_id,
        )

        # Directly insert or use job queue
        payload_str = json.dumps(payload_raw) if isinstance(payload_raw, dict) else payload_raw
        now = now_utc()
        expires_at = now + timedelta(seconds=lease_seconds)
        job_id = "job_slice2c_01"

        job = JobModel(
            job_id=job_id,
            task_type="run_assessment",
            payload_json=payload_str,
            status=JobStatus.RUNNING,
            attempts=1,
            max_attempts=3,
            locked_by=worker_id,
            locked_at=now,
            lease_token="token_alpha_123",
            lease_expires_at=expires_at,
            lease_generation=1,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
        session.commit()

        return (
            ass_id,
            job_id,
            {
                "locked_by": worker_id,
                "lease_token": "token_alpha_123",
                "lease_generation": 1,
            },
        )
    finally:
        session.close()


def _make_findings(ass_id: str, title: str = "Test Finding"):
    now = now_utc()
    finding = UniversalFinding(
        finding_id=f"fnd_{title.lower().replace(' ', '_')[:20]}",
        assessment_id=ass_id,
        detector_id="data_integrity.structural_validator",
        detector_version="1.0.0",
        category=FindingCategory.STRUCTURAL_DEFECT,
        severity=SeverityLevel.HIGH,
        title=title,
        description="Verification finding for Slice 2C test suite",
        confidence=0.99,
        reproducibility=ReproducibilityMetadata(execution_timestamp=now, seed=42),
        created_at=now,
    )
    detector_run = {
        "run_id": f"run_{ass_id[:8]}",
        "detector_id": "data_integrity.structural_validator",
        "detector_version": "1.0.0",
        "status": CapabilityStatus.COMPLETED,
        "duration_ms": 10,
        "started_at": now,
        "completed_at": now,
    }
    return [detector_run], [finding]


# =========================================================================
# Finding F-01: Structured json_extract() Payload Validation Tests
# =========================================================================


def test_f01_compact_json_payload_authorizes_correctly():
    """Verify that compact JSON (no whitespace around keys/values) authorizes via json_extract()."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            # Compact JSON without spaces: {"assessment_id":"asm_slice2c_test"}
            compact_payload = '{"assessment_id":"asm_slice2c_test","dataset_id":"ds_dummy"}'
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, compact_payload)
            runs, findings = _make_findings(ass_id, "Compact JSON Success")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs,
                    findings=findings,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id=creds["locked_by"],
                    lease_token=creds["lease_token"],
                    lease_generation=creds["lease_generation"],
                )
                fnds = repo.get_findings(ass_id)
                assert len(fnds) == 1
                assert fnds[0].title == "Compact JSON Success"
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_pretty_printed_and_whitespace_variations():
    """Verify that pretty-printed JSON, tabs, newlines, and uneven whitespace authorize identically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            pretty_payload = (
                '{\n\t  "assessment_id"  :  \t "asm_slice2c_test"  , \n "extra" : 123\n}'
            )
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, pretty_payload)
            runs, findings = _make_findings(ass_id, "Pretty JSON Success")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs,
                    findings=findings,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id=creds["locked_by"],
                    lease_token=creds["lease_token"],
                    lease_generation=creds["lease_generation"],
                )
                fnds = repo.get_findings(ass_id)
                assert len(fnds) == 1
                assert fnds[0].title == "Pretty JSON Success"
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_escaped_characters_in_payload():
    """Verify that escaped unicode or special characters in JSON payload do not corrupt extraction."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            escaped_payload = json.dumps(
                {
                    "assessment_id": "asm_slice2c_test",
                    "notes": 'path with "quotes" and \\ slashes and \u2603 unicode snowman',
                }
            )
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, escaped_payload)
            runs, findings = _make_findings(ass_id, "Escaped Success")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs,
                    findings=findings,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id=creds["locked_by"],
                    lease_token=creds["lease_token"],
                    lease_generation=creds["lease_generation"],
                )
                fnds = repo.get_findings(ass_id)
                assert len(fnds) == 1
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_missing_assessment_id_in_payload_rejected():
    """Verify that a job payload lacking assessment_id is rejected with JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            payload_without_id = json.dumps({"dataset_id": "ds_dummy", "other_key": 42})
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, payload_without_id)
            runs, findings = _make_findings(ass_id, "Should Fail")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "assigned to assessment" in str(exc_info.value) or "invalid" in str(
                    exc_info.value
                )
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_malformed_json_payload_rejected_cleanly():
    """Verify that a corrupt non-JSON payload string fails closed with JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            malformed_payload = "NOT_VALID_JSON{:::garbled"
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, malformed_payload)
            runs, findings = _make_findings(ass_id, "Should Fail")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert (
                    "malformed" in str(exc_info.value).lower()
                    or "invalid" in str(exc_info.value).lower()
                )
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_mismatched_assessment_id_rejected():
    """Verify that a job payload referencing a different assessment_id is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            mismatched_payload = json.dumps({"assessment_id": "asm_OTHER_DIFFERENT_123"})
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, mismatched_payload)
            runs, findings = _make_findings(ass_id, "Should Fail")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "assigned to assessment 'asm_OTHER_DIFFERENT_123'" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f01_type_mismatched_assessment_id_rejected():
    """Verify that non-string assessment_id (e.g. integer or list) does not bypass json_extract()."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            numeric_payload = json.dumps({"assessment_id": 999999})
            ass_id, job_id, creds = _setup_assessment_with_job(session_factory, numeric_payload)
            runs, findings = _make_findings(ass_id, "Should Fail")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError):
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


# =========================================================================
# Finding F-02: Mandatory Lease Credentials & Fail-Closed Guards
# =========================================================================


def test_f02_missing_job_id_fails_closed():
    """Verify that calling publish_assessment_results with job_id=None raises JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}
            )
            runs, findings = _make_findings(ass_id, "Unauthenticated")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=None,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "Publication requires all lease credentials" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f02_missing_worker_id_fails_closed():
    """Verify that calling publish_assessment_results with worker_id=None raises JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}
            )
            runs, findings = _make_findings(ass_id, "Missing Worker")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=None,
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "Publication requires all lease credentials" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f02_missing_lease_token_fails_closed():
    """Verify that calling publish_assessment_results with lease_token=None raises JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}
            )
            runs, findings = _make_findings(ass_id, "Missing Token")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=None,
                        lease_generation=creds["lease_generation"],
                    )
                assert "Publication requires all lease credentials" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f02_missing_lease_generation_fails_closed():
    """Verify that calling publish_assessment_results with lease_generation=None raises JobNotClaimedError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}
            )
            runs, findings = _make_findings(ass_id, "Missing Gen")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobNotClaimedError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=None,
                    )
                assert "Publication requires all lease credentials" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f02_completed_or_failed_job_cannot_publish():
    """Verify that a job already in COMPLETED or FAILED state rejects publication."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}
            )
            runs, findings = _make_findings(ass_id, "State Check")

            session = session_factory()
            try:
                # Manually set job to COMPLETED
                job = session.get(JobModel, job_id)
                job.status = JobStatus.COMPLETED
                session.commit()

                repo = SqliteAssessmentRepository(session)
                with pytest.raises(InvalidJobStateTransitionError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "non-running state" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


def test_f02_expired_lease_cannot_publish():
    """Verify that an expired lease raises JobLeaseExpiredError upon publication."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)
        try:
            ass_id, job_id, creds = _setup_assessment_with_job(
                session_factory, {"assessment_id": "asm_slice2c_test"}, lease_seconds=-10
            )
            runs, findings = _make_findings(ass_id, "Expired Check")

            session = session_factory()
            try:
                repo = SqliteAssessmentRepository(session)
                with pytest.raises(JobLeaseExpiredError) as exc_info:
                    repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=creds["locked_by"],
                        lease_token=creds["lease_token"],
                        lease_generation=creds["lease_generation"],
                    )
                assert "expired at" in str(exc_info.value)
                assert len(repo.get_findings(ass_id)) == 0
            finally:
                session.close()
        finally:
            engine.dispose()


# =========================================================================
# Finding F-03: Explicit Quarantine Metadata Failure & Recovery
# =========================================================================


def test_f03_quarantine_metadata_failure_raises_explicit_error():
    """Verify that failure writing quarantine_record.json raises QuarantineMetadataError with contents preserved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        cfg = IngestionConfig(
            quarantine_dir=base / "quarantine",
            staging_dir=base / "staging",
            snapshots_dir=base / "snapshots",
        )
        cfg.quarantine_dir.mkdir(parents=True)
        cfg.snapshots_dir.mkdir(parents=True)

        mgr = SnapshotManager(config=cfg)
        snap_id = "snap_0123456789abcdef"
        snap_dir = cfg.snapshots_dir / snap_id
        snap_dir.mkdir(parents=True)
        test_file = snap_dir / "sample.txt"
        test_file.write_text("critical forensic evidence", encoding="utf-8")

        # Mock Path.open so that writing quarantine_record.json fails
        original_open = Path.open

        def mocked_open(path_self, *args, **kwargs):
            if "quarantine_record.json" in str(path_self):
                raise PermissionError("Simulated disk read-only / metadata write failure")
            return original_open(path_self, *args, **kwargs)

        with (
            patch.object(Path, "open", side_effect=mocked_open),
            pytest.raises(QuarantineMetadataError) as exc_info,
        ):
            mgr.quarantine_snapshot(
                snapshot_id=snap_id,
                reason="Tampering detected",
                verification_errors=["manifest missing"],
            )

        assert "QUARANTINE_METADATA_WRITE_FAILED" in exc_info.value.code
        assert exc_info.value.details["snapshot_id"] == snap_id

        # Verify: original directory was moved into quarantine and evidence preserved!
        assert not snap_dir.exists()
        q_parent = cfg.quarantine_dir / "quarantined_snapshots"
        assert q_parent.exists()
        quarantined_dirs = list(q_parent.iterdir())
        assert len(quarantined_dirs) == 1
        ev_file = quarantined_dirs[0] / "sample.txt"
        assert ev_file.exists()
        assert ev_file.read_text(encoding="utf-8") == "critical forensic evidence"


def test_f03_create_snapshot_recovers_after_quarantine_metadata_failure(capsys):
    """Verify that create_snapshot catches QuarantineMetadataError, logs warning, and creates snapshot."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        cfg = IngestionConfig(
            quarantine_dir=base / "quarantine",
            staging_dir=base / "staging",
            snapshots_dir=base / "snapshots",
        )
        cfg.quarantine_dir.mkdir(parents=True)
        cfg.staging_dir.mkdir(parents=True)
        cfg.snapshots_dir.mkdir(parents=True)

        src_dir = base / "extracted_src"
        src_dir.mkdir(parents=True)
        (src_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        normalized = NormalizedDataset(
            original_format="COCO",
            normalization_version="1.0.0",
            categories=["sample"],
            samples=[
                NormalizedSample(sample_id="s1", image_relative_path="image.png"),
            ],
            metadata={},
        )

        mgr = SnapshotManager(config=cfg)
        # Create valid snapshot first
        snap, path1 = mgr.create_snapshot(src_dir, normalized, asset_id="ds_test")
        snap_id = snap.snapshot_id

        # Corrupt the snapshot so next create_snapshot triggers quarantine
        man_p = path1 / "manifest.json"
        import stat

        man_p.chmod(stat.S_IWRITE | stat.S_IREAD)
        man_p.write_bytes(b"CORRUPTED_MANIFEST")

        # Mock quarantine_snapshot to raise QuarantineMetadataError
        original_quarantine = mgr.quarantine_snapshot

        def fake_quarantine(*args, **kwargs):
            try:
                original_quarantine(*args, **kwargs)
            finally:
                raise QuarantineMetadataError("Simulated metadata write failure during recreation")

        with patch.object(mgr, "quarantine_snapshot", side_effect=fake_quarantine):
            snap2, path2 = mgr.create_snapshot(src_dir, normalized, asset_id="ds_test")

        # Snapshot was successfully remade!
        assert snap2.snapshot_id == snap_id
        assert path2.exists()
        assert (path2 / "manifest.json").exists()

        # Warning logged to stderr
        captured = capsys.readouterr()
        assert "WARNING: Quarantine audit record write failed" in captured.err


# =========================================================================
# Finding F-04: POSIX Advisory Lock Documentation Verification
# =========================================================================


def test_f04_lock_docstrings_document_advisory_vs_mandatory_distinction():
    """Verify that SnapshotManager docstrings explicitly document POSIX advisory vs Windows mandatory locks."""
    mgr = SnapshotManager(config=IngestionConfig())

    lock_doc = inspect.getdoc(mgr._acquire_snapshot_lock)
    assert lock_doc is not None
    assert "mandatory byte-range locks" in lock_doc
    assert "advisory" in lock_doc.lower()
    assert "fcntl.flock" in lock_doc
    assert "non-cooperative" in lock_doc.lower()

    reader_doc = inspect.getdoc(mgr.reader_lock)
    assert reader_doc is not None
    assert "cooperative and in-process only" in reader_doc.lower()
    assert "advisory" in reader_doc.lower()
