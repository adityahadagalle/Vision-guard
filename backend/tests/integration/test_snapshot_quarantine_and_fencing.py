"""Adversarial integration tests for Snapshot Quarantine and Assessment Publication Fencing.

Tests verify Phase 2, Slice 2B objectives:
- Snapshot Quarantine: invalid snapshots are safely quarantined rather than deleted,
  active readers are protected, concurrent retries are mutually excluded and idempotent,
  and symlinks/path manipulation are blocked.
- Publication Fencing: assessment result publication is strictly bound to active job
  leases (job_id, worker_id, lease_token, lease_generation), preventing stale/expired
  workers from overwriting results or corrupting state.
"""

import json
import os
import stat
import tempfile
import threading
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

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
    SqliteJobQueue,
)
from backend.app.db.session import now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    SeverityLevel,
)
from backend.app.domain.entities.finding import (
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    PathTraversalError,
    QuarantineCollisionError,
    SnapshotIntegrityError,
    SnapshotInUseError,
    SymlinkNotPermittedError,
)
from backend.app.modules.data_integrity.ingestion.formats.base import NormalizedSample
from backend.app.modules.data_integrity.ingestion.normalizer import NormalizedDataset
from backend.app.modules.data_integrity.ingestion.snapshot import SnapshotManager


def _build_test_db(db_path: Path):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine, session_factory


def _create_sample_dataset_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    f1 = root / "image_01.png"
    f1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    f2 = root / "image_02.png"
    f2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x11" * 64)
    return root


def _make_sample_normalized_dataset() -> NormalizedDataset:
    return NormalizedDataset(
        original_format="standard_image",
        normalization_version="1.0.0",
        categories=["defect", "nominal"],
        samples=[
            NormalizedSample(sample_id="s1", image_relative_path="image_01.png"),
            NormalizedSample(sample_id="s2", image_relative_path="image_02.png"),
        ],
        metadata={},
    )


# ==============================================================================
# OBJECTIVE A: SNAPSHOT QUARANTINE ADVERSARIAL TESTS
# ==============================================================================


def test_invalid_snapshot_is_quarantined_not_deleted():
    """Verify that an invalid existing snapshot is moved to quarantine with audit record, not deleted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
            staging_dir=tmp_path / "staging",
        )
        mgr = SnapshotManager(config=cfg)

        src_dir = _create_sample_dataset_root(tmp_path / "src_dataset")
        norm = _make_sample_normalized_dataset()

        # 1. Create a clean snapshot
        snap, snap_dir = mgr.create_snapshot(src_dir, norm)
        assert snap_dir.exists()
        original_manifest = snap_dir / "manifest.json"
        assert original_manifest.exists()

        # 2. Corrupt the snapshot by modifying a file
        target_img = snap_dir / "image_01.png"
        os.chmod(target_img, stat.S_IWRITE)
        target_img.write_bytes(b"TAMPERED_DATA_BYTES")
        os.chmod(target_img, stat.S_IREAD)

        # 3. Call create_snapshot again for the same dataset
        # Should detect tampering, quarantine the corrupted snapshot, and recreate cleanly
        new_snap, new_snap_dir = mgr.create_snapshot(src_dir, norm)
        assert new_snap_dir.exists()
        assert new_snap.snapshot_id == snap.snapshot_id

        # 4. Verify that quarantine directory was populated
        quarantine_root = cfg.quarantine_dir / "quarantined_snapshots"
        assert quarantine_root.exists()
        quarantined_dirs = list(quarantine_root.iterdir())
        assert len(quarantined_dirs) == 1
        q_dir = quarantined_dirs[0]
        assert snap.snapshot_id in q_dir.name

        # Verify tampered file was preserved in quarantine for forensics
        tampered_copy = q_dir / "image_01.png"
        assert tampered_copy.exists()
        assert tampered_copy.read_bytes() == b"TAMPERED_DATA_BYTES"

        # Verify audit quarantine record
        q_record_file = q_dir / "quarantine_record.json"
        assert q_record_file.exists()
        q_record = json.loads(q_record_file.read_bytes())
        assert q_record["snapshot_id"] == snap.snapshot_id
        assert q_record["status"] == "QUARANTINED"
        assert len(q_record["verification_errors"]) > 0
        assert q_record["expected_manifest_sha256"] == snap.manifest_sha256

        # Verify new snapshot in snapshots_dir is valid and intact
        verif = mgr.verify_snapshot(new_snap.snapshot_id, new_snap.manifest_sha256)
        assert verif.is_valid


def test_quarantine_destination_collision_fails_closed():
    """Verify that if a quarantine destination already exists, collision error is raised and nothing overwritten."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
            staging_dir=tmp_path / "staging",
        )
        mgr = SnapshotManager(config=cfg)

        src_dir = _create_sample_dataset_root(tmp_path / "src_dataset")
        norm = _make_sample_normalized_dataset()

        snap, snap_dir = mgr.create_snapshot(src_dir, norm)

        # Pre-create colliding quarantine destination
        colliding_dest = cfg.quarantine_dir / "quarantined_snapshots" / "fixed_colliding_quarantine"
        colliding_dest.mkdir(parents=True, exist_ok=True)
        canary_file = colliding_dest / "canary.txt"
        canary_file.write_text("DO_NOT_OVERWRITE", encoding="utf-8")

        # Attempt quarantine targeting the colliding destination
        with pytest.raises(QuarantineCollisionError) as exc_info:
            mgr.quarantine_snapshot(
                snapshot_id=snap.snapshot_id,
                reason="Testing collision rejection",
                destination_override=colliding_dest,
            )

        assert "already exists" in str(exc_info.value)
        # Canary must remain intact
        assert canary_file.exists()
        assert canary_file.read_text(encoding="utf-8") == "DO_NOT_OVERWRITE"
        # Original snapshot must NOT have been deleted
        assert snap_dir.exists()


def test_active_reader_protection_prevents_quarantine():
    """Verify that an active reader lock prevents quarantine and fails closed with SnapshotInUseError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
            staging_dir=tmp_path / "staging",
        )
        mgr = SnapshotManager(config=cfg)

        src_dir = _create_sample_dataset_root(tmp_path / "src_dataset")
        norm = _make_sample_normalized_dataset()
        snap, snap_dir = mgr.create_snapshot(src_dir, norm)

        # Acquire an active reader lock
        with mgr.reader_lock(snap.snapshot_id, reader_id="reader_worker_42"):
            assert mgr.has_active_readers(snap.snapshot_id)
            assert mgr.active_readers_count(snap.snapshot_id) == 1

            # Attempt quarantine while reader is active
            with pytest.raises(SnapshotInUseError) as exc_info:
                mgr.quarantine_snapshot(
                    snapshot_id=snap.snapshot_id,
                    reason="Attempting quarantine during active read",
                )
            assert "active readers registered" in str(exc_info.value)
            # Directory must NOT be moved or deleted
            assert snap_dir.exists()

        # After reader lock exits, active readers count is 0
        assert not mgr.has_active_readers(snap.snapshot_id)


def test_concurrent_retries_for_same_snapshot():
    """Verify two concurrent threads attempting creation/retry for the same snapshot succeed without conflict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
            staging_dir=tmp_path / "staging",
        )
        mgr = SnapshotManager(config=cfg)

        src_dir = _create_sample_dataset_root(tmp_path / "src_dataset")
        norm = _make_sample_normalized_dataset()

        results = []
        errors = []

        def worker_task(thread_id: int):
            try:
                s, p = mgr.create_snapshot(src_dir, norm)
                results.append((thread_id, s.snapshot_id, p, s.manifest_sha256))
            except Exception as e:
                errors.append((thread_id, e))

        t1 = threading.Thread(target=worker_task, args=(1,))
        t2 = threading.Thread(target=worker_task, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Concurrent workers encountered errors: {errors}"
        assert len(results) == 2
        # Both workers must obtain the exact same valid snapshot
        assert results[0][1] == results[1][1]
        final_dir = results[0][2]
        assert final_dir.exists()
        verif = mgr.verify_snapshot(results[0][1], expected_manifest_sha256=results[0][3])
        # Verify valid snapshot exists and is verified
        assert verif.is_valid


def test_symlink_and_path_traversal_blocked_in_snapshot_manager():
    """Verify that path traversal and symlink attempts on snapshot IDs fail closed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
        )
        mgr = SnapshotManager(config=cfg)

        # Path traversal with ..
        with pytest.raises(PathTraversalError):
            mgr._validate_snapshot_id("snap_12345/../../etc")

        with pytest.raises(PathTraversalError):
            mgr._validate_snapshot_id("snap_1234/../traversal")

        # Invalid format
        with pytest.raises(SnapshotIntegrityError):
            mgr._validate_snapshot_id("arbitrary_id_without_prefix")

        # Symlink detection
        real_dir = cfg.snapshots_dir / "snap_1111222233334444"
        real_dir.mkdir(parents=True, exist_ok=True)
        symlink_dir = cfg.snapshots_dir / "snap_aaaabbbbccccdddd"
        try:
            os.symlink(real_dir, symlink_dir, target_is_directory=True)
            with pytest.raises(SymlinkNotPermittedError):
                mgr._validate_snapshot_id("snap_aaaabbbbccccdddd")
        except (OSError, NotImplementedError):
            # If OS does not permit symlink creation without admin rights, skip symlink check
            pass


def test_valid_snapshot_reuse_remains_functional():
    """Verify that an intact snapshot is verified and reused idempotently without touching quarantine."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = IngestionConfig(
            snapshots_dir=tmp_path / "snapshots",
            quarantine_dir=tmp_path / "quarantine",
        )
        mgr = SnapshotManager(config=cfg)

        src_dir = _create_sample_dataset_root(tmp_path / "src_dataset")
        norm = _make_sample_normalized_dataset()

        snap1, path1 = mgr.create_snapshot(src_dir, norm)
        snap2, path2 = mgr.create_snapshot(src_dir, norm)

        assert snap1.snapshot_id == snap2.snapshot_id
        assert path1 == path2
        assert not (cfg.quarantine_dir / "quarantined_snapshots").exists()


# ==============================================================================
# OBJECTIVE B: ASSESSMENT PUBLICATION LEASE FENCING ADVERSARIAL TESTS
# ==============================================================================


def _setup_fencing_assessment(session_factory, lease_seconds: int = 300):
    session = session_factory()
    try:
        ds_repo = SqliteDatasetRepository(session)
        ass_repo = SqliteAssessmentRepository(session)
        job_queue = SqliteJobQueue(session)

        ds_id = f"ds_{uuid.uuid4().hex[:8]}"
        snap_id = f"snap_{uuid.uuid4().hex[:16]}"
        ds_repo.create_dataset(
            dataset_id=ds_id,
            name="Test Fencing Dataset",
            dataset_format="standard_image",
            storage_path="/tmp/fake_raw",
            sha256="0" * 64,
            size_bytes=1024,
        )
        ds_repo.create_snapshot(
            snapshot_id=snap_id,
            dataset_id=ds_id,
            manifest_sha256="a" * 64,
            storage_path="/tmp/fake_storage",
            file_count=10,
            total_size_bytes=1024,
        )

        # 2. Create assessment
        ass_id = f"ass_{uuid.uuid4().hex[:8]}"
        ass_repo.create_assessment(
            assessment_id=ass_id,
            name="Test Fencing Assessment",
            snapshot_id=snap_id,
            config={"detectors": []},
            dataset_id=ds_id,
        )

        # 3. Submit background job
        job_id = job_queue.submit_job(
            task_type="assessment",
            payload={"assessment_id": ass_id, "snapshot_id": snap_id, "dataset_id": ds_id},
        )

        # 4. Claim job by Worker A
        claimed_job = job_queue.claim_next_job(
            worker_id="worker_alpha", lease_duration_seconds=lease_seconds
        )
        assert claimed_job is not None
        assert claimed_job["job_id"] == job_id
        claimed_job["worker_id"] = claimed_job["locked_by"]

        return ass_id, job_id, claimed_job
    finally:
        session.close()


def _make_findings(ass_id: str, title: str = "Fencing Finding"):
    f = UniversalFinding(
        finding_id=f"fnd_{uuid.uuid4().hex[:8]}",
        assessment_id=ass_id,
        detector_id="data_integrity.structural_validator",
        detector_version="1.0.0",
        category=FindingCategory.LABEL_ANOMALY,
        severity=SeverityLevel.HIGH,
        title=title,
        description="Integrity anomaly under test",
        confidence=0.99,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now_utc(),
            seed=42,
        ),
        created_at=now_utc(),
    )
    r = {
        "run_id": f"run_{uuid.uuid4().hex[:8]}",
        "detector_id": "data_integrity.structural_validator",
        "detector_version": "1.0.0",
        "status": CapabilityStatus.COMPLETED,
        "duration_ms": 10,
        "started_at": now_utc(),
        "completed_at": now_utc(),
    }
    return [r], [f]


def test_valid_active_lease_can_publish():
    """Verify that a worker holding a valid, active lease publishes results cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job = _setup_fencing_assessment(session_factory)
            runs, findings = _make_findings(ass_id)

            session = session_factory()
            try:
                ass_repo = SqliteAssessmentRepository(session)

                # Worker alpha publishes with valid credentials
                ass_repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs,
                    findings=findings,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id=job["worker_id"],
                    lease_token=job["lease_token"],
                    lease_generation=job["lease_generation"],
                )

                # Verification: findings exist, status is COMPLETED
                published = ass_repo.get_findings(ass_id)
                assert len(published) == 1
                ass = ass_repo.get_assessment(ass_id)
                assert ass["status"] == AssessmentStatus.COMPLETED
            finally:
                session.close()
        finally:
            engine.dispose()


def test_expired_lease_cannot_publish():
    """Verify that a worker whose lease has expired is rejected with JobLeaseExpiredError and results unchanged."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job = _setup_fencing_assessment(session_factory, lease_seconds=1)
            runs, findings = _make_findings(ass_id)

            # Fast-forward / expire the lease
            session = session_factory()
            try:
                job_model = session.get(JobModel, job_id)
                job_model.lease_expires_at = now_utc() - timedelta(seconds=10)
                session.commit()

                ass_repo = SqliteAssessmentRepository(session)

                # Worker alpha attempts to publish after lease expiry
                with pytest.raises(JobLeaseExpiredError) as exc_info:
                    ass_repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id=job["worker_id"],
                        lease_token=job["lease_token"],
                        lease_generation=job["lease_generation"],
                    )

                assert "expired" in str(exc_info.value).lower()

                # Findings must NOT have been published
                published = ass_repo.get_findings(ass_id)
                assert len(published) == 0
                ass = ass_repo.get_assessment(ass_id)
                assert ass["status"] != AssessmentStatus.COMPLETED
            finally:
                session.close()
        finally:
            engine.dispose()


def test_old_token_cannot_publish_after_reassignment():
    """Verify that after lease reassignment to Worker B, Worker A with old token is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job_alpha = _setup_fencing_assessment(session_factory)

            # Worker Alpha times out, stale recovery occurs, Worker Beta claims job
            session = session_factory()
            try:
                job_queue = SqliteJobQueue(session)
                job_model = session.get(JobModel, job_id)
                job_model.lease_expires_at = now_utc() - timedelta(seconds=10)
                session.commit()

                job_queue.recover_stale_jobs()
                job_beta = job_queue.claim_next_job(
                    worker_id="worker_beta", lease_duration_seconds=300
                )
                assert job_beta is not None
                assert job_beta["locked_by"] == "worker_beta"
                assert job_beta["lease_token"] != job_alpha["lease_token"]

                ass_repo = SqliteAssessmentRepository(session)
                runs_a, findings_a = _make_findings(ass_id)

                # Worker Alpha attempts to publish using old token & generation
                with pytest.raises(JobNotClaimedError) as exc_info:
                    ass_repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs_a,
                        findings=findings_a,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id="worker_alpha",
                        lease_token=job_alpha["lease_token"],
                        lease_generation=job_alpha["lease_generation"],
                    )
                assert "Publication rejected" in str(exc_info.value)

                # Now Worker Beta publishes valid results
                runs_b, findings_b = _make_findings(ass_id, title="Worker Beta Finding")
                ass_repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs_b,
                    findings=findings_b,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id="worker_beta",
                    lease_token=job_beta["lease_token"],
                    lease_generation=job_beta["lease_generation"],
                )

                published = ass_repo.get_findings(ass_id)
                assert len(published) == 1
                assert published[0].title == "Worker Beta Finding"
            finally:
                session.close()
        finally:
            engine.dispose()


def test_same_worker_with_replaced_token_cannot_publish():
    """Verify that if the same worker claims a job again (new token), publication with old token is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job_claim1 = _setup_fencing_assessment(session_factory)

            session = session_factory()
            try:
                job_queue = SqliteJobQueue(session)
                # Expire claim 1 and re-queue
                job_model = session.get(JobModel, job_id)
                job_model.lease_expires_at = now_utc() - timedelta(seconds=10)
                session.commit()

                job_queue.recover_stale_jobs()
                # Same worker claims again
                job_claim2 = job_queue.claim_next_job(
                    worker_id="worker_alpha", lease_duration_seconds=300
                )
                assert job_claim2["lease_token"] != job_claim1["lease_token"]

                ass_repo = SqliteAssessmentRepository(session)
                runs, findings = _make_findings(ass_id)

                # Worker Alpha attempts publication with claim 1 token
                with pytest.raises(JobNotClaimedError):
                    ass_repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs,
                        findings=findings,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id="worker_alpha",
                        lease_token=job_claim1["lease_token"],
                        lease_generation=job_claim1["lease_generation"],
                    )

                # Claim 2 token succeeds
                ass_repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs,
                    findings=findings,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id="worker_alpha",
                    lease_token=job_claim2["lease_token"],
                    lease_generation=job_claim2["lease_generation"],
                )
                assert len(ass_repo.get_findings(ass_id)) == 1
            finally:
                session.close()
        finally:
            engine.dispose()


def test_stale_worker_cannot_overwrite_newer_completed_results():
    """Verify that once a valid worker completes the job, a delayed stale worker cannot overwrite results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job_alpha = _setup_fencing_assessment(session_factory)

            session = session_factory()
            try:
                job_queue = SqliteJobQueue(session)

                # Expire Worker Alpha, Worker Beta claims and finishes
                job_model = session.get(JobModel, job_id)
                job_model.lease_expires_at = now_utc() - timedelta(seconds=5)
                session.commit()
                job_queue.recover_stale_jobs()

                job_beta = job_queue.claim_next_job(worker_id="worker_beta")
                ass_repo = SqliteAssessmentRepository(session)

                # Worker Beta publishes and completes
                runs_b, findings_b = _make_findings(ass_id, title="Valid Beta Result")
                ass_repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs_b,
                    findings=findings_b,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id="worker_beta",
                    lease_token=job_beta["lease_token"],
                    lease_generation=job_beta["lease_generation"],
                )
                job_queue.complete_job(
                    job_id=job_id, worker_id="worker_beta", lease_token=job_beta["lease_token"]
                )

                # Delayed Worker Alpha wakes up and attempts to overwrite
                runs_a, findings_a = _make_findings(ass_id, title="Stale Alpha Overwrite")
                with pytest.raises(InvalidJobStateTransitionError) as exc_info:
                    ass_repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs_a,
                        findings=findings_a,
                        evidence_records=[],
                        final_status=AssessmentStatus.COMPLETED,
                        job_id=job_id,
                        worker_id="worker_alpha",
                        lease_token=job_alpha["lease_token"],
                        lease_generation=job_alpha["lease_generation"],
                    )
                assert "non-running state" in str(exc_info.value).lower()

                # Existing results from Worker Beta must remain intact
                current_findings = ass_repo.get_findings(ass_id)
                assert len(current_findings) == 1
                assert current_findings[0].title == "Valid Beta Result"
            finally:
                session.close()
        finally:
            engine.dispose()


def test_concurrent_publishers_only_authorized_lease_succeeds():
    """Verify two concurrent publishers (one authorized, one stale) preserve data integrity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job_alpha = _setup_fencing_assessment(session_factory)

            session = session_factory()
            try:
                job_queue = SqliteJobQueue(session)

                # Stale Worker Alpha vs Authorized Worker Beta
                job_model = session.get(JobModel, job_id)
                job_model.lease_expires_at = now_utc() - timedelta(seconds=5)
                session.commit()
                job_queue.recover_stale_jobs()
                job_beta = job_queue.claim_next_job(worker_id="worker_beta")

                results = {}

                def publish_alpha():
                    s = session_factory()
                    try:
                        repo = SqliteAssessmentRepository(s)
                        runs, findings = _make_findings(ass_id, title="Alpha")
                        try:
                            repo.publish_assessment_results(
                                assessment_id=ass_id,
                                detector_runs=runs,
                                findings=findings,
                                evidence_records=[],
                                job_id=job_id,
                                worker_id="worker_alpha",
                                lease_token=job_alpha["lease_token"],
                                lease_generation=job_alpha["lease_generation"],
                            )
                            results["alpha"] = "SUCCESS"
                        except Exception as e:
                            results["alpha"] = type(e).__name__
                    finally:
                        s.close()

                def publish_beta():
                    s = session_factory()
                    try:
                        repo = SqliteAssessmentRepository(s)
                        runs, findings = _make_findings(ass_id, title="Beta")
                        try:
                            repo.publish_assessment_results(
                                assessment_id=ass_id,
                                detector_runs=runs,
                                findings=findings,
                                evidence_records=[],
                                job_id=job_id,
                                worker_id="worker_beta",
                                lease_token=job_beta["lease_token"],
                                lease_generation=job_beta["lease_generation"],
                            )
                            results["beta"] = "SUCCESS"
                        except Exception as e:
                            results["beta"] = type(e).__name__
                    finally:
                        s.close()

                t_alpha = threading.Thread(target=publish_alpha)
                t_beta = threading.Thread(target=publish_beta)
                t_alpha.start()
                t_beta.start()
                t_alpha.join()
                t_beta.join()

                # Beta must succeed; Alpha must fail
                assert results["beta"] == "SUCCESS"
                assert results["alpha"] in ("JobNotClaimedError", "InvalidJobStateTransitionError")

                # Assessment has Beta's findings
                check_repo = SqliteAssessmentRepository(session)
                fnds = check_repo.get_findings(ass_id)
                assert len(fnds) == 1
                assert fnds[0].title == "Beta"
            finally:
                session.close()
        finally:
            engine.dispose()


def test_failed_authorization_preserves_previous_results():
    """Verify that a rejected publication leaves all existing findings, evidence, and status untouched."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        engine, session_factory = _build_test_db(db_path)

        try:
            ass_id, job_id, job_alpha = _setup_fencing_assessment(session_factory)
            session = session_factory()
            try:
                ass_repo = SqliteAssessmentRepository(session)

                # 1. Publish baseline results with Worker Alpha
                runs_orig, findings_orig = _make_findings(ass_id, title="Original Baseline Finding")
                ass_repo.publish_assessment_results(
                    assessment_id=ass_id,
                    detector_runs=runs_orig,
                    findings=findings_orig,
                    evidence_records=[],
                    final_status=AssessmentStatus.COMPLETED,
                    job_id=job_id,
                    worker_id="worker_alpha",
                    lease_token=job_alpha["lease_token"],
                    lease_generation=job_alpha["lease_generation"],
                )

                # 2. Attempt unauthorized publication with rogue worker ID
                runs_bad, findings_bad = _make_findings(ass_id, title="Rogue Attacker Finding")
                with pytest.raises(JobNotClaimedError):
                    ass_repo.publish_assessment_results(
                        assessment_id=ass_id,
                        detector_runs=runs_bad,
                        findings=findings_bad,
                        evidence_records=[],
                        final_status=AssessmentStatus.FAILED,
                        job_id=job_id,
                        worker_id="rogue_worker",
                        lease_token="invalid_token",
                        lease_generation=999,
                    )

                # 3. Verify original baseline remains 100% intact
                fnds = ass_repo.get_findings(ass_id)
                assert len(fnds) == 1
                assert fnds[0].title == "Original Baseline Finding"
                asm = ass_repo.get_assessment(ass_id)
                assert asm["status"] == AssessmentStatus.COMPLETED
            finally:
                session.close()
        finally:
            engine.dispose()
