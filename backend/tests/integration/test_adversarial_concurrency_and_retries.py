"""Adversarial tests for SQLite job queue concurrency, lease fencing, and retry safety.

Covers all 16 required verification scenarios across multi-process concurrency,
lease token fencing, stale job recovery, startup reconciliation, ingestion idempotency,
and assessment publication atomicity.
"""

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import (
    Base,
    JobModel,
)
from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.repositories.job_repository import (
    JobLeaseExpiredError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.session import _set_sqlite_pragmas, now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    IngestionStatus,
    JobStatus,
    SeverityLevel,
)
from backend.app.domain.entities.finding import (
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.service import IngestionService
from backend.app.modules.data_integrity.ingestion.snapshot import SnapshotManager
from backend.app.services.orchestrator import AssessmentOrchestrator
from backend.tests.integration.test_end_to_end_assessment import _create_coco_zip


def _multiprocess_claim_worker(db_path: str, worker_id: str) -> list[str]:
    """Module-level worker target executed inside separate OS processes."""
    eng = create_engine(f"sqlite:///{db_path}", echo=False)
    event.listen(eng, "connect", _set_sqlite_pragmas)
    session_factory = sessionmaker(bind=eng)
    session = session_factory()
    queue = SqliteJobQueue(session)

    claimed: list[str] = []
    while True:
        job = queue.claim_next_job(worker_id=worker_id, lease_duration_seconds=300)
        if not job:
            break
        claimed.append(job["job_id"])

    session.close()
    eng.dispose()
    return claimed


@pytest.fixture
def env(tmp_path: Path):
    """Sets up an isolated SQLite database in WAL mode and workspace directories."""
    db_file = tmp_path / "adversarial.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    config = IngestionConfig(
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        snapshots_dir=tmp_path / "snapshots",
    )

    yield session_factory, config, str(db_file.as_posix())
    engine.dispose()


# ============================================================================
# PART 1: Concurrency and Claiming (Scenarios 1-4)
# ============================================================================


def test_scenario_01_multiprocess_concurrency(env, tmp_path: Path):
    """Scenario 1: 20 jobs claimed by 6 independent OS processes; 0 duplicates, 0 lost."""
    session_factory, _, db_path = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    # Enqueue 20 jobs
    job_ids = [queue.submit_job(task_type="ingestion", payload={"index": i}) for i in range(20)]
    session.close()

    # Launch 6 separate OS processes via subprocess.Popen
    worker_names = [f"proc_worker_{i}" for i in range(6)]
    procs = []
    out_files = []

    for name in worker_names:
        out_f = tmp_path / f"{name}.json"
        out_files.append(out_f)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.tests.integration.adversarial_worker",
                db_path,
                name,
                str(out_f),
            ]
        )
        procs.append(p)

    for p in procs:
        p.wait(timeout=30)
        assert p.returncode == 0

    all_claimed: list[str] = []
    for out_f in out_files:
        claimed = json.loads(out_f.read_text(encoding="utf-8"))
        all_claimed.extend(claimed)

    # Assert exact counts and zero duplicates
    assert len(all_claimed) == 20
    assert len(set(all_claimed)) == 20
    assert set(all_claimed) == set(job_ids)

    # Verify database state: all 20 in RUNNING
    verify_session = session_factory()
    models = verify_session.execute(select(JobModel)).scalars().all()
    assert len(models) == 20
    assert all(m.status == JobStatus.RUNNING for m in models)
    verify_session.close()


def test_scenario_02_concurrent_claim_and_stale_recovery(env):
    """Scenario 2: Concurrent claim and queue-recovery race never double-processes jobs."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="test_task", payload={})

    # Worker 1 claims the job
    claimed = queue.claim_next_job(worker_id="worker_1", lease_duration_seconds=300)
    assert claimed is not None
    assert claimed["job_id"] == jid

    # Running stale job recovery immediately does NOT recover active unexpired job
    recovered = queue.recover_stale_jobs(stale_threshold_seconds=300)
    assert recovered == 0

    # Worker 2 attempts to claim from queue; must return None
    assert queue.claim_next_job(worker_id="worker_2") is None
    session.close()


def test_scenario_03_queue_empty_behavior(env):
    """Scenario 3: 4 workers attempt to claim from empty queue; all return None cleanly."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    results = [queue.claim_next_job(worker_id=f"worker_{i}") for i in range(4)]
    assert all(r is None for r in results)
    session.close()


def test_scenario_04_fifo_ordering_preservation(env):
    """Scenario 4: 10 jobs submitted in sequence; claimed in strict FIFO order."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    submitted_ids = []
    for i in range(10):
        submitted_ids.append(queue.submit_job(task_type="ordered_task", payload={"seq": i}))

    claimed_ids = []
    for _ in range(10):
        claimed = queue.claim_next_job(worker_id="fifo_worker")
        assert claimed is not None
        claimed_ids.append(claimed["job_id"])

    assert claimed_ids == submitted_ids
    session.close()


# ============================================================================
# PART 2: Lease Fencing (Scenarios 5-9)
# ============================================================================


def test_scenario_05_stale_worker_completion_rejected(env):
    """Scenario 5: Stale worker completion rejected after lease expires and job re-claimed."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="fenced_task", payload={})
    w1_claim = queue.claim_next_job(worker_id="worker_1", lease_duration_seconds=10)
    assert w1_claim is not None
    w1_token = w1_claim["lease_token"]

    # Simulate lease expiry
    m = session.get(JobModel, jid)
    m.lease_expires_at = now_utc() - timedelta(seconds=1)
    session.commit()

    # Re-queue expired job and let Worker 2 claim it
    queue.recover_stale_jobs(stale_threshold_seconds=0)
    w2_claim = queue.claim_next_job(worker_id="worker_2", lease_duration_seconds=300)
    assert w2_claim is not None
    assert w2_claim["lease_token"] != w1_token

    # Worker 1 attempts completion with stale lease -> MUST BE REJECTED
    with pytest.raises(JobNotClaimedError):
        queue.complete_job(job_id=jid, worker_id="worker_1", lease_token=w1_token)

    # Worker 2 successfully completes
    queue.complete_job(job_id=jid, worker_id="worker_2", lease_token=w2_claim["lease_token"])
    assert queue.get_job(jid)["status"] == JobStatus.COMPLETED
    session.close()


def test_scenario_06_stale_worker_failure_rejected(env):
    """Scenario 6: Stale worker failure rejected after lease expires and job re-claimed."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="fail_test", payload={})
    w1_claim = queue.claim_next_job(worker_id="worker_1", lease_duration_seconds=10)
    w1_token = w1_claim["lease_token"]

    # Expire lease and re-claim by Worker 2
    m = session.get(JobModel, jid)
    m.lease_expires_at = now_utc() - timedelta(seconds=1)
    session.commit()
    queue.recover_stale_jobs(stale_threshold_seconds=0)
    assert queue.claim_next_job(worker_id="worker_2") is not None

    # Worker 1 attempts failure with stale lease -> MUST BE REJECTED
    with pytest.raises(JobNotClaimedError):
        queue.fail_job(
            job_id=jid, worker_id="worker_1", lease_token=w1_token, error_message="Late fail"
        )

    # Job remains cleanly in Worker 2's lease
    assert queue.get_job(jid)["locked_by"] == "worker_2"
    session.close()


def test_scenario_07_matching_worker_stale_token_rejected(env):
    """Scenario 7: Worker with matching worker_id but stale lease_token rejected."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="token_test", payload={})
    c1 = queue.claim_next_job(worker_id="worker_same", lease_duration_seconds=5)
    token_old = c1["lease_token"]

    # Simulate worker crash and recovery
    m = session.get(JobModel, jid)
    m.lease_expires_at = now_utc() - timedelta(seconds=1)
    session.commit()
    queue.recover_stale_jobs(stale_threshold_seconds=0)

    # Same worker claims again and gets new token
    c2 = queue.claim_next_job(worker_id="worker_same", lease_duration_seconds=300)
    token_new = c2["lease_token"]
    assert token_old != token_new

    # Completing with old token rejected even though worker_id matches
    with pytest.raises(JobNotClaimedError):
        queue.complete_job(job_id=jid, worker_id="worker_same", lease_token=token_old)

    # Completing with new token succeeds
    queue.complete_job(job_id=jid, worker_id="worker_same", lease_token=token_new)
    assert queue.get_job(jid)["status"] == JobStatus.COMPLETED
    session.close()


def test_scenario_08_lease_renewal_prevents_stale_recovery(env):
    """Scenario 8: Worker successfully renews lease; job is NOT recovered by stale recovery."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="renewal_task", payload={})
    c = queue.claim_next_job(worker_id="worker_active", lease_duration_seconds=2)
    tok = c["lease_token"]

    # Renew lease before expiry
    new_expires = queue.renew_lease(
        job_id=jid, worker_id="worker_active", lease_token=tok, additional_seconds=300
    )
    assert new_expires > now_utc()

    # Stale recovery does NOT touch this job
    recovered = queue.recover_stale_jobs(stale_threshold_seconds=0)
    assert recovered == 0
    assert queue.get_job(jid)["status"] == JobStatus.RUNNING

    # Worker finishes job successfully
    queue.complete_job(job_id=jid, worker_id="worker_active", lease_token=tok)
    assert queue.get_job(jid)["status"] == JobStatus.COMPLETED
    session.close()


def test_scenario_09_renewal_after_expiration_rejected(env):
    """Scenario 9: Renewal after lease expiration is rejected; job recovered for re-claim."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    jid = queue.submit_job(task_type="late_renewal", payload={})
    c = queue.claim_next_job(worker_id="worker_slow", lease_duration_seconds=10)
    tok = c["lease_token"]

    # Artificially expire lease
    m = session.get(JobModel, jid)
    m.lease_expires_at = now_utc() - timedelta(seconds=5)
    session.commit()

    # Renewal attempt is rejected
    with pytest.raises(JobLeaseExpiredError):
        queue.renew_lease(job_id=jid, worker_id="worker_slow", lease_token=tok)

    # Stale recovery re-queues the job
    assert queue.recover_stale_jobs(stale_threshold_seconds=0) == 1
    assert queue.get_job(jid)["status"] == JobStatus.QUEUED
    session.close()


# ============================================================================
# PART 3: Startup Recovery (Scenarios 10-11)
# ============================================================================


def test_scenario_10_startup_recovery_preserves_unexpired_leases(env):
    """Scenario 10: Startup reconciliation recovers expired jobs and preserves active leases."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    j_a = queue.submit_job(task_type="job_a", payload={})
    j_b = queue.submit_job(task_type="job_b", payload={})
    j_c = queue.submit_job(task_type="job_c", payload={})

    queue.claim_next_job(worker_id="w_a")
    queue.claim_next_job(worker_id="w_b")
    queue.claim_next_job(worker_id="w_c")

    # Set up scenario:
    # Job A: lease expired 10s ago
    # Job B: lease expires in 200s (active concurrent worker)
    # Job C: NULL lease_expires_at, locked 400s ago
    m_a = session.get(JobModel, j_a)
    m_a.lease_expires_at = now_utc() - timedelta(seconds=10)

    m_b = session.get(JobModel, j_b)
    m_b.lease_expires_at = now_utc() + timedelta(seconds=200)

    m_c = session.get(JobModel, j_c)
    m_c.lease_expires_at = None
    m_c.locked_at = now_utc() - timedelta(seconds=400)
    session.commit()

    # Reconcile on startup
    recovered_count = queue.recover_strands_on_startup()
    assert recovered_count == 2  # Recovers A and C only

    assert queue.get_job(j_a)["status"] == JobStatus.QUEUED
    assert queue.get_job(j_b)["status"] == JobStatus.RUNNING  # Preserved!
    assert queue.get_job(j_c)["status"] == JobStatus.QUEUED
    session.close()


def test_scenario_11_startup_recovery_respects_max_attempts(env):
    """Scenario 11: Startup recovery re-queues if attempts < max_attempts, fails if exhausted."""
    session_factory, _, _ = env
    session = session_factory()
    queue = SqliteJobQueue(session)

    # Job A: max_attempts=3, attempts=1
    j_a = queue.submit_job(task_type="retriable", payload={}, max_attempts=3)
    queue.claim_next_job(worker_id="w_a")

    # Job B: max_attempts=1, attempts=1 (exhausted)
    j_b = queue.submit_job(task_type="non_retriable", payload={}, max_attempts=1)
    queue.claim_next_job(worker_id="w_b")

    # Expire leases
    m_a = session.get(JobModel, j_a)
    m_a.lease_expires_at = now_utc() - timedelta(seconds=10)
    m_b = session.get(JobModel, j_b)
    m_b.lease_expires_at = now_utc() - timedelta(seconds=10)
    session.commit()

    assert queue.recover_strands_on_startup() == 2
    assert queue.get_job(j_a)["status"] == JobStatus.QUEUED
    assert queue.get_job(j_b)["status"] == JobStatus.FAILED
    session.close()


# ============================================================================
# PART 4: Ingestion Retry Safety (Scenarios 12-14)
# ============================================================================


def test_scenario_12_corrupted_snapshot_cleanup_and_recovery(env):
    """Scenario 12: Interrupted snapshot creation (corrupted directory) cleaned up and repaired."""
    session_factory, config, _ = env
    session = session_factory()
    orchestrator = AssessmentOrchestrator(session=session, config=config)

    # Setup archive and dataset
    archive_bytes = _create_coco_zip()
    q_path = config.quarantine_dir / "coco_corrupt.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    q_path.write_bytes(archive_bytes)

    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_corrupt_test",
        name="Corrupt Test",
        dataset_format="AUTODETECT",
        storage_path=str(q_path),
        sha256="0" * 64,
        size_bytes=len(archive_bytes),
        status=IngestionStatus.QUARANTINED,
    )

    # First: manually create a corrupted snapshot directory
    # Determine snapshot ID by running normalizer on archive
    service = IngestionService(config=config)
    res_dry = service.ingest_archive(
        q_path, filename="coco_corrupt.zip", asset_id="ds_corrupt_test"
    )
    valid_snap_id = res_dry.snapshot.snapshot_id
    snap_dir = config.snapshots_dir / valid_snap_id

    # Tamper/corrupt the snapshot directory
    import stat

    manifest_p = snap_dir / "manifest.json"
    manifest_p.chmod(stat.S_IWRITE | stat.S_IREAD)
    manifest_p.write_bytes(b"CORRUPTED_MANIFEST_GARBAGE")

    # Enqueue ingestion job via orchestrator to simulate retry after failure
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_corrupt_test",
            "quarantine_path": str(q_path),
            "filename": "coco_corrupt.zip",
        },
    )

    # Orchestrator should detect corrupt directory, purge it, and create valid snapshot
    job_result = orchestrator.process_next_job(worker_id="repair_worker")
    assert job_result is not None
    assert job_result["status"] == JobStatus.COMPLETED

    # Verify snapshot is valid and verified
    mgr = SnapshotManager(config=config)
    verif = mgr.verify_snapshot(
        valid_snap_id, expected_manifest_sha256=res_dry.snapshot.manifest_sha256
    )
    assert verif.is_valid is True
    session.close()


def test_scenario_13_identical_archive_snapshot_reuse(env):
    """Scenario 13: Ingestion retry for identical archive verifies and reuses snapshot cleanly."""
    session_factory, config, _ = env
    session = session_factory()
    orchestrator = AssessmentOrchestrator(session=session, config=config)

    archive_bytes = _create_coco_zip()
    q_path = config.quarantine_dir / "coco_reuse.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    q_path.write_bytes(archive_bytes)

    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_reuse_test",
        name="Reuse Test",
        dataset_format="AUTODETECT",
        storage_path=str(q_path),
        sha256="0" * 64,
        size_bytes=len(archive_bytes),
        status=IngestionStatus.QUARANTINED,
    )

    # Run ingestion job 1
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_reuse_test",
            "quarantine_path": str(q_path),
            "filename": "coco_reuse.zip",
        },
    )
    res1 = orchestrator.process_next_job(worker_id="w_1")
    assert res1["status"] == JobStatus.COMPLETED
    snap_id = res1["result"]["snapshot_id"]

    # Submit second ingestion job for identical archive (retry simulation)
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_reuse_test",
            "quarantine_path": str(q_path),
            "filename": "coco_reuse.zip",
        },
    )
    res2 = orchestrator.process_next_job(worker_id="w_2")
    assert res2["status"] == JobStatus.COMPLETED
    assert res2["result"]["snapshot_id"] == snap_id
    session.close()


def test_scenario_14_missing_db_record_retry_creates_cleanly(env):
    """Scenario 14: Snapshot exists on disk but DB record missing; retry creates DB record cleanly."""
    session_factory, config, _ = env
    session = session_factory()
    ds_repo = SqliteDatasetRepository(session)

    archive_bytes = _create_coco_zip()
    q_path = config.quarantine_dir / "coco_db_sync.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    q_path.write_bytes(archive_bytes)

    service = IngestionService(config=config)
    res = service.ingest_archive(q_path, filename="coco_db_sync.zip", asset_id="ds_sync_test")
    snap_id = res.snapshot.snapshot_id

    # Register dataset record first (satisfying foreign key constraint)
    ds_repo.create_dataset(
        dataset_id="ds_sync_test",
        name="Sync Test",
        dataset_format="COCO",
        storage_path=str(q_path),
        sha256="0" * 64,
        size_bytes=len(archive_bytes),
    )

    # Simulate missing DB record (e.g. database commit failed during previous run)
    assert ds_repo.get_snapshot(snap_id) is None

    # Create snapshot DB record using idempotent repository method
    created_entity = ds_repo.create_snapshot(
        snapshot_id=snap_id,
        dataset_id="ds_sync_test",
        manifest_sha256=res.snapshot.manifest_sha256,
        storage_path=str(res.snapshot_path),
        file_count=res.snapshot.file_count,
        total_size_bytes=res.snapshot.total_size_bytes,
    )
    assert created_entity.snapshot_id == snap_id

    # Verify subsequent call returns existing entity without error
    repeat_entity = ds_repo.create_snapshot(
        snapshot_id=snap_id,
        dataset_id="ds_sync_test",
        manifest_sha256=res.snapshot.manifest_sha256,
        storage_path=str(res.snapshot_path),
        file_count=res.snapshot.file_count,
        total_size_bytes=res.snapshot.total_size_bytes,
    )
    assert repeat_entity.snapshot_id == snap_id
    session.close()


# ============================================================================
# PART 5: Assessment Retry Safety (Scenarios 15-16)
# ============================================================================


def test_scenario_15_assessment_retry_safe_no_unique_constraint_violation(env):
    """Scenario 15: Re-running completed assessment succeeds without UNIQUE constraint violation."""
    session_factory, config, _ = env
    session = session_factory()
    orchestrator = AssessmentOrchestrator(session=session, config=config)

    # 1. Ingest dataset
    archive_bytes = _create_coco_zip()
    q_path = config.quarantine_dir / "coco_assess.zip"
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    q_path.write_bytes(archive_bytes)

    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_assess_test",
        name="Assess Test",
        dataset_format="AUTODETECT",
        storage_path=str(q_path),
        sha256="0" * 64,
        size_bytes=len(archive_bytes),
        status=IngestionStatus.QUARANTINED,
    )
    orchestrator.job_queue.submit_job(
        task_type="ingestion",
        payload={
            "dataset_id": "ds_assess_test",
            "quarantine_path": str(q_path),
            "filename": "coco_assess.zip",
        },
    )
    ing_res = orchestrator.process_next_job(worker_id="w_ing")
    snap_id = ing_res["result"]["snapshot_id"]

    # 2. Create assessment
    assess_repo = SqliteAssessmentRepository(session)
    assess_id = "asm_retry_test"
    assess_repo.create_assessment(
        assessment_id=assess_id,
        name="Retry Safe Assessment",
        snapshot_id=snap_id,
        config={"detectors": ["data_integrity.structural_validator"]},
        dataset_id="ds_assess_test",
    )

    # Run assessment run 1
    orchestrator.job_queue.submit_job(
        task_type="assessment",
        payload={"assessment_id": assess_id, "snapshot_id": snap_id},
    )
    res1 = orchestrator.process_next_job(worker_id="w_asm_1")
    assert res1["status"] == JobStatus.COMPLETED
    findings_run1 = assess_repo.get_findings(assess_id)

    # Run assessment run 2 (simulating retry of assessment job)
    orchestrator.job_queue.submit_job(
        task_type="assessment",
        payload={"assessment_id": assess_id, "snapshot_id": snap_id},
    )
    res2 = orchestrator.process_next_job(worker_id="w_asm_2")
    assert res2["status"] == JobStatus.COMPLETED

    findings_run2 = assess_repo.get_findings(assess_id)
    # Findings must match count exactly (no duplicate findings)
    assert len(findings_run2) == len(findings_run1)
    session.close()


def test_scenario_16_assessment_partial_failure_cleanup_on_retry(env):
    """Scenario 16: Partial findings from failed assessment replaced cleanly upon retry."""
    session_factory, config, _ = env
    session = session_factory()
    assess_repo = SqliteAssessmentRepository(session)

    # 1. Manually populate partial findings in assessment repository
    assess_id = "asm_partial_fail"
    snap_id = "snap_dummy12345678"

    # Create dummy snapshot record so foreign keys validate
    ds_repo = SqliteDatasetRepository(session)
    ds_repo.create_dataset(
        dataset_id="ds_dummy",
        name="Dummy",
        dataset_format="COCO",
        storage_path="/tmp/dummy",
        sha256="0" * 64,
        size_bytes=100,
    )
    ds_repo.create_snapshot(
        snapshot_id=snap_id,
        dataset_id="ds_dummy",
        manifest_sha256="0" * 64,
        storage_path="/tmp/dummy",
        file_count=1,
        total_size_bytes=100,
    )

    assess_repo.create_assessment(
        assessment_id=assess_id,
        name="Partial Assessment",
        snapshot_id=snap_id,
        config={},
        dataset_id="ds_dummy",
    )

    # Record 1 partial finding
    partial_finding = UniversalFinding(
        finding_id="fnd_partial_001",
        assessment_id=assess_id,
        detector_id="data_integrity.structural_validator",
        detector_version="1.0.0",
        category=FindingCategory.STRUCTURAL_DEFECT,
        severity=SeverityLevel.HIGH,
        title="Partial Corrupt File",
        description="Found during aborted run",
        confidence=1.0,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now_utc(),
            seed=42,
        ),
        created_at=now_utc(),
    )
    assess_repo.record_findings(assess_id, [partial_finding])
    assert len(assess_repo.get_findings(assess_id)) == 1

    # 2. Now simulate atomic publication of a completed re-run with new findings under active worker lease
    job_queue = SqliteJobQueue(session)
    job_id = job_queue.submit_job(
        task_type="run_assessment",
        payload={"assessment_id": assess_id, "snapshot_id": snap_id},
    )
    claimed_job = job_queue.claim_next_job(worker_id="w_scenario_16", lease_duration_seconds=60)
    assert claimed_job is not None
    assert claimed_job["job_id"] == job_id

    new_finding_1 = UniversalFinding(
        finding_id="fnd_clean_001",
        assessment_id=assess_id,
        detector_id="data_integrity.structural_validator",
        detector_version="1.0.0",
        category=FindingCategory.STRUCTURAL_DEFECT,
        severity=SeverityLevel.LOW,
        title="Clean Finding 1",
        description="Fresh run finding",
        confidence=0.9,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now_utc(),
            seed=42,
        ),
        created_at=now_utc(),
    )
    new_finding_2 = UniversalFinding(
        finding_id="fnd_clean_002",
        assessment_id=assess_id,
        detector_id="data_integrity.label_validator",
        detector_version="1.0.0",
        category=FindingCategory.LABEL_ANOMALY,
        severity=SeverityLevel.MEDIUM,
        title="Clean Finding 2",
        description="Fresh run finding",
        confidence=0.95,
        reproducibility=ReproducibilityMetadata(
            execution_timestamp=now_utc(),
            seed=42,
        ),
        created_at=now_utc(),
    )

    detector_runs = [
        {
            "run_id": "run_new_1",
            "detector_id": "data_integrity.structural_validator",
            "detector_version": "1.0.0",
            "status": CapabilityStatus.COMPLETED,
            "duration_ms": 15,
            "started_at": now_utc(),
            "completed_at": now_utc(),
        }
    ]

    assess_repo.publish_assessment_results(
        assessment_id=assess_id,
        detector_runs=detector_runs,
        findings=[new_finding_1, new_finding_2],
        evidence_records=[],
        final_status=AssessmentStatus.COMPLETED,
        job_id=job_id,
        worker_id=claimed_job["locked_by"],
        lease_token=claimed_job["lease_token"],
        lease_generation=claimed_job["lease_generation"],
    )

    # Verify that partial findings were completely purged and only clean findings remain
    final_findings = assess_repo.get_findings(assess_id)
    assert len(final_findings) == 2
    finding_ids = {f.finding_id for f in final_findings}
    assert "fnd_partial_001" not in finding_ids
    assert finding_ids == {"fnd_clean_001", "fnd_clean_002"}

    # Verify assessment status is COMPLETED
    asm = assess_repo.get_assessment(assess_id)
    assert asm["status"] == AssessmentStatus.COMPLETED
    session.close()
