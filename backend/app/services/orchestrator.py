"""Background job orchestrator and workflow execution engine.

Coordinates end-to-end dataset ingestion, snapshot verification, and Tier-1
integrity assessment workflows while ensuring strict state machine invariants.
"""

import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.repositories.job_repository import (
    JobLeaseExpiredError,
    JobNotClaimedError,
    SqliteJobQueue,
)
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.db.session import now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    AssetType,
    CapabilityStatus,
    IngestionStatus,
)
from backend.app.domain.ports.detector import AssessmentContext
from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.data_integrity.detectors import register_tier1_detectors
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.service import IngestionService
from backend.app.modules.data_integrity.ingestion.snapshot import SnapshotManager
from backend.app.modules.model_integrity.artifact import ModelArtifactManager
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.conflict import ModelConflictEngine
from backend.app.modules.model_integrity.detectors import register_model_integrity_detectors

logger = logging.getLogger(__name__)


class AssessmentOrchestrator:
    """Orchestrates asynchronous ingestion and assessment pipelines within the modular monolith."""

    def __init__(
        self,
        session: Session,
        config: IngestionConfig | None = None,
        registry: DetectorRegistry | None = None,
    ) -> None:
        self._session = session
        self.config = config or IngestionConfig()
        self.dataset_repo = SqliteDatasetRepository(session)
        self.assessment_repo = SqliteAssessmentRepository(session)
        self.job_queue = SqliteJobQueue(session)

        # Initialize detector registry with Tier-1 plugins and Model Integrity plugins
        self.registry = registry or DetectorRegistry()
        if not self.registry.list_ids():
            register_tier1_detectors(self.registry)
        if not any(d.startswith("model_integrity.") for d in self.registry.list_ids()):
            register_model_integrity_detectors(self.registry)

    def process_next_job(self, worker_id: str) -> dict[str, Any] | None:
        """Claims and executes the next queued task.

        Returns:
            Completed or failed job record, or None if no tasks are queued.
        """
        job = self.job_queue.claim_next_job(worker_id=worker_id)
        if not job:
            return None

        job_id = job["job_id"]
        task_type = job["task_type"]
        payload = job.get("payload", {})
        lease_token = job.get("lease_token")
        lease_generation = job.get("lease_generation")

        try:
            if task_type == "ingestion":
                self._execute_ingestion(job_id, worker_id, payload, lease_token=lease_token)
            elif task_type == "model_ingestion":
                self._execute_model_ingestion(
                    job_id,
                    worker_id,
                    payload,
                    lease_token=lease_token,
                    lease_generation=lease_generation,
                )
            elif task_type == "assessment":
                self._execute_assessment(
                    job_id,
                    worker_id,
                    payload,
                    lease_token=lease_token,
                    lease_generation=lease_generation,
                )

            else:
                self.job_queue.fail_job(
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    lease_generation=lease_generation,
                    error_message=f"Unknown task_type '{task_type}'.",
                )
        except (JobLeaseExpiredError, JobNotClaimedError) as lease_err:
            logger.warning(
                "Worker '%s' lost active lease for job '%s' (generation %s): %s",
                worker_id,
                job_id,
                lease_generation,
                lease_err,
            )
            return self.job_queue.get_job(job_id)
        except Exception as err:
            with contextlib.suppress(JobLeaseExpiredError, JobNotClaimedError):
                self.job_queue.fail_job(
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    lease_generation=lease_generation,
                    error_message=f"Unhandled exception during task execution: {err}",
                )

        return self.job_queue.get_job(job_id)

    def _execute_ingestion(
        self,
        job_id: str,
        worker_id: str,
        payload: dict[str, Any],
        lease_token: str | None = None,
    ) -> None:
        """Executes secure dataset ingestion, normalization, and snapshot creation."""
        dataset_id = payload["dataset_id"]
        quarantine_path = Path(payload["quarantine_path"])
        filename = payload.get("filename")

        ingestion_service = IngestionService(config=self.config)

        try:
            self.dataset_repo.update_dataset_status(dataset_id, IngestionStatus.INSPECTING)
            result = ingestion_service.ingest_archive(
                stream_or_path=quarantine_path,
                filename=filename,
                asset_id=dataset_id,
            )

            # Persist the immutable snapshot entity in SQLite (idempotent)
            snap = result.snapshot
            self.dataset_repo.create_snapshot(
                snapshot_id=snap.snapshot_id,
                dataset_id=dataset_id,
                manifest_sha256=snap.manifest_sha256,
                storage_path=str(result.snapshot_path),
                file_count=snap.file_count,
                total_size_bytes=snap.total_size_bytes,
                asset_type=snap.asset_type,
                schema_version=snap.schema_version,
                metadata=snap.metadata,
            )

            self.dataset_repo.update_dataset_status(dataset_id, IngestionStatus.ACCEPTED)

            self.job_queue.complete_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                result={
                    "dataset_id": dataset_id,
                    "snapshot_id": snap.snapshot_id,
                    "manifest_sha256": snap.manifest_sha256,
                    "file_count": snap.file_count,
                },
            )
        except Exception as err:
            self.dataset_repo.update_dataset_status(dataset_id, IngestionStatus.REJECTED)
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                error_message=f"Ingestion rejected or failed: {err}",
            )

    def _execute_model_ingestion(
        self,
        job_id: str,
        worker_id: str,
        payload: dict[str, Any],
        lease_token: str | None = None,
        lease_generation: int | None = None,
    ) -> None:
        """Executes secure model checkpoint ingestion and snapshot creation under lease fencing."""
        model_id = payload["model_id"]
        quarantine_path = Path(payload["quarantine_path"])
        filename = payload.get("filename", "model.pt")
        expected_sha256 = payload.get("expected_sha256")

        model_repo = SqliteModelRepository(self._session)
        model_config = ModelIntegrityConfig(
            snapshots_dir=self.config.snapshots_dir,
            quarantine_dir=self.config.quarantine_dir,
            staging_dir=self.config.staging_dir,
        )
        artifact_mgr = ModelArtifactManager(model_config)

        # 1. Lease-fenced state transition: RECEIVED -> INSPECTING
        model_repo.transition_model_to_inspecting(
            model_id=model_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_token=lease_token or "",
            lease_generation=lease_generation or 0,
        )

        try:
            # 2. Ingest model artifact into content-addressed immutable snapshot
            result = artifact_mgr.ingest_model(
                stream_or_path=quarantine_path,
                filename=filename,
                model_id=model_id,
                expected_sha256=expected_sha256,
            )

            # 3. Publish ingestion success atomically under lease fencing
            model_repo.publish_model_ingestion_success(
                model_id=model_id,
                snapshot=result.snapshot,
                storage_path=str(result.snapshot_dir),
                sha256=result.sha256,
                size_bytes=result.size_bytes,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token or "",
                lease_generation=lease_generation or 0,
                result={
                    "model_id": model_id,
                    "snapshot_id": result.snapshot.snapshot_id,
                    "sha256": result.sha256,
                    "size_bytes": result.size_bytes,
                },
            )
        except (JobLeaseExpiredError, JobNotClaimedError):
            # Lease expired or generation changed; fail closed immediately without overwriting newer state
            raise
        except Exception as err:
            with contextlib.suppress(JobLeaseExpiredError, JobNotClaimedError):
                model_repo.publish_model_ingestion_failure(
                    model_id=model_id,
                    error_message=f"Model ingestion rejected or failed: {err}",
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=lease_token or "",
                    lease_generation=lease_generation or 0,
                )
            raise

    def _execute_assessment(
        self,
        job_id: str,
        worker_id: str,
        payload: dict[str, Any],
        lease_token: str | None = None,
        lease_generation: int | None = None,
    ) -> None:
        """Executes full assessment pipeline: verification, Tier-1 detectors, finding persistence."""
        assessment_id = payload["assessment_id"]
        snapshot_id = payload["snapshot_id"]

        assessment = self.assessment_repo.get_assessment(assessment_id)
        if not assessment:
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                error_message=f"Assessment '{assessment_id}' not found.",
            )
            return

        # 1. Transition: INITIALIZING
        self.assessment_repo.update_status(assessment_id, AssessmentStatus.INITIALIZING)

        # 2. Transition: INGESTING (Snapshot verification)
        self.assessment_repo.update_status(assessment_id, AssessmentStatus.INGESTING)

        authoritative_manifest_sha256 = self.dataset_repo.get_authoritative_manifest_sha256(
            snapshot_id
        )
        if not authoritative_manifest_sha256:
            self.assessment_repo.update_status(
                assessment_id,
                AssessmentStatus.FAILED,
                status_reason="Authoritative snapshot manifest digest missing from persistence store.",
            )
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                error_message="Authoritative snapshot manifest digest missing.",
            )
            return

        # Run cryptographic snapshot verification against authoritative digest
        storage_path = self.dataset_repo.get_snapshot_storage_path(snapshot_id)
        if storage_path and Path(storage_path).parent.exists():
            snap_config = IngestionConfig(
                snapshots_dir=Path(storage_path).parent,
                quarantine_dir=self.config.quarantine_dir,
                staging_dir=self.config.staging_dir,
            )
            snapshot_mgr = SnapshotManager(config=snap_config)
        else:
            snapshot_mgr = SnapshotManager(config=self.config)

        verif_result = snapshot_mgr.verify_snapshot(
            snapshot_id=snapshot_id,
            expected_manifest_sha256=authoritative_manifest_sha256,
        )

        if not verif_result.is_valid:
            error_details = "; ".join(verif_result.errors)
            self.assessment_repo.update_status(
                assessment_id,
                AssessmentStatus.FAILED,
                status_reason=f"Snapshot verification failed: {error_details}",
            )
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                error_message=f"Snapshot verification failed: {error_details}",
            )
            return

        # 3. Transition: ANALYZING
        self.assessment_repo.update_status(assessment_id, AssessmentStatus.ANALYZING)

        snapshot_entity = self.dataset_repo.get_snapshot(snapshot_id)

        config = assessment.get("config", {})
        requested_detectors = config.get("detectors")

        registered_detectors = self.registry.list_all()
        if requested_detectors:
            detectors_to_run = [
                d for d in registered_detectors if d.detector_id in requested_detectors
            ]
        elif snapshot_entity and snapshot_entity.asset_type == AssetType.MODEL_CHECKPOINT:
            detectors_to_run = [
                d for d in registered_detectors if d.requirements.requires_model_checkpoint
            ]
        else:
            detectors_to_run = [
                d for d in registered_detectors if not d.requirements.requires_model_checkpoint
            ]

        any_detector_failed = False
        detector_runs: list[dict[str, Any]] = []
        all_findings = []
        all_evidence = []

        context = AssessmentContext(
            snapshot=snapshot_entity,
            config={
                **config,
                "snapshot_root": storage_path,
                "assessment_id": assessment_id,
                "job_id": job_id,
                "lease_generation": lease_generation,
                "_db_session": self._session,
                "_model_repo": SqliteModelRepository(self._session),
            },
        )

        for detector in detectors_to_run:
            run_id = f"run_{uuid.uuid4().hex}"
            started_at = now_utc()
            t0 = time.perf_counter()

            try:
                result = detector.evaluate(context)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                completed_at = now_utc()

                if result.status == CapabilityStatus.FAILED:
                    any_detector_failed = True

                detector_runs.append(
                    {
                        "run_id": run_id,
                        "detector_id": detector.detector_id,
                        "detector_version": detector.version,
                        "status": result.status,
                        "duration_ms": duration_ms,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "status_reason": result.status_reason,
                        "metrics": result.metadata,
                    }
                )

                # Collect findings and evidence
                for finding in result.findings:
                    f_with_run = finding.model_copy(update={"run_id": run_id})
                    all_findings.append(f_with_run)

                all_evidence.extend(result.evidence)

            except Exception as err:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                completed_at = now_utc()
                any_detector_failed = True

                detector_runs.append(
                    {
                        "run_id": run_id,
                        "detector_id": detector.detector_id,
                        "detector_version": detector.version,
                        "status": CapabilityStatus.FAILED,
                        "duration_ms": duration_ms,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "status_reason": f"Detector crashed with unexpected exception: {err}",
                        "metrics": {},
                    }
                )

        # 4. Transition: AGGREGATING
        self.assessment_repo.update_status(assessment_id, AssessmentStatus.AGGREGATING)

        # ADR-012: Multi-detector visible conflict evaluation
        is_model_assessment = (
            snapshot_entity and snapshot_entity.asset_type == AssetType.MODEL_CHECKPOINT
        ) or any(r["detector_id"].startswith("model_integrity.") for r in detector_runs)
        if is_model_assessment:
            conflict_engine = ModelConflictEngine()
            sample_rel_path = Path(storage_path).name if storage_path else None
            conflict_findings, conflict_evidence = conflict_engine.evaluate_conflicts(
                assessment_id=assessment_id,
                snapshot_id=snapshot_id,
                asset_id=snapshot_entity.asset_id if snapshot_entity else assessment_id,
                detector_runs=detector_runs,
                findings=all_findings,
                evidence_records=all_evidence,
                context_config=config,
                sample_path=sample_rel_path,
            )
            all_findings.extend(conflict_findings)
            all_evidence.extend(conflict_evidence)

        # 5. Final Assessment Resolution & Atomic Results Publication
        if any_detector_failed:
            status_reason = "One or more detectors encountered unhandled failures during analysis."
            self.assessment_repo.publish_assessment_results(
                assessment_id=assessment_id,
                detector_runs=detector_runs,
                findings=all_findings,
                evidence_records=all_evidence,
                final_status=AssessmentStatus.FAILED,
                status_reason=status_reason,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
            )
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                error_message="Assessment failed: detector execution failure detected.",
            )
        else:
            self.assessment_repo.publish_assessment_results(
                assessment_id=assessment_id,
                detector_runs=detector_runs,
                findings=all_findings,
                evidence_records=all_evidence,
                final_status=AssessmentStatus.COMPLETED,
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
            )
            self.job_queue.complete_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                result={
                    "assessment_id": assessment_id,
                    "findings_count": len(all_findings),
                    "evidence_count": len(all_evidence),
                },
            )
