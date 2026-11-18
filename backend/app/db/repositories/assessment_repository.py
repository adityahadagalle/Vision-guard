"""SQLite repository adapter for assessment lifecycles, runs, findings, and evidence."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from backend.app.db.models import (
    AssessmentModel,
    AssetSnapshotModel,
    DetectorRunModel,
    EvidenceRecordModel,
    FindingAffectedAssetModel,
    FindingModel,
    JobModel,
)
from backend.app.db.repositories.job_repository import (
    InvalidJobStateTransitionError,
    JobLeaseExpiredError,
    JobNotClaimedError,
)
from backend.app.db.session import now_utc
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    JobStatus,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.domain.ports.assessment_repository import AssessmentRepositoryPort


class SqliteAssessmentRepository(AssessmentRepositoryPort):
    """Stores and retrieves assessment lifecycles, runs, universal findings, and evidence."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_assessment(
        self,
        assessment_id: str,
        name: str,
        snapshot_id: str,
        config: dict[str, Any],
        dataset_id: str | None = None,
        model_id: str | None = None,
        auto_commit: bool = True,
    ) -> None:
        """Initializes a new assessment record."""
        # Resolve dataset_id or model_id from snapshot if not explicitly provided
        effective_dataset_id = dataset_id
        effective_model_id = model_id
        if not effective_dataset_id and not effective_model_id:
            snap_stmt = select(AssetSnapshotModel).where(
                AssetSnapshotModel.snapshot_id == snapshot_id
            )
            snap = self._session.execute(snap_stmt).scalar_one_or_none()
            if not snap:
                raise ValueError(f"Snapshot '{snapshot_id}' does not exist.")
            effective_dataset_id = snap.dataset_id
            effective_model_id = snap.model_id
            if not effective_dataset_id and not effective_model_id:
                raise ValueError(
                    f"Snapshot '{snapshot_id}' has neither associated dataset nor model."
                )

        config_canonical_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        config_digest = hashlib.sha256(config_canonical_bytes).hexdigest()

        now = now_utc()
        model = AssessmentModel(
            assessment_id=assessment_id,
            name=name,
            dataset_id=effective_dataset_id,
            model_id=effective_model_id,
            snapshot_id=snapshot_id,
            status=AssessmentStatus.PENDING,
            status_reason=None,
            config_digest=config_digest,
            config_json=json.dumps(config, sort_keys=True),
            started_at=now,
            completed_at=None,
        )
        self._session.add(model)
        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        """Retrieves assessment metadata, status, and summary by identifier."""
        stmt = select(AssessmentModel).where(AssessmentModel.assessment_id == assessment_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is None:
            return None

        # Fetch detector runs
        runs_stmt = select(DetectorRunModel).where(DetectorRunModel.assessment_id == assessment_id)
        runs = self._session.execute(runs_stmt).scalars().all()

        config = json.loads(model.config_json) if model.config_json else {}

        return {
            "assessment_id": model.assessment_id,
            "name": model.name,
            "dataset_id": model.dataset_id,
            "model_id": model.model_id,
            "snapshot_id": model.snapshot_id,
            "status": model.status,
            "status_reason": model.status_reason,
            "config_digest": model.config_digest,
            "config": config,
            "started_at": model.started_at,
            "completed_at": model.completed_at,
            "runs": [
                {
                    "run_id": r.run_id,
                    "detector_id": r.detector_id,
                    "detector_version": r.detector_version,
                    "status": r.status,
                    "status_reason": r.status_reason,
                    "duration_ms": r.duration_ms,
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                }
                for r in runs
            ],
        }

    def update_status(
        self,
        assessment_id: str,
        status: AssessmentStatus,
        status_reason: str | None = None,
    ) -> None:
        """Updates the operational state of an assessment."""
        stmt = select(AssessmentModel).where(AssessmentModel.assessment_id == assessment_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is None:
            raise KeyError(f"Assessment '{assessment_id}' not found.")

        model.status = status
        model.status_reason = status_reason
        if status in (
            AssessmentStatus.COMPLETED,
            AssessmentStatus.FAILED,
            AssessmentStatus.CANCELLED,
        ):
            model.completed_at = now_utc()
        self._session.commit()

    def record_detector_run(
        self,
        run_id: str,
        assessment_id: str,
        detector_id: str,
        detector_version: str,
        status: CapabilityStatus,
        duration_ms: int,
        started_at: datetime,
        completed_at: datetime | None = None,
        status_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Records a detector run with capability state and execution metrics."""
        existing = self._session.get(DetectorRunModel, run_id)
        if existing is not None:
            self._session.delete(existing)
            self._session.flush()

        model = DetectorRunModel(
            run_id=run_id,
            assessment_id=assessment_id,
            detector_id=detector_id,
            detector_version=detector_version,
            status=status,
            status_reason=status_reason,
            duration_ms=duration_ms,
            started_at=started_at,
            completed_at=completed_at,
            metrics_json=json.dumps(metrics or {}, sort_keys=True),
        )
        self._session.add(model)
        self._session.commit()

    def record_findings(
        self,
        assessment_id: str,
        findings: list[UniversalFinding],
    ) -> None:
        """Stores integrity findings produced during an assessment."""
        for finding in findings:
            existing = self._session.get(FindingModel, finding.finding_id)
            if existing is not None:
                self._session.execute(
                    delete(FindingAffectedAssetModel).where(
                        FindingAffectedAssetModel.finding_id == finding.finding_id
                    )
                )
                self._session.execute(
                    delete(EvidenceRecordModel).where(
                        EvidenceRecordModel.finding_id == finding.finding_id
                    )
                )
                self._session.delete(existing)
                self._session.flush()

            finding_model = FindingModel(
                finding_id=finding.finding_id,
                assessment_id=assessment_id,
                run_id=finding.run_id,
                detector_id=finding.detector_id,
                detector_version=finding.detector_version,
                category=finding.category,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                anomaly_score=finding.anomaly_score,
                confidence=finding.confidence,
                suggested_disposition=finding.suggested_disposition,
                status=finding.status,
                detector_config_json=json.dumps(
                    finding.reproducibility.detector_config, sort_keys=True
                ),
                reproducibility_json=json.dumps(
                    {
                        "seed": finding.reproducibility.seed,
                        "software_versions": finding.reproducibility.software_versions,
                        "execution_timestamp": finding.reproducibility.execution_timestamp.isoformat(),
                    },
                    sort_keys=True,
                ),
                created_at=finding.created_at,
            )
            self._session.add(finding_model)

            for aff in finding.affected_assets:
                aff_model = FindingAffectedAssetModel(
                    finding_id=finding.finding_id,
                    asset_id=aff.asset_id,
                    sample_path=aff.sample_path,
                    sha256=aff.sha256,
                )
                self._session.add(aff_model)

        self._session.commit()

    def record_evidence(
        self,
        evidence_records: list[EvidenceRecord],
    ) -> None:
        """Stores forensic evidence records supporting findings."""
        for ev in evidence_records:
            existing = self._session.get(EvidenceRecordModel, ev.evidence_id)
            if existing is not None:
                self._session.delete(existing)
                self._session.flush()

            ev_model = EvidenceRecordModel(
                evidence_id=ev.evidence_id,
                finding_id=ev.finding_id,
                evidence_type=ev.evidence_type,
                signal_name=ev.signal_name,
                rationale=ev.rationale,
                created_at=ev.created_at,
                sample_reference=ev.sample_reference,
                evidence_digest=ev.evidence_digest,
                observed_value=ev.observed_value,
                threshold_value=ev.threshold_value,
                details_json=json.dumps(ev.metadata, sort_keys=True),
            )
            self._session.add(ev_model)
        self._session.commit()

    def publish_assessment_results(
        self,
        assessment_id: str,
        detector_runs: list[dict[str, Any]],
        findings: list[UniversalFinding],
        evidence_records: list[EvidenceRecord],
        final_status: AssessmentStatus = AssessmentStatus.COMPLETED,
        status_reason: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        lease_token: str | None = None,
        lease_generation: int | None = None,
    ) -> None:
        """Atomically commits detector runs, findings, evidence, and final assessment status in a single transaction.

        Mandatory worker lease fencing:
            Every call requires all four lease credentials: ``job_id``, ``worker_id``,
            ``lease_token``, and ``lease_generation``. Missing or incomplete credentials
            fail closed immediately with ``JobNotClaimedError``. Unauthenticated or
            unfenced publication is prohibited.

        Validation rules:
            - Validates that the active lease is held by ``worker_id`` with ``lease_token``
              and matching ``lease_generation``.
            - Enforces that the job is in ``RUNNING`` state and has not expired.
            - Uses SQLite ``json_extract()`` to atomically verify that the job's payload
              matches the provided ``assessment_id``, resisting JSON formatting differences.
            - Purges prior partial runs, findings, and evidence for this assessment_id to
              guarantee retry safety and idempotent publication.
        """
        # Step 0: Mandatory Lease Credentials Guard (F-02: Fail closed if any credential is missing)
        if job_id is None or worker_id is None or lease_token is None or lease_generation is None:
            raise JobNotClaimedError(
                "Publication requires all lease credentials: job_id, worker_id, lease_token, "
                "and lease_generation. Unauthenticated publication is not permitted."
            )

        now = now_utc()

        # Step 1: Atomic Lease Fencing & Payload Verification (F-01: json_extract)
        stmt = text(
            """
            UPDATE jobs
            SET updated_at = :now
            WHERE job_id = :job_id
              AND locked_by = :worker_id
              AND lease_token = :lease_token
              AND lease_generation = :lease_generation
              AND status = 'RUNNING'
              AND lease_expires_at >= :now
              AND json_valid(payload_json) = 1
              AND json_extract(payload_json, '$.assessment_id') = :assessment_id
            RETURNING job_id, payload_json;
            """
        )
        res = self._session.execute(
            stmt,
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "lease_token": lease_token,
                "lease_generation": lease_generation,
                "assessment_id": assessment_id,
                "now": now,
            },
        ).fetchone()

        if not res:
            # Fencing check failed! Rollback and inspect reasons to raise exact structured error
            self._session.rollback()
            job_model = self._session.execute(
                select(JobModel).where(JobModel.job_id == job_id)
            ).scalar_one_or_none()

            if job_model is None:
                raise KeyError(f"Job '{job_id}' not found.")

            if (
                job_model.status != JobStatus.RUNNING.value
                and job_model.status != JobStatus.RUNNING
            ):
                raise InvalidJobStateTransitionError(
                    f"Cannot publish assessment results: job '{job_id}' is in non-running state '{job_model.status}'."
                )

            if job_model.locked_by != worker_id or job_model.lease_token != lease_token:
                raise JobNotClaimedError(
                    f"Publication rejected: worker '{worker_id}' with token '{lease_token}' does not hold active "
                    f"lease for job '{job_id}' (held by '{job_model.locked_by}' with token '{job_model.lease_token}')."
                )

            if job_model.lease_generation != lease_generation:
                raise JobNotClaimedError(
                    f"Publication rejected: lease generation {lease_generation} does not match active "
                    f"generation {job_model.lease_generation} for job '{job_id}'."
                )

            if job_model.lease_expires_at is not None and job_model.lease_expires_at < now:
                raise JobLeaseExpiredError(
                    f"Publication rejected: lease for job '{job_id}' expired at "
                    f"{job_model.lease_expires_at.isoformat()} (current time: {now.isoformat()})."
                )

            # Check assessment_id in payload using structured JSON parsing
            try:
                payload_data = (
                    json.loads(job_model.payload_json)
                    if isinstance(job_model.payload_json, str)
                    else job_model.payload_json
                )
                if (
                    not isinstance(payload_data, dict)
                    or payload_data.get("assessment_id") != assessment_id
                ):
                    raise JobNotClaimedError(
                        f"Publication rejected: job '{job_id}' is assigned to assessment "
                        f"'{payload_data.get('assessment_id') if isinstance(payload_data, dict) else None}', "
                        f"not '{assessment_id}'."
                    )
            except Exception as exc:
                if isinstance(exc, JobNotClaimedError):
                    raise
                raise JobNotClaimedError(
                    f"Publication rejected: invalid or malformed job payload: {exc}"
                ) from exc

            raise JobNotClaimedError(
                f"Publication rejected: lease authorization failed for job '{job_id}'."
            )

        # 1. Purge prior findings and child models for this assessment
        existing_finding_ids = list(
            self._session.execute(
                select(FindingModel.finding_id).where(FindingModel.assessment_id == assessment_id)
            )
            .scalars()
            .all()
        )
        if existing_finding_ids:
            self._session.execute(
                delete(FindingAffectedAssetModel).where(
                    FindingAffectedAssetModel.finding_id.in_(existing_finding_ids)
                )
            )
            self._session.execute(
                delete(EvidenceRecordModel).where(
                    EvidenceRecordModel.finding_id.in_(existing_finding_ids)
                )
            )
            self._session.execute(
                delete(FindingModel).where(FindingModel.assessment_id == assessment_id)
            )

        # 2. Purge prior detector runs
        self._session.execute(
            delete(DetectorRunModel).where(DetectorRunModel.assessment_id == assessment_id)
        )

        # 3. Add detector runs
        for run in detector_runs:
            run_model = DetectorRunModel(
                run_id=run["run_id"],
                assessment_id=assessment_id,
                detector_id=run["detector_id"],
                detector_version=run["detector_version"],
                status=run["status"],
                status_reason=run.get("status_reason"),
                duration_ms=run.get("duration_ms", 0),
                started_at=run["started_at"],
                completed_at=run.get("completed_at"),
                metrics_json=json.dumps(run.get("metrics") or {}, sort_keys=True),
            )
            self._session.add(run_model)

        # 4. Add findings
        for finding in findings:
            finding_model = FindingModel(
                finding_id=finding.finding_id,
                assessment_id=assessment_id,
                run_id=finding.run_id,
                detector_id=finding.detector_id,
                detector_version=finding.detector_version,
                category=finding.category,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                anomaly_score=finding.anomaly_score,
                confidence=finding.confidence,
                suggested_disposition=finding.suggested_disposition,
                status=finding.status,
                detector_config_json=json.dumps(
                    finding.reproducibility.detector_config, sort_keys=True
                ),
                reproducibility_json=json.dumps(
                    {
                        "seed": finding.reproducibility.seed,
                        "software_versions": finding.reproducibility.software_versions,
                        "execution_timestamp": finding.reproducibility.execution_timestamp.isoformat(),
                    },
                    sort_keys=True,
                ),
                created_at=finding.created_at,
            )
            self._session.add(finding_model)

            for aff in finding.affected_assets:
                aff_model = FindingAffectedAssetModel(
                    finding_id=finding.finding_id,
                    asset_id=aff.asset_id,
                    sample_path=aff.sample_path,
                    sha256=aff.sha256,
                )
                self._session.add(aff_model)

        # 5. Add evidence records
        for ev in evidence_records:
            ev_model = EvidenceRecordModel(
                evidence_id=ev.evidence_id,
                finding_id=ev.finding_id,
                evidence_type=ev.evidence_type,
                signal_name=ev.signal_name,
                rationale=ev.rationale,
                created_at=ev.created_at,
                sample_reference=ev.sample_reference,
                evidence_digest=ev.evidence_digest,
                observed_value=ev.observed_value,
                threshold_value=ev.threshold_value,
                details_json=json.dumps(ev.metadata, sort_keys=True),
            )
            self._session.add(ev_model)

        # 6. Update assessment model status and completion
        stmt = select(AssessmentModel).where(AssessmentModel.assessment_id == assessment_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is not None:
            model.status = final_status
            model.status_reason = status_reason
            if final_status in (
                AssessmentStatus.COMPLETED,
                AssessmentStatus.FAILED,
                AssessmentStatus.CANCELLED,
            ):
                model.completed_at = now

        self._session.commit()

    def get_findings(self, assessment_id: str) -> list[UniversalFinding]:
        """Retrieves all findings associated with a specific assessment."""
        findings, _ = self.get_findings_paginated(assessment_id=assessment_id, offset=0, limit=None)
        return findings

    def get_findings_paginated(
        self,
        assessment_id: str,
        offset: int = 0,
        limit: int | None = 50,
        category: str | None = None,
        severity: str | None = None,
    ) -> tuple[list[UniversalFinding], int]:
        """Retrieves paginated findings with optional category and severity filters.

        Returns:
            Tuple of (list of UniversalFinding domain entities, total count).
        """
        query = select(FindingModel).where(FindingModel.assessment_id == assessment_id)
        count_query = select(func.count(FindingModel.finding_id)).where(
            FindingModel.assessment_id == assessment_id
        )

        if category:
            query = query.where(FindingModel.category == category)
            count_query = count_query.where(FindingModel.category == category)
        if severity:
            query = query.where(FindingModel.severity == severity)
            count_query = count_query.where(FindingModel.severity == severity)

        total_count = self._session.execute(count_query).scalar_one()

        query = query.order_by(FindingModel.created_at.asc())
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        models = self._session.execute(query).scalars().all()

        findings: list[UniversalFinding] = []
        for m in models:
            # Load affected assets
            aff_stmt = select(FindingAffectedAssetModel).where(
                FindingAffectedAssetModel.finding_id == m.finding_id
            )
            aff_models = self._session.execute(aff_stmt).scalars().all()
            affected_assets = [
                AffectedAssetReference(
                    asset_id=a.asset_id,
                    sample_path=a.sample_path,
                    sha256=a.sha256,
                )
                for a in aff_models
            ]

            # Load evidence ids
            ev_stmt = select(EvidenceRecordModel.evidence_id).where(
                EvidenceRecordModel.finding_id == m.finding_id
            )
            evidence_ids = list(self._session.execute(ev_stmt).scalars().all())

            repro_raw = json.loads(m.reproducibility_json) if m.reproducibility_json else {}
            detector_config = json.loads(m.detector_config_json) if m.detector_config_json else {}
            raw_ts = (
                datetime.fromisoformat(repro_raw["execution_timestamp"])
                if "execution_timestamp" in repro_raw
                else m.created_at
            )
            ts = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=UTC)

            reproducibility = ReproducibilityMetadata(
                execution_timestamp=ts,
                seed=repro_raw.get("seed"),
                software_versions=repro_raw.get("software_versions", {}),
                detector_config=detector_config,
            )

            created_at = m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC)
            finding = UniversalFinding(
                finding_id=m.finding_id,
                assessment_id=m.assessment_id,
                detector_id=m.detector_id,
                detector_version=m.detector_version,
                category=FindingCategory(m.category),
                severity=SeverityLevel(m.severity),
                title=m.title,
                description=m.description,
                confidence=m.confidence if m.confidence is not None else 1.0,
                reproducibility=reproducibility,
                created_at=created_at,
                run_id=m.run_id,
                status=CapabilityStatus(m.status),
                suggested_disposition=SuggestedDisposition(m.suggested_disposition),
                anomaly_score=m.anomaly_score,
                affected_assets=affected_assets,
                evidence_ids=evidence_ids,
                details=detector_config,
            )
            findings.append(finding)

        return findings, total_count

    def get_evidence_for_finding(self, finding_id: str) -> list[EvidenceRecord]:
        """Retrieves all evidence records supporting a finding."""
        stmt = (
            select(EvidenceRecordModel)
            .where(EvidenceRecordModel.finding_id == finding_id)
            .order_by(EvidenceRecordModel.created_at.asc())
        )
        models = self._session.execute(stmt).scalars().all()
        result: list[EvidenceRecord] = []
        for m in models:
            meta = json.loads(m.details_json) if m.details_json else {}
            created_at = m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC)
            ev = EvidenceRecord(
                evidence_id=m.evidence_id,
                evidence_type=EvidenceType(m.evidence_type),
                finding_id=m.finding_id,
                signal_name=m.signal_name,
                rationale=m.rationale,
                created_at=created_at,
                sample_reference=m.sample_reference,
                evidence_digest=m.evidence_digest,
                observed_value=m.observed_value,
                threshold_value=m.threshold_value,
                metadata=meta,
            )
            result.append(ev)
        return result
