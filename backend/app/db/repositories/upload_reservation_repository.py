"""SQLite repository for durable cross-process upload quota and concurrency reservations.

Ensures strict cross-process concurrency limits, global quarantine disk quota enforcement,
atomic reservations before streaming bytes, and safe recovery of orphaned partial uploads
without deleting or modifying accepted forensic snapshots.
"""

import contextlib
import logging
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.app.db.models import UploadReservationModel
from backend.app.db.session import now_utc


class UploadQuotaError(Exception):
    """Base exception for upload reservation and quota failures."""

    def __init__(
        self, message: str, code: str = "UPLOAD_QUOTA_ERROR", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class UploadConcurrencyLimitExceededError(UploadQuotaError):
    """Raised when concurrent upload quota across API processes is exhausted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="CONCURRENCY_LIMIT_EXCEEDED", details=details)


class QuarantineQuotaExceededError(UploadQuotaError):
    """Raised when global quarantine storage capacity or active reservations exceed quota."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="QUOTA_EXCEEDED", details=details)


class DiskSpaceExhaustedError(UploadQuotaError):
    """Raised when host filesystem partition lacks adequate disk space."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="DISK_SPACE_EXHAUSTED", details=details)


class ReservationExpiredError(UploadQuotaError):
    """Raised when an upload reservation has expired or lapsed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="RESERVATION_EXPIRED", details=details)


logger = logging.getLogger("visionguard.upload_quota")


def get_directory_size_bytes(directory: Path, exclude_subpaths: set[str] | None = None) -> int:
    """Computes total bytes in a directory tree excluding symlinks and specific paths."""
    if not directory.exists():
        return 0
    total = 0
    excludes = exclude_subpaths or set()
    try:
        for p in directory.rglob("*"):
            if any(str(p).startswith(exc) for exc in excludes):
                continue
            if p.is_file() and not p.is_symlink():
                with contextlib.suppress(OSError):
                    total += p.stat().st_size
    except OSError as err:
        logger.warning("Error computing directory size for %s: %s", directory, err)
    return total


class SqliteUploadReservationRepository:
    """Manages cross-process atomic upload reservations in SQLite."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def acquire_reservation(
        self,
        asset_type: str,
        requested_bytes: int,
        quarantine_dir: Path,
        max_concurrent_uploads: int = 3,
        global_quarantine_quota_bytes: int = 5 * 1024 * 1024 * 1024,
        timeout_seconds: int = 900,
        safety_margin_bytes: int = 50 * 1024 * 1024,
        client_ip: str | None = None,
    ) -> UploadReservationModel:
        """Atomically acquires a concurrency slot and reserves quota bytes.

        Args:
            asset_type: Type of asset being uploaded ("model" or "dataset").
            requested_bytes: Anticipated maximum upload size (e.g. 500 MB).
            quarantine_dir: Target quarantine directory.
            max_concurrent_uploads: Maximum concurrent active uploads allowed.
            global_quarantine_quota_bytes: Global quarantine storage quota.
            timeout_seconds: Lease timeout before reservation is marked expired.
            safety_margin_bytes: Minimum disk partition head-room required.
            client_ip: Optional client identifier for audit logging.

        Returns:
            Durable UploadReservationModel instance.

        Raises:
            UploadConcurrencyLimitExceededError: When active slots >= max_concurrent or DB locked.
            QuarantineQuotaExceededError: When committed + reserved bytes exceed global quota.
            DiskSpaceExhaustedError: When host disk partition lacks adequate space.
        """
        # 1. Acquire SQLite IMMEDIATE write lock before reading counts or quotas
        bind = self._session.get_bind()
        if bind and bind.dialect.name == "sqlite":
            if self._session.in_transaction():
                self._session.commit()
            try:
                self._session.execute(text("BEGIN IMMEDIATE"))
            except Exception as err:
                err_msg = str(err).lower()
                if "cannot start a transaction within a transaction" in err_msg:
                    # An explicit or implicit transaction is already active on this session
                    pass
                elif "busy" in err_msg or "locked" in err_msg:
                    logger.warning("SQLite busy/locked during BEGIN IMMEDIATE: %s", err)
                    raise UploadConcurrencyLimitExceededError(
                        "Upload slot unavailable due to database concurrency contention. Please retry.",
                        details={"sqlite_error": str(err)},
                    ) from err
                else:
                    raise

        now = now_utc()

        # 2. Reconcile expired active reservations within the same atomic transaction
        stmt_expire = (
            select(UploadReservationModel)
            .where(UploadReservationModel.status == "ACTIVE")
            .where(UploadReservationModel.expires_at < now)
        )
        stale_records = list(self._session.execute(stmt_expire).scalars().all())
        q_dir_resolved = quarantine_dir.resolve()
        for stale in stale_records:
            if stale.quarantine_path:
                target = Path(stale.quarantine_path).resolve()
                if str(target).startswith(str(q_dir_resolved)) and target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink(missing_ok=True)
                        if target.parent != q_dir_resolved and target.parent.exists():
                            shutil.rmtree(target.parent, ignore_errors=True)
            stale.status = "EXPIRED"
            stale.updated_at = now
        # Do not intermediate commit here: keep entire check-and-insert in one transaction

        # 3. Concurrency limit check
        active_count_stmt = (
            select(func.count(UploadReservationModel.reservation_id))
            .where(UploadReservationModel.status == "ACTIVE")
            .where(UploadReservationModel.expires_at >= now)
        )
        active_count = self._session.execute(active_count_stmt).scalar() or 0
        if active_count >= max_concurrent_uploads:
            self._session.rollback()
            raise UploadConcurrencyLimitExceededError(
                f"Maximum concurrent upload limit ({max_concurrent_uploads}) reached across processes. "
                f"Active uploads in progress: {active_count}."
            )

        # 4. Quota calculation
        active_reservations_stmt = (
            select(UploadReservationModel)
            .where(UploadReservationModel.status == "ACTIVE")
            .where(UploadReservationModel.expires_at >= now)
        )
        active_reservations = list(self._session.execute(active_reservations_stmt).scalars().all())
        active_reserved_bytes = sum(r.reserved_bytes for r in active_reservations)
        active_in_flight_paths = {
            str(Path(r.quarantine_path).resolve()) for r in active_reservations if r.quarantine_path
        }

        # Committed disk bytes in quarantine not in active in-flight paths
        committed_disk_bytes = get_directory_size_bytes(
            quarantine_dir, exclude_subpaths=active_in_flight_paths
        )

        projected_total = committed_disk_bytes + active_reserved_bytes + requested_bytes
        if projected_total > global_quarantine_quota_bytes:
            self._session.rollback()
            raise QuarantineQuotaExceededError(
                f"Global quarantine storage quota ({global_quarantine_quota_bytes} bytes) exceeded. "
                f"Current committed disk: {committed_disk_bytes} bytes, "
                f"active reservations: {active_reserved_bytes} bytes, "
                f"requested: {requested_bytes} bytes (projected: {projected_total} bytes)."
            )

        # 5. Host disk partition check
        try:
            free_space = shutil.disk_usage(quarantine_dir).free
            if free_space < requested_bytes + safety_margin_bytes:
                self._session.rollback()
                raise DiskSpaceExhaustedError(
                    f"Host filesystem has insufficient free space ({free_space} bytes) "
                    f"for requested reservation of {requested_bytes} bytes."
                )
        except OSError as err:
            logger.warning("Could not probe disk free space on %s: %s", quarantine_dir, err)

        # 6. Atomically insert and commit reservation
        res_id = f"res_{uuid.uuid4().hex}"
        reservation = UploadReservationModel(
            reservation_id=res_id,
            asset_type=asset_type,
            reserved_bytes=requested_bytes,
            actual_bytes=None,
            quarantine_path=None,
            status="ACTIVE",
            client_ip=client_ip,
            expires_at=now + timedelta(seconds=timeout_seconds),
            created_at=now,
            updated_at=now,
        )
        self._session.add(reservation)
        self._session.commit()
        return reservation

    def update_reservation_path(self, reservation_id: str, quarantine_path: str) -> None:
        """Associates the in-progress quarantine file path with the reservation."""
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.reservation_id == reservation_id
        )
        res = self._session.execute(stmt).scalar_one_or_none()
        if res:
            res.quarantine_path = quarantine_path
            res.updated_at = now_utc()
            self._session.commit()

    def complete_reservation(
        self,
        reservation_id: str,
        actual_bytes: int,
        quarantine_path: str,
        auto_commit: bool = True,
    ) -> UploadReservationModel:
        """Marks an upload reservation as completed with actual byte count.

        Args:
            reservation_id: ID of the reservation to complete.
            actual_bytes: Total number of streamed bytes.
            quarantine_path: Authoritative quarantine file path.
            auto_commit: Whether to commit immediately or flush within an external transaction.
        """
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.reservation_id == reservation_id
        )
        res = self._session.execute(stmt).scalar_one_or_none()
        if not res:
            raise KeyError(f"Upload reservation '{reservation_id}' not found.")

        now = now_utc()
        if res.status != "ACTIVE" or res.expires_at < now:
            raise ReservationExpiredError(
                f"Upload reservation '{reservation_id}' is no longer active (status: '{res.status}', expired: {res.expires_at < now}).",
                details={
                    "reservation_id": reservation_id,
                    "status": res.status,
                    "expires_at": str(res.expires_at),
                },
            )

        res.status = "COMPLETED"
        res.actual_bytes = actual_bytes
        res.quarantine_path = quarantine_path
        res.updated_at = now

        if auto_commit:
            self._session.commit()
        else:
            self._session.flush()
        return res

    def renew_reservation_lease(
        self,
        reservation_id: str,
        extension_seconds: int = 900,
    ) -> UploadReservationModel:
        """Renews an active reservation lease for in-flight streaming uploads.

        Args:
            reservation_id: The ID of the reservation being streamed.
            extension_seconds: Number of seconds from now to extend expires_at.

        Returns:
            Updated UploadReservationModel.

        Raises:
            ReservationExpiredError: If the reservation does not exist, is not ACTIVE,
                or has already expired.
        """
        now = now_utc()
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.reservation_id == reservation_id
        )
        res = self._session.execute(stmt).scalar_one_or_none()
        if not res:
            raise KeyError(f"Upload reservation '{reservation_id}' not found.")

        if res.status != "ACTIVE" or res.expires_at < now:
            raise ReservationExpiredError(
                f"Cannot renew upload reservation '{reservation_id}': status is '{res.status}' and expires_at is '{res.expires_at}' (current: '{now}').",
                details={
                    "reservation_id": reservation_id,
                    "status": res.status,
                    "expires_at": str(res.expires_at),
                },
            )

        res.expires_at = now + timedelta(seconds=extension_seconds)
        res.updated_at = now
        self._session.commit()
        return res

    def release_reservation(
        self,
        reservation_id: str,
        reason: str = "RELEASED",
        cleanup_file: bool = True,
        quarantine_dir: Path | None = None,
    ) -> None:
        """Releases the reservation slot and cleans up any orphaned partial upload."""
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.reservation_id == reservation_id
        )
        res = self._session.execute(stmt).scalar_one_or_none()
        if not res:
            return

        # Do not allow releasing or modifying a completed reservation
        if res.status == "COMPLETED":
            logger.warning(
                "Attempted to release already completed reservation %s; ignoring.", reservation_id
            )
            return

        if cleanup_file and res.quarantine_path and quarantine_dir:
            target = Path(res.quarantine_path).resolve()
            q_dir = quarantine_dir.resolve()
            # DEFENSE IN DEPTH: strictly ensure target is inside quarantine directory
            if str(target).startswith(str(q_dir)) and target.exists():
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
                    if target.parent != q_dir and target.parent.exists():
                        shutil.rmtree(target.parent, ignore_errors=True)

        res.status = reason
        res.updated_at = now_utc()
        self._session.commit()

    def reconcile_and_cleanup_stale(
        self,
        quarantine_dir: Path,
    ) -> int:
        """Startup and periodic reconciliation of orphaned quarantine uploads.

        Safely cleans up stale files while never modifying accepted snapshots or models.
        """
        now = now_utc()
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.status == "ACTIVE",
            UploadReservationModel.expires_at < now,
        )
        stale_records = list(self._session.execute(stmt).scalars().all())
        cleaned_count = 0
        q_dir_resolved = quarantine_dir.resolve()

        for rec in stale_records:
            if rec.quarantine_path:
                target = Path(rec.quarantine_path).resolve()
                if str(target).startswith(str(q_dir_resolved)) and target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink(missing_ok=True)
                        if target.parent != q_dir_resolved and target.parent.exists():
                            shutil.rmtree(target.parent, ignore_errors=True)
            rec.status = "EXPIRED"
            rec.updated_at = now
            cleaned_count += 1

        if stale_records:
            self._session.commit()
            logger.info("Reconciled and cleared %d expired upload reservation(s).", cleaned_count)

        return cleaned_count

    def get_reservation(self, reservation_id: str) -> dict[str, Any] | None:
        """Retrieves an upload reservation by ID."""
        stmt = select(UploadReservationModel).where(
            UploadReservationModel.reservation_id == reservation_id
        )
        res = self._session.execute(stmt).scalar_one_or_none()
        if not res:
            return None
        return {
            "reservation_id": res.reservation_id,
            "asset_type": res.asset_type,
            "reserved_bytes": res.reserved_bytes,
            "actual_bytes": res.actual_bytes,
            "quarantine_path": res.quarantine_path,
            "status": res.status,
            "client_ip": res.client_ip,
            "expires_at": res.expires_at,
            "created_at": res.created_at,
            "updated_at": res.updated_at,
        }
