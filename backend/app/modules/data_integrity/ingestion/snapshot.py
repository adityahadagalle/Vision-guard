"""Immutable dataset snapshot creation and verification engine.

Computes streaming SHA-256 digests for all accepted files, constructs canonical
manifests adhering to RFC 8785, enforces immutable filesystem permissions,
and validates snapshot integrity against unauthorized modifications or missing files.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.asset import AssetSnapshot, ManifestEntry
from backend.app.domain.entities.enums import AssetType
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    PathTraversalError,
    QuarantineCollisionError,
    QuarantineMetadataError,
    SnapshotAlreadyExistsError,
    SnapshotIntegrityError,
    SnapshotInUseError,
    SymlinkNotPermittedError,
)
from backend.app.modules.data_integrity.ingestion.normalizer import NormalizedDataset


def _force_remove_tree(path: Path) -> None:
    """Recursively removes a directory tree, resetting read-only flags on Windows if necessary."""

    def _onerror(func: Any, p: str, exc_info: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def canonical_json_bytes(obj: Any) -> bytes:
    """Serializes a JSON-compatible Python object into deterministic key-sorted JSON bytes.

    Lexicographically sorts dictionary keys, enforces compact formatting without
    extraneous whitespace, and encodes to UTF-8. Note: Standard Python json.dumps is
    used here as a deterministic serialization scheme; it is not a strict RFC 8785
    implementation for IEEE 754 float representations or non-BMP surrogate key sorting.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class SnapshotVerificationResult(BaseModel):
    """Forensic report detailing whether a snapshot matches its canonical manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    snapshot_id: str
    manifest_sha256: str
    checked_files: int
    missing_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    extra_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def all_defects(self) -> list[str]:
        """Aggregates root errors, missing files, modified files, and extra files into a single defect list."""
        defects = list(self.errors)
        for m in self.missing_files:
            defects.append(f"Missing file: {m}")
        for m in self.modified_files:
            defects.append(f"Modified file (SHA-256 mismatch): {m}")
        for e in self.extra_files:
            defects.append(f"Unexpected extra file: {e}")
        return defects


class SnapshotManager:
    """Creates, seals, and verifies content-addressed dataset snapshots.

    Note: Read-only filesystem permissions set on snapshot directories provide an
    operational safeguard against accidental file modification, not an adversarial
    kernel-level immutability guarantee. Cryptographic verification relies on the
    authoritative manifest SHA-256 digest.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        self.config.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _validate_snapshot_id(self, snapshot_id: str) -> Path:
        """Validates snapshot ID against path traversal, forbidden characters, and symlinks."""
        if not snapshot_id or not isinstance(snapshot_id, str):
            raise SnapshotIntegrityError("Snapshot ID must be a non-empty string.")

        if "/" in snapshot_id or "\\" in snapshot_id or ".." in snapshot_id:
            raise PathTraversalError(
                f"Path traversal detected in snapshot ID: '{snapshot_id}'.",
                details={"snapshot_id": snapshot_id},
            )

        if not re.match(r"^snap_[a-f0-9]{16,64}$", snapshot_id):
            raise SnapshotIntegrityError(
                f"Invalid snapshot ID format '{snapshot_id}'. Expected 'snap_<hex>'.",
                details={"snapshot_id": snapshot_id},
            )

        target = (self.config.snapshots_dir / snapshot_id).resolve()
        try:
            target.relative_to(self.config.snapshots_dir.resolve())
        except ValueError as err:
            raise PathTraversalError(
                f"Snapshot path escapes snapshot storage root: '{snapshot_id}'.",
                details={"snapshot_id": snapshot_id, "resolved": str(target)},
            ) from err

        if target.is_symlink():
            raise SymlinkNotPermittedError(
                f"Snapshot path '{snapshot_id}' is a symlink, which is prohibited.",
                details={"snapshot_id": snapshot_id},
            )

        return target

    def has_active_readers(self, snapshot_id: str) -> bool:
        """Returns True if any active reader locks are registered for snapshot_id."""
        readers_dir = self.config.snapshots_dir / ".active_readers" / snapshot_id
        if not readers_dir.exists() or not readers_dir.is_dir():
            return False
        return any(readers_dir.iterdir())

    def active_readers_count(self, snapshot_id: str) -> int:
        """Returns the number of active registered reader locks for snapshot_id."""
        readers_dir = self.config.snapshots_dir / ".active_readers" / snapshot_id
        if not readers_dir.exists() or not readers_dir.is_dir():
            return 0
        return len(list(readers_dir.iterdir()))

    @contextmanager
    def reader_lock(
        self, snapshot_id: str, reader_id: str | None = None
    ) -> Generator[str, None, None]:
        """Context manager registering an active reader on a snapshot.

        While a reader lock is active, quarantine and destructive snapshot operations
        are prohibited and will fail closed to preserve read consistency.

        Coordination model:
            This mechanism is **cooperative and in-process only**. Reader locks are
            tracked by writing a marker file to a shared ``.active_readers`` directory.
            On POSIX, a non-cooperative external process that does not use this context
            manager can read snapshot files concurrently with a quarantine operation,
            because POSIX ``fcntl.flock`` is advisory. On Windows, an open file handle
            prevents directory rename at the OS level regardless of this reader lock.
        """
        self._validate_snapshot_id(snapshot_id)
        target_dir = self.config.snapshots_dir / snapshot_id
        if not target_dir.exists() or not target_dir.is_dir():
            raise SnapshotIntegrityError(
                f"Cannot acquire reader lock: snapshot '{snapshot_id}' does not exist.",
                details={"snapshot_id": snapshot_id},
            )

        rid = reader_id or f"reader_{uuid.uuid4().hex[:8]}"
        readers_dir = self.config.snapshots_dir / ".active_readers" / snapshot_id
        readers_dir.mkdir(parents=True, exist_ok=True)
        marker = readers_dir / f"{rid}.lock"
        marker.write_text(f"{os.getpid()}:{datetime.now(UTC).isoformat()}", encoding="utf-8")
        try:
            yield rid
        finally:
            try:
                if marker.exists():
                    marker.unlink()
                if readers_dir.exists() and not any(readers_dir.iterdir()):
                    readers_dir.rmdir()
            except Exception:
                pass

    @contextmanager
    def _acquire_snapshot_lock(self, snapshot_id: str) -> Generator[None, None, None]:
        """Cross-process and cross-thread file lock protecting creation and quarantine of a specific snapshot.

        Platform behavior:
            Windows (nt): Uses ``msvcrt.locking`` with ``LK_NBLCK`` mode, which provides
                mandatory byte-range locks enforced by the Windows kernel. Any process
                attempting to access the locked byte range will receive an OS-level error,
                regardless of whether it cooperates. This provides strong cross-process
                mutual exclusion.

            POSIX (Linux, macOS): Uses ``fcntl.flock(LOCK_EX)``, which provides **advisory**
                locks only. Advisory locks are enforced only between cooperating processes
                that also call ``flock``. A non-cooperative process that opens or reads
                snapshot files directly (without calling ``flock``) is **not blocked** and
                can read files concurrently with a quarantine rename operation.

                The ``reader_lock`` context manager and its file-marker system are also
                cooperative and in-process only. They do not protect against external,
                non-cooperative readers on POSIX.

                Where mandatory cross-process exclusion is required on POSIX, additional
                OS-level controls (directory permissions, mount options, kernel namespaces)
                must be provisioned externally. VisionGuard does not provide this guarantee
                in the current implementation.

        Raises:
            SnapshotInUseError: If the lock cannot be acquired within 10 seconds (Windows
                mandatory timeout) or if the lock file cannot be created.
        """
        locks_dir = self.config.snapshots_dir / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / f"{snapshot_id}.lock"

        f = lock_file.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                t0 = time.perf_counter()
                while True:
                    try:
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as err:
                        if time.perf_counter() - t0 > 10.0:
                            raise SnapshotInUseError(
                                f"Timed out waiting for snapshot lock '{snapshot_id}'."
                            ) from err
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    f.seek(0)
                    with suppress(OSError):
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()
            except Exception:
                pass

    def quarantine_snapshot(
        self,
        snapshot_id: str,
        reason: str,
        verification_errors: list[str] | None = None,
        expected_manifest_sha256: str | None = None,
        observed_manifest_sha256: str | None = None,
        destination_override: Path | None = None,
    ) -> Path:
        """Safely relocates an invalid or corrupted snapshot into the quarantine store.

        Preserves all original contents for forensic investigation, writes an audit quarantine record,
        and strictly refuses to move or delete snapshots with active registered readers or locked files.

        Args:
            snapshot_id: Identifier of the snapshot to quarantine.
            reason: High-level reason for quarantine (e.g. tampering, manifest mismatch).
            verification_errors: Detailed list of verification error strings.
            expected_manifest_sha256: Authoritative expected manifest digest if available.
            observed_manifest_sha256: Observed/computed manifest digest if available.
            destination_override: Optional specific destination (primarily for collision testing).

        Returns:
            Path to the quarantined directory.

        Raises:
            SnapshotInUseError: If the snapshot has active registered readers or locked file handles.
            QuarantineCollisionError: If the quarantine destination already exists.
            SnapshotIntegrityError: If snapshot ID is invalid or directory does not exist.
        """
        target_dir = self._validate_snapshot_id(snapshot_id)
        if not target_dir.exists() or not target_dir.is_dir():
            raise SnapshotIntegrityError(
                f"Cannot quarantine snapshot '{snapshot_id}': directory does not exist.",
                details={"snapshot_id": snapshot_id, "path": str(target_dir)},
            )

        # 1. Active reader check - fail closed rather than moving or deleting
        if self.has_active_readers(snapshot_id):
            active_count = self.active_readers_count(snapshot_id)
            raise SnapshotInUseError(
                f"Cannot quarantine snapshot '{snapshot_id}': snapshot has {active_count} active readers registered. "
                "Failing closed to preserve read consistency.",
                details={"snapshot_id": snapshot_id, "active_readers": active_count},
            )

        # 2. Extract observed manifest digest if not already provided
        if observed_manifest_sha256 is None:
            manifest_file = target_dir / "manifest.json"
            if manifest_file.exists() and manifest_file.is_file():
                try:
                    with manifest_file.open("rb") as mf:
                        observed_manifest_sha256 = hashlib.sha256(mf.read()).hexdigest()
                except Exception:
                    observed_manifest_sha256 = None

        # 3. Determine unique quarantine destination
        quarantine_base = self.config.quarantine_dir / "quarantined_snapshots"
        quarantine_base.mkdir(parents=True, exist_ok=True)

        if destination_override is not None:
            quarantine_dest = destination_override
        else:
            timestamp_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            unique_suffix = uuid.uuid4().hex[:8]
            quarantine_dest = quarantine_base / f"{snapshot_id}_{timestamp_str}_{unique_suffix}"

        # Requirement 5: Never overwrite an existing quarantine directory
        if quarantine_dest.exists():
            raise QuarantineCollisionError(
                f"Quarantine destination '{quarantine_dest}' already exists. Overwrite prohibited.",
                details={"snapshot_id": snapshot_id, "quarantine_dest": str(quarantine_dest)},
            )

        # 4. Attempt atomic move into quarantine destination
        try:
            target_dir.rename(quarantine_dest)
        except (PermissionError, OSError) as err:
            # If file is locked by an active process on Windows (WinError 32 or 5)
            # Fail closed: DO NOT force-remove or delete
            if isinstance(err, PermissionError) or getattr(err, "winerror", None) in (5, 32):
                raise SnapshotInUseError(
                    f"Cannot quarantine snapshot '{snapshot_id}': files are locked by an active reader/process: {err}",
                    details={"snapshot_id": snapshot_id, "error": str(err)},
                ) from err

            # If cross-device link (EXDEV), perform copytree then safe removal
            try:
                shutil.copytree(target_dir, quarantine_dest)
                _force_remove_tree(target_dir)
            except Exception as move_err:
                if quarantine_dest.exists():
                    _force_remove_tree(quarantine_dest)
                raise SnapshotIntegrityError(
                    f"Failed to relocate snapshot '{snapshot_id}' to quarantine: {move_err}",
                    details={"snapshot_id": snapshot_id, "error": str(move_err)},
                ) from move_err

        # 5. Write audit quarantine record inside the quarantined directory
        quarantine_record = {
            "quarantine_id": quarantine_dest.name,
            "snapshot_id": snapshot_id,
            "quarantined_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "verification_errors": verification_errors or [],
            "expected_manifest_sha256": expected_manifest_sha256,
            "observed_manifest_sha256": observed_manifest_sha256,
            "original_source_path": str(target_dir),
            "status": "QUARANTINED",
            "metadata_complete": False,
        }
        record_dest = quarantine_dest / "quarantine_record.json"
        try:
            os.chmod(quarantine_dest, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            quarantine_record["metadata_complete"] = True
            with record_dest.open("wb") as rf:
                rf.write(canonical_json_bytes(quarantine_record))
            os.chmod(record_dest, stat.S_IREAD)
        except Exception as meta_err:
            raise QuarantineMetadataError(
                f"Snapshot '{snapshot_id}' was successfully moved to quarantine at "
                f"'{quarantine_dest}', but the audit record (quarantine_record.json) could not "
                f"be written: {meta_err}. The quarantined contents are preserved for forensic "
                f"investigation. Manually create or restore the audit record to complete quarantine.",
                details={
                    "snapshot_id": snapshot_id,
                    "quarantine_dest": str(quarantine_dest),
                    "error": str(meta_err),
                },
            ) from meta_err

        return quarantine_dest

    def create_snapshot(
        self,
        extracted_root: Path,
        normalized: NormalizedDataset,
        asset_id: str | None = None,
        allow_existing: bool = True,
    ) -> tuple[AssetSnapshot, Path]:
        """Seals an extracted, normalized dataset directory into an immutable snapshot.

        Args:
            extracted_root: Path to the validated extracted dataset files.
            normalized: NormalizedDataset describing categories, samples, and metadata.
            asset_id: Logical asset identifier (e.g. 'cifar10_raw'). Generated if None.
            allow_existing: If True, existing valid snapshots matching the manifest digest
                are verified and returned idempotently. If False, raises SnapshotAlreadyExistsError.

        Returns:
            Tuple of (AssetSnapshot domain entity, Path to immutable snapshot root).

        Raises:
            SnapshotAlreadyExistsError: If a snapshot exists and allow_existing is False.
            SnapshotIntegrityError: On manifest generation or sealing failure.
        """
        resolved_src = extracted_root.resolve()
        if not resolved_src.exists() or not resolved_src.is_dir():
            raise SnapshotIntegrityError(
                f"Source directory does not exist: '{extracted_root}'.",
                details={"path": str(extracted_root)},
            )

        # 1. Collect and hash every regular file in the dataset directory
        manifest_entries: list[ManifestEntry] = []
        total_size = 0

        # Scan all regular files
        all_files = sorted(
            [p for p in resolved_src.rglob("*") if p.is_file() and not p.name.startswith(".")],
            key=lambda p: str(p.relative_to(resolved_src)).replace("\\", "/"),
        )

        for file_path in all_files:
            rel_str = str(file_path.relative_to(resolved_src)).replace("\\", "/")
            file_size = file_path.stat().st_size
            file_sha256 = self._compute_streaming_sha256(file_path)

            entry = ManifestEntry(
                relative_path=rel_str,
                sha256=file_sha256,
                size_bytes=file_size,
            )
            manifest_entries.append(entry)
            total_size += file_size

        if not manifest_entries:
            raise SnapshotIntegrityError(
                f"Cannot create snapshot: zero files found in '{extracted_root}'.",
                details={"path": str(extracted_root)},
            )

        # 2. Construct canonical manifest data structure (sorted by relative_path)
        manifest_entries.sort(key=lambda e: e.relative_path)
        manifest_files_dict = [
            {
                "relative_path": e.relative_path,
                "sha256": e.sha256,
                "size_bytes": e.size_bytes,
            }
            for e in manifest_entries
        ]

        manifest_data: dict[str, Any] = {
            "dataset_format": normalized.original_format,
            "files": manifest_files_dict,
            "manifest_version": "1.0.0",
            "normalization_version": normalized.normalization_version,
            "total_bytes": total_size,
            "total_files": len(manifest_entries),
        }

        # 3. Compute deterministic manifest SHA-256 digest
        manifest_bytes = canonical_json_bytes(manifest_data)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        snapshot_id = f"snap_{manifest_digest[:16]}"
        assigned_asset_id = asset_id or f"asset_{snapshot_id}"

        target_dir = self.config.snapshots_dir / snapshot_id

        # Wrap existing check, quarantine, and staging promotion in snapshot concurrency lock
        with self._acquire_snapshot_lock(snapshot_id):
            # 4. Check for existing snapshot / retry-safe verification
            if target_dir.exists():
                if not allow_existing:
                    raise SnapshotAlreadyExistsError(
                        f"Snapshot with ID '{snapshot_id}' already exists in snapshot store.",
                        details={"snapshot_id": snapshot_id, "path": str(target_dir)},
                    )

                verif = self.verify_snapshot(snapshot_id, expected_manifest_sha256=manifest_digest)
                if verif.is_valid:
                    # Existing snapshot is intact and matches the canonical manifest digest.
                    # Safe to reuse idempotently on job retry.
                    now = datetime.now(UTC)
                    snapshot_entity = AssetSnapshot(
                        snapshot_id=snapshot_id,
                        asset_id=assigned_asset_id,
                        asset_type=AssetType.DATASET_SNAPSHOT,
                        manifest_sha256=manifest_digest,
                        created_at=now,
                        file_count=len(manifest_entries),
                        total_size_bytes=total_size,
                        schema_version="1.0.0",
                        metadata={
                            "format": normalized.original_format,
                            "categories": normalized.categories,
                            "manifest_path": "manifest.json",
                        },
                    )
                    return snapshot_entity, target_dir

                # Snapshot directory exists but failed verification (partial/corrupted from previous crash).
                # Relocate to quarantine store to preserve evidence, instead of destructively deleting.
                all_defects = verif.all_defects
                try:
                    self.quarantine_snapshot(
                        snapshot_id=snapshot_id,
                        reason=f"Snapshot verification failed on recreation: {'; '.join(all_defects) if all_defects else 'verification failed'}",
                        verification_errors=all_defects,
                        expected_manifest_sha256=manifest_digest,
                    )
                except QuarantineMetadataError as qme:
                    # Quarantine contents are preserved. Audit record write failed.
                    # Snapshot creation continues so recovery is not blocked.
                    import sys

                    print(
                        f"WARNING: Quarantine audit record write failed for snapshot '{snapshot_id}': {qme}. "
                        "Contents are preserved in quarantine.",
                        file=sys.stderr,
                    )

            # 5. Staging directory with unique random suffix for atomic promotion
            staging_dir = (
                self.config.snapshots_dir / f".staging_{snapshot_id}_{uuid.uuid4().hex[:8]}"
            )
            if staging_dir.exists():
                _force_remove_tree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Copy all files to staging directory
                for entry in manifest_entries:
                    src_file = resolved_src / Path(entry.relative_path)
                    dst_file = staging_dir / Path(entry.relative_path)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)

                # Write canonical manifest.json
                manifest_dest = staging_dir / "manifest.json"
                with manifest_dest.open("wb") as mf:
                    mf.write(manifest_bytes)

                # Write normalized metadata.json
                metadata_dest = staging_dir / "metadata.json"
                metadata_bytes = canonical_json_bytes(
                    {
                        "categories": normalized.categories,
                        "metadata": normalized.metadata,
                        "normalization_version": normalized.normalization_version,
                        "original_format": normalized.original_format,
                        "snapshot_id": snapshot_id,
                        "validation_errors": normalized.validation_errors,
                        "validation_warnings": normalized.validation_warnings,
                    }
                )
                with metadata_dest.open("wb") as mdf:
                    mdf.write(metadata_bytes)

                # Make all files in staging read-only
                self._set_read_only(staging_dir)

                # Atomic rename from staging to final snapshot target
                staging_dir.rename(target_dir)

                # 6. Instantiate and return Slice 1A AssetSnapshot domain entity
                now = datetime.now(UTC)
                snapshot_entity = AssetSnapshot(
                    snapshot_id=snapshot_id,
                    asset_id=assigned_asset_id,
                    asset_type=AssetType.DATASET_SNAPSHOT,
                    manifest_sha256=manifest_digest,
                    created_at=now,
                    file_count=len(manifest_entries),
                    total_size_bytes=total_size,
                    schema_version="1.0.0",
                    metadata={
                        "format": normalized.original_format,
                        "categories": normalized.categories,
                        "manifest_path": "manifest.json",
                    },
                )

                return snapshot_entity, target_dir

            except SnapshotAlreadyExistsError:
                if staging_dir.exists():
                    _force_remove_tree(staging_dir)
                raise
            except Exception as err:
                if staging_dir.exists():
                    _force_remove_tree(staging_dir)
                raise SnapshotIntegrityError(
                    f"Failed to seal snapshot '{snapshot_id}': {err}",
                    details={"snapshot_id": snapshot_id, "error": str(err)},
                ) from err

    def verify_snapshot(
        self,
        snapshot_id: str,
        expected_manifest_sha256: str | None = None,
    ) -> SnapshotVerificationResult:
        """Verifies that every file in an immutable snapshot matches its manifest digest.

        Verifies the manifest against an authoritative expected SHA-256 digest, checks
        snapshot ID prefix consistency, detects modified/missing/extra files, and flags
        manifest tampering as a root-level error.

        Args:
            snapshot_id: Identifier of the snapshot to verify.
            expected_manifest_sha256: Authoritative SHA-256 digest of the manifest
                (from the stored AssetSnapshot record). If None, verification fails closed.

        Returns:
            SnapshotVerificationResult detailing verification status and itemized defects.
        """
        errors: list[str] = []
        try:
            snapshot_dir = self._validate_snapshot_id(snapshot_id)
        except Exception as err:
            return SnapshotVerificationResult(
                is_valid=False,
                snapshot_id=snapshot_id,
                manifest_sha256="",
                checked_files=0,
                errors=[f"Snapshot ID validation failed: {err}"],
            )

        # 1. Authoritative expected digest check (fail closed if missing)
        if expected_manifest_sha256 is None or not expected_manifest_sha256.strip():
            errors.append(
                "Authoritative expected manifest SHA-256 digest was not provided; "
                "snapshot verification cannot proceed without an authoritative anchor."
            )
        else:
            expected_manifest_sha256 = expected_manifest_sha256.strip().lower()
            if not snapshot_id.startswith(f"snap_{expected_manifest_sha256[:16]}"):
                errors.append(
                    f"Snapshot ID '{snapshot_id}' does not match prefix of expected "
                    f"manifest digest '{expected_manifest_sha256[:16]}'."
                )

        if not snapshot_dir.exists() or not snapshot_dir.is_dir():
            errors.append(f"Snapshot directory '{snapshot_id}' does not exist.")
            return SnapshotVerificationResult(
                is_valid=False,
                snapshot_id=snapshot_id,
                manifest_sha256="",
                checked_files=0,
                errors=errors,
            )

        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            errors.append("Manifest file 'manifest.json' is missing from snapshot root.")
            return SnapshotVerificationResult(
                is_valid=False,
                snapshot_id=snapshot_id,
                manifest_sha256="",
                checked_files=0,
                errors=errors,
            )

        try:
            with manifest_path.open("rb") as f:
                manifest_bytes = f.read()
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        except Exception as err:
            errors.append(f"Failed to read manifest.json: {err}")
            return SnapshotVerificationResult(
                is_valid=False,
                snapshot_id=snapshot_id,
                manifest_sha256="",
                checked_files=0,
                errors=errors,
            )

        # 2. Check computed manifest digest against snapshot_id prefix and expected anchor
        if not snapshot_id.startswith(f"snap_{manifest_digest[:16]}"):
            errors.append(
                f"Snapshot ID '{snapshot_id}' is inconsistent with computed manifest "
                f"digest prefix '{manifest_digest[:16]}'."
            )

        if expected_manifest_sha256 and manifest_digest != expected_manifest_sha256:
            errors.append(
                f"Root manifest digest mismatch: computed '{manifest_digest}' != "
                f"expected '{expected_manifest_sha256}'. Manifest tampering detected."
            )

        try:
            manifest_data = json.loads(manifest_bytes)
        except Exception as err:
            errors.append(f"Failed to parse manifest.json: {err}")
            return SnapshotVerificationResult(
                is_valid=False,
                snapshot_id=snapshot_id,
                manifest_sha256=manifest_digest,
                checked_files=0,
                errors=errors,
            )

        manifest_files = manifest_data.get("files", [])
        expected_files: dict[str, str] = {f["relative_path"]: f["sha256"] for f in manifest_files}

        missing: list[str] = []
        modified: list[str] = []
        checked = 0

        # 3. Check all expected files against manifest entries
        for rel_path, expected_hash in expected_files.items():
            checked += 1
            file_path = snapshot_dir / Path(rel_path)
            if not file_path.exists() or not file_path.is_file():
                missing.append(rel_path)
                continue

            actual_hash = self._compute_streaming_sha256(file_path)
            if actual_hash != expected_hash:
                modified.append(rel_path)

        # 4. Check for unexpected extra files in snapshot (excluding metadata.json and manifest.json)
        extra: list[str] = []
        for p in snapshot_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(snapshot_dir)).replace("\\", "/")
                if rel not in expected_files and rel not in ("manifest.json", "metadata.json"):
                    extra.append(rel)

        is_valid = len(errors) == 0 and len(missing) == 0 and len(modified) == 0 and len(extra) == 0

        return SnapshotVerificationResult(
            is_valid=is_valid,
            snapshot_id=snapshot_id,
            manifest_sha256=manifest_digest,
            checked_files=checked,
            missing_files=missing,
            modified_files=modified,
            extra_files=extra,
            errors=errors,
        )

    def _compute_streaming_sha256(self, file_path: Path) -> str:
        """Calculates SHA-256 for a file using streaming reads."""
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(self.config.chunk_size_bytes)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _set_read_only(target_dir: Path) -> None:
        """Marks all files and directories in target_dir read-only."""
        for p in target_dir.rglob("*"):
            if p.is_file():
                # Read-only permission: owner/group/other read
                p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
