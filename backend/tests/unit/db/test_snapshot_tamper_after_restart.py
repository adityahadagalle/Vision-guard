"""Security regression test verifying snapshot manifest tampering detection after restart.

Verifies:
1. Snapshot creation persists the authoritative manifest_sha256 to SQLite.
2. Complete teardown and recreation of the database session/engine simulates an application restart.
3. Reloading the snapshot record retrieves the authoritative manifest digest from SQLite.
4. If an adversary modifies a file AND updates manifest.json on disk to match the new file hash,
   verification using the reloaded authoritative digest detects root-level manifest tampering.
5. Snapshot verification fails closed and rejects the tampered snapshot.
"""

import hashlib
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base
from backend.app.db.repositories.dataset_repository import SqliteDatasetRepository
from backend.app.db.session import _set_sqlite_pragmas
from backend.app.domain.entities.enums import IngestionStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.snapshot import (
    SnapshotManager,
    canonical_json_bytes,
)


def test_tampered_snapshot_detected_after_application_restart(tmp_path: Path):
    """End-to-end regression test: file + manifest modification detected across restart."""
    db_file = tmp_path / "persistence_restart.db"
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{db_file.as_posix()}"

    # =========================================================================
    # Phase 1: Initial Application Session - Create Snapshot and Persist
    # =========================================================================
    engine1 = create_engine(db_url, echo=False)
    event.listen(engine1, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(bind=engine1)
    session_factory1 = sessionmaker(bind=engine1)
    session1 = session_factory1()

    dataset_repo1 = SqliteDatasetRepository(session1)
    dataset_repo1.create_dataset(
        dataset_id="ds_target_01",
        name="Target Dataset",
        dataset_format="FOLDER_PER_CLASS",
        storage_path="/tmp/target.zip",
        sha256="0" * 64,
        size_bytes=5000,
        status=IngestionStatus.ACCEPTED,
    )

    # 1. Compute canonical manifest first
    manifest_data = {
        "dataset_format": "FOLDER_PER_CLASS",
        "files": [
            {
                "relative_path": "images/0001.jpg",
                "sha256": hashlib.sha256(b"AUTHENTIC_IMAGE_DATA_1").hexdigest(),
                "size_bytes": len(b"AUTHENTIC_IMAGE_DATA_1"),
            },
            {
                "relative_path": "images/0002.jpg",
                "sha256": hashlib.sha256(b"AUTHENTIC_IMAGE_DATA_2").hexdigest(),
                "size_bytes": len(b"AUTHENTIC_IMAGE_DATA_2"),
            },
        ],
        "manifest_version": "1.0.0",
        "normalization_version": "1.0.0",
        "total_bytes": len(b"AUTHENTIC_IMAGE_DATA_1") + len(b"AUTHENTIC_IMAGE_DATA_2"),
        "total_files": 2,
    }
    manifest_bytes = canonical_json_bytes(manifest_data)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot_id = f"snap_{manifest_sha256[:16]}"

    # Build physical snapshot directory with files
    snap_dir = snapshots_dir / snapshot_id
    images_dir = snap_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    img1 = images_dir / "0001.jpg"
    img1.write_bytes(b"AUTHENTIC_IMAGE_DATA_1")

    img2 = images_dir / "0002.jpg"
    img2.write_bytes(b"AUTHENTIC_IMAGE_DATA_2")

    manifest_file = snap_dir / "manifest.json"
    manifest_file.write_bytes(manifest_bytes)

    # Persist snapshot entity in SQLite
    snapshot_entity = dataset_repo1.create_snapshot(
        snapshot_id=snapshot_id,
        dataset_id="ds_target_01",
        manifest_sha256=manifest_sha256,
        storage_path=snap_dir.as_posix(),
        file_count=2,
        total_size_bytes=len(b"AUTHENTIC_IMAGE_DATA_1") + len(b"AUTHENTIC_IMAGE_DATA_2"),
    )
    assert snapshot_entity.manifest_sha256 == manifest_sha256

    # =========================================================================
    # Phase 2: Application Shutdown / Teardown
    # =========================================================================
    session1.close()
    engine1.dispose()

    # =========================================================================
    # Phase 3: Adversary modifies a snapshot file AND recalculates manifest.json
    # =========================================================================
    # Tamper with image file
    img1.write_bytes(b"MALICIOUS_POISON_DATA_1")
    tampered_img1_sha = hashlib.sha256(b"MALICIOUS_POISON_DATA_1").hexdigest()

    # Adversary recalculates manifest.json so internal hashes would look clean
    # if someone only checked manifest.json against the filesystem!
    tampered_manifest_data = {
        "dataset_format": "FOLDER_PER_CLASS",
        "files": [
            {
                "relative_path": "images/0001.jpg",
                "sha256": tampered_img1_sha,
                "size_bytes": len(b"MALICIOUS_POISON_DATA_1"),
            },
            {
                "relative_path": "images/0002.jpg",
                "sha256": hashlib.sha256(b"AUTHENTIC_IMAGE_DATA_2").hexdigest(),
                "size_bytes": len(b"AUTHENTIC_IMAGE_DATA_2"),
            },
        ],
        "manifest_version": "1.0.0",
        "normalization_version": "1.0.0",
        "total_bytes": len(b"MALICIOUS_POISON_DATA_1") + len(b"AUTHENTIC_IMAGE_DATA_2"),
        "total_files": 2,
    }
    tampered_manifest_bytes = canonical_json_bytes(tampered_manifest_data)
    manifest_file.write_bytes(tampered_manifest_bytes)

    # =========================================================================
    # Phase 4: Application Restart - Reload from SQLite and Verify
    # =========================================================================
    # Start fresh engine and session simulating a new application boot
    engine2 = create_engine(db_url, echo=False)
    event.listen(engine2, "connect", _set_sqlite_pragmas)
    session_factory2 = sessionmaker(bind=engine2)
    session2 = session_factory2()

    dataset_repo2 = SqliteDatasetRepository(session2)

    # Reload snapshot record and retrieve the authoritative digest from SQLite
    reloaded_snapshot = dataset_repo2.get_snapshot(snapshot_id)
    assert reloaded_snapshot is not None
    authoritative_manifest_sha256 = reloaded_snapshot.manifest_sha256
    assert authoritative_manifest_sha256 == manifest_sha256  # Matches original uncorrupted hash

    # Now verify the snapshot on disk using SnapshotManager
    ingestion_cfg = IngestionConfig(snapshots_dir=snapshots_dir)
    mgr = SnapshotManager(config=ingestion_cfg)
    result = mgr.verify_snapshot(
        snapshot_id=snapshot_id,
        expected_manifest_sha256=authoritative_manifest_sha256,
    )

    # Must detect tampering and fail closed!
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any(
        "Root manifest digest mismatch" in err or "tamper" in err.lower() for err in result.errors
    )

    session2.close()
    engine2.dispose()
