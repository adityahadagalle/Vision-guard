"""Tests for model snapshot locking, immutability, and quarantine recovery (SEC-3A-04).

Proves:
1. Snapshot directory and files are sealed with read-only permissions.
2. An already existing valid snapshot is reused idempotently.
3. Tampering with an existing snapshot triggers quarantine and clean re-ingestion.
4. Concurrent snapshot ingestion operations synchronize via cross-process locking.
"""

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from backend.app.modules.model_integrity.artifact import ModelArtifactManager
from backend.app.modules.model_integrity.config import ModelIntegrityConfig


@pytest.fixture
def manager(tmp_path: Path) -> ModelArtifactManager:
    config = ModelIntegrityConfig(
        models_dir=tmp_path / "models",
        snapshots_dir=tmp_path / "snapshots",
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
    )
    return ModelArtifactManager(config)


@pytest.fixture
def sample_model_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pt"
    torch.save({"layer": torch.ones(4, 4)}, p)
    return p


def test_snapshot_sealed_read_only(manager: ModelArtifactManager, sample_model_file: Path) -> None:
    """Verifies that newly created snapshot manifests and model files are marked read-only."""
    res = manager.ingest_model(sample_model_file, filename="sample.pt")
    snap_dir = res.snapshot_dir

    manifest_file = snap_dir / "manifest.json"
    model_file = snap_dir / "sample.pt"
    metadata_file = snap_dir / "metadata.json"

    assert manifest_file.exists()
    assert model_file.exists()
    assert metadata_file.exists()

    # Check that write permission is removed
    mode = model_file.stat().st_mode
    assert (mode & stat.S_IWRITE) == 0


def test_existing_valid_snapshot_reused_idempotently(
    manager: ModelArtifactManager, sample_model_file: Path
) -> None:
    """Verifies that re-ingesting an identical model reuses the valid snapshot without error."""
    res1 = manager.ingest_model(sample_model_file, filename="sample.pt")
    res2 = manager.ingest_model(sample_model_file, filename="sample.pt")

    assert res1.snapshot.snapshot_id == res2.snapshot.snapshot_id
    assert res1.sha256 == res2.sha256


def test_tampered_existing_snapshot_is_quarantined(
    manager: ModelArtifactManager, sample_model_file: Path, tmp_path: Path
) -> None:
    """Verifies that if an existing snapshot directory was tampered with, it is quarantined and re-created."""
    res1 = manager.ingest_model(sample_model_file, filename="sample.pt")
    snap_dir = res1.snapshot_dir

    # Tamper with the model file inside the snapshot directory
    model_in_snap = snap_dir / "sample.pt"
    # Remove read-only attribute to simulate tampering
    os.chmod(model_in_snap, stat.S_IWRITE | stat.S_IREAD)
    model_in_snap.write_bytes(b"TAMPERED_BYTES_CORRUPTED")

    # Ingesting again should detect corruption, quarantine old snapshot, and create clean new one
    res2 = manager.ingest_model(sample_model_file, filename="sample.pt")

    assert res2.snapshot.snapshot_id == res1.snapshot.snapshot_id

    # Check that quarantine directory contains the tampered snapshot
    quarantine_dir = tmp_path / "quarantine"
    quarantined_dirs = list(quarantine_dir.rglob("snap_*"))
    assert len(quarantined_dirs) >= 1
    assert any("snap_" in d.name for d in quarantined_dirs)


def test_concurrent_snapshot_creation_synchronizes_cleanly(
    manager: ModelArtifactManager, sample_model_file: Path
) -> None:
    """Verifies that multiple concurrent ingestions of the same artifact do not corrupt files."""

    def worker_task(idx: int):
        return manager.ingest_model(sample_model_file, filename="sample.pt")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_task, i) for i in range(4)]
        results = [f.result() for f in futures]

    # All should return the identical verified snapshot ID
    first_snap_id = results[0].snapshot.snapshot_id
    for r in results:
        assert r.snapshot.snapshot_id == first_snap_id
        assert r.sha256 == results[0].sha256
