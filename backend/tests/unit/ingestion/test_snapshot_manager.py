"""Unit tests for SnapshotManager and immutable snapshot verification."""

import json
import stat
from pathlib import Path

import pytest

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    SnapshotAlreadyExistsError,
)
from backend.app.modules.data_integrity.ingestion.normalizer import NormalizedDataset
from backend.app.modules.data_integrity.ingestion.snapshot import (
    SnapshotManager,
    canonical_json_bytes,
)


@pytest.fixture
def snapshot_setup(tmp_path: Path) -> tuple[SnapshotManager, Path]:
    config = IngestionConfig(snapshots_dir=tmp_path / "snapshots")
    mgr = SnapshotManager(config)

    # Create dummy extracted dataset directory
    extracted = tmp_path / "extracted_data"
    extracted.mkdir()
    (extracted / "images").mkdir()
    (extracted / "images" / "b_img.jpg").write_bytes(b"content_b")
    (extracted / "images" / "a_img.jpg").write_bytes(b"content_a")
    (extracted / "labels.txt").write_text("dummy labels", encoding="utf-8")

    return mgr, extracted


def test_canonical_json_bytes_determinism() -> None:
    """Verifies that canonical_json_bytes sorts keys and uses compact separators."""
    d1 = {"b": 2, "a": 1, "nested": {"z": 9, "x": 8}}
    d2 = {"a": 1, "nested": {"x": 8, "z": 9}, "b": 2}

    assert canonical_json_bytes(d1) == canonical_json_bytes(d2)
    assert canonical_json_bytes(d1) == b'{"a":1,"b":2,"nested":{"x":8,"z":9}}'


def test_create_snapshot_manifest_sorting_and_hashing(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """Verifies that snapshot manifest entries are strictly sorted and hashed."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat", "dog"],
        samples=[],
    )

    snapshot_entity, snapshot_root = mgr.create_snapshot(
        extracted, normalized, asset_id="asset_cifar"
    )

    assert snapshot_entity.snapshot_id.startswith("snap_")
    assert snapshot_entity.file_count == 3
    assert snapshot_entity.total_size_bytes == len(b"content_b") + len(b"content_a") + len(
        b"dummy labels"
    )
    assert snapshot_root.exists()

    # Read sealed manifest.json
    manifest_path = snapshot_root / "manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Verify sorting: images/a_img.jpg, images/b_img.jpg, labels.txt
    rel_paths = [f["relative_path"] for f in manifest_data["files"]]
    assert rel_paths == sorted(rel_paths)
    assert rel_paths == ["images/a_img.jpg", "images/b_img.jpg", "labels.txt"]


def test_snapshot_overwrite_prevention(snapshot_setup: tuple[SnapshotManager, Path]) -> None:
    """Verifies that attempting to overwrite an existing snapshot raises SnapshotAlreadyExistsError when forbidden, and safely reuses it by default."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    # First creation succeeds
    s1, root1 = mgr.create_snapshot(extracted, normalized)

    # Second creation with identical files when allow_existing=False must fail
    with pytest.raises(SnapshotAlreadyExistsError, match="already exists"):
        mgr.create_snapshot(extracted, normalized, allow_existing=False)

    # Default retry behavior (allow_existing=True) verifies and returns the snapshot idempotently
    s2, root2 = mgr.create_snapshot(extracted, normalized, allow_existing=True)
    assert s2.snapshot_id == s1.snapshot_id
    assert root2 == root1


def test_snapshot_verification_tamper_detection(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """Verifies that snapshot verification detects modified, missing, and unexpected extra files."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    snapshot_entity, snapshot_root = mgr.create_snapshot(extracted, normalized)
    sid = snapshot_entity.snapshot_id

    # 1. Clean verification with authoritative expected manifest digest
    res_clean = mgr.verify_snapshot(sid, expected_manifest_sha256=snapshot_entity.manifest_sha256)
    assert res_clean.is_valid is True
    assert len(res_clean.errors) == 0
    assert len(res_clean.missing_files) == 0
    assert len(res_clean.modified_files) == 0
    assert len(res_clean.extra_files) == 0

    # 2. Modify one file (requires temporarily clearing read-only on Windows/POSIX)
    target_file = snapshot_root / "images" / "a_img.jpg"
    target_file.chmod(stat.S_IWRITE | stat.S_IREAD)
    target_file.write_bytes(b"tampered content replacing content_a")

    res_modified = mgr.verify_snapshot(
        sid, expected_manifest_sha256=snapshot_entity.manifest_sha256
    )
    assert res_modified.is_valid is False
    assert "images/a_img.jpg" in res_modified.modified_files

    # 3. Delete one file
    target_file.unlink()
    res_missing = mgr.verify_snapshot(sid, expected_manifest_sha256=snapshot_entity.manifest_sha256)
    assert res_missing.is_valid is False
    assert "images/a_img.jpg" in res_missing.missing_files

    # 4. Add an extra unexpected file
    extra_file = snapshot_root / "images" / "unauthorized.jpg"
    extra_file.write_bytes(b"unauthorized payload")
    res_extra = mgr.verify_snapshot(sid, expected_manifest_sha256=snapshot_entity.manifest_sha256)
    assert res_extra.is_valid is False
    assert "images/unauthorized.jpg" in res_extra.extra_files


def test_tamper_file_and_manifest_both_detected(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """The critical test: modifying both a file AND updating manifest.json must NOT produce is_valid=True."""
    import hashlib

    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    snapshot_entity, snapshot_root = mgr.create_snapshot(extracted, normalized)
    sid = snapshot_entity.snapshot_id
    authoritative_digest = snapshot_entity.manifest_sha256

    # 1. Tamper with a file
    target_file = snapshot_root / "images" / "a_img.jpg"
    target_file.chmod(stat.S_IWRITE | stat.S_IREAD)
    target_file.write_bytes(b"malicious tampered content")
    tampered_file_sha256 = hashlib.sha256(b"malicious tampered content").hexdigest()

    # 2. Also update manifest.json so that file hash would match if manifest were blindly trusted
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for file_entry in manifest_data["files"]:
        if file_entry["relative_path"] == "images/a_img.jpg":
            file_entry["sha256"] = tampered_file_sha256
            file_entry["size_bytes"] = len(b"malicious tampered content")

    manifest_path.write_bytes(canonical_json_bytes(manifest_data))

    # 3. Verify: Must fail and flag root manifest tampering!
    res = mgr.verify_snapshot(sid, expected_manifest_sha256=authoritative_digest)
    assert res.is_valid is False
    assert any("Manifest tampering detected" in err for err in res.errors)


def test_tamper_manifest_only_detected(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """Modifying manifest.json alone without changing files must trigger root tampering error."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    snapshot_entity, snapshot_root = mgr.create_snapshot(extracted, normalized)
    sid = snapshot_entity.snapshot_id
    authoritative_digest = snapshot_entity.manifest_sha256

    manifest_path = snapshot_root / "manifest.json"
    manifest_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["total_files"] = 999  # Tamper with metadata field
    manifest_path.write_bytes(canonical_json_bytes(manifest_data))

    res = mgr.verify_snapshot(sid, expected_manifest_sha256=authoritative_digest)
    assert res.is_valid is False
    assert any("Manifest tampering detected" in err for err in res.errors)


def test_missing_authoritative_digest_fails_closed(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """Verification must fail closed when expected_manifest_sha256 is None or empty."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    snapshot_entity, _ = mgr.create_snapshot(extracted, normalized)
    sid = snapshot_entity.snapshot_id

    # Omit expected_manifest_sha256 -> must fail closed
    res_none = mgr.verify_snapshot(sid, expected_manifest_sha256=None)
    assert res_none.is_valid is False
    assert any(
        "Authoritative expected manifest SHA-256 digest was not provided" in err
        for err in res_none.errors
    )

    # Empty string -> must fail closed
    res_empty = mgr.verify_snapshot(sid, expected_manifest_sha256="   ")
    assert res_empty.is_valid is False
    assert any(
        "Authoritative expected manifest SHA-256 digest was not provided" in err
        for err in res_empty.errors
    )


def test_malformed_manifest_json_fails_closed(
    snapshot_setup: tuple[SnapshotManager, Path],
) -> None:
    """A corrupted or malformed manifest.json must be caught with a clear error."""
    mgr, extracted = snapshot_setup
    normalized = NormalizedDataset(
        original_format="COCO_DETECTION",
        categories=["cat"],
        samples=[],
    )

    snapshot_entity, snapshot_root = mgr.create_snapshot(extracted, normalized)
    sid = snapshot_entity.snapshot_id
    authoritative_digest = snapshot_entity.manifest_sha256

    manifest_path = snapshot_root / "manifest.json"
    manifest_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    manifest_path.write_bytes(b"{not-valid-json-syntax")

    res = mgr.verify_snapshot(sid, expected_manifest_sha256=authoritative_digest)
    assert res.is_valid is False
    assert any(
        "Failed to parse manifest.json" in err or "Manifest tampering detected" in err
        for err in res.errors
    )
