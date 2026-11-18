"""Unit tests for ModelArtifactManager and secure PyTorch artifact ingestion."""

import hashlib
import io
from pathlib import Path

import pytest
import torch

from backend.app.domain.entities.enums import AssetType
from backend.app.modules.model_integrity.artifact import ModelArtifactManager
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.exceptions import (
    EmptyModelFileError,
    InvalidModelExtensionError,
    ModelDigestMismatchError,
    ModelFileTooLargeError,
)


@pytest.fixture
def temp_config(tmp_path: Path) -> ModelIntegrityConfig:
    return ModelIntegrityConfig(
        max_model_bytes=10 * 1024 * 1024,  # 10 MB for tests
        quarantine_dir=tmp_path / "quarantine",
        staging_dir=tmp_path / "staging",
        models_dir=tmp_path / "models",
        snapshots_dir=tmp_path / "snapshots",
    )


def test_ingest_valid_pt_file(temp_config: ModelIntegrityConfig, tmp_path: Path) -> None:
    """Verifies successful ingestion of a valid .pt PyTorch checkpoint."""
    mgr = ModelArtifactManager(temp_config)
    state_dict = {"fc.weight": torch.randn(5, 5), "fc.bias": torch.zeros(5)}
    model_src = tmp_path / "clean_model.pt"
    torch.save(state_dict, model_src)

    result = mgr.ingest_model(model_src, filename="clean_model.pt")

    assert result.filename == "clean_model.pt"
    assert len(result.sha256) == 64
    assert result.size_bytes > 0
    assert result.storage_path.exists()
    assert result.snapshot.asset_type == AssetType.MODEL_CHECKPOINT
    assert result.snapshot.manifest_sha256 is not None
    assert (result.snapshot_dir / "manifest.json").exists()


def test_ingest_valid_pth_file(temp_config: ModelIntegrityConfig, tmp_path: Path) -> None:
    """Verifies successful ingestion of a valid .pth checkpoint."""
    mgr = ModelArtifactManager(temp_config)
    state_dict = {"conv.weight": torch.randn(8, 3, 3, 3)}
    model_src = tmp_path / "model.pth"
    torch.save(state_dict, model_src)

    result = mgr.ingest_model(model_src, filename="model.pth")
    assert result.filename == "model.pth"
    assert result.storage_path.exists()


def test_reject_unsupported_extensions(temp_config: ModelIntegrityConfig) -> None:
    """Verifies that non-.pt/.pth extensions are strictly rejected."""
    mgr = ModelArtifactManager(temp_config)
    stream = io.BytesIO(b"fake model content")

    for invalid_ext in ["model.bin", "model.onnx", "model.h5", "model.pkl", "model.txt"]:
        with pytest.raises(InvalidModelExtensionError) as exc_info:
            mgr.ingest_model(stream, filename=invalid_ext)
        assert "Unsupported model file extension" in exc_info.value.message


def test_reject_empty_model_file(temp_config: ModelIntegrityConfig) -> None:
    """Verifies that 0-byte model files are rejected."""
    mgr = ModelArtifactManager(temp_config)
    stream = io.BytesIO(b"")

    with pytest.raises(EmptyModelFileError) as exc_info:
        mgr.ingest_model(stream, filename="empty.pt")
    assert "empty" in exc_info.value.message


def test_reject_oversized_model_file(temp_config: ModelIntegrityConfig) -> None:
    """Verifies that model files exceeding max_model_bytes are rejected."""
    tight_config = ModelIntegrityConfig(
        max_model_bytes=1000,
        quarantine_dir=temp_config.quarantine_dir,
        staging_dir=temp_config.staging_dir,
        models_dir=temp_config.models_dir,
        snapshots_dir=temp_config.snapshots_dir,
    )
    mgr = ModelArtifactManager(tight_config)
    stream = io.BytesIO(b"X" * 2000)

    with pytest.raises(ModelFileTooLargeError) as exc_info:
        mgr.ingest_model(stream, filename="big_model.pt")
    assert "exceeds maximum allowed size" in exc_info.value.message


def test_expected_sha256_verification_match(
    temp_config: ModelIntegrityConfig, tmp_path: Path
) -> None:
    """Verifies that ingestion succeeds when expected_sha256 matches observed."""
    mgr = ModelArtifactManager(temp_config)
    content = b"pytorch checkpoint bytes simulation"
    real_sha256 = hashlib.sha256(content).hexdigest()

    result = mgr.ingest_model(
        io.BytesIO(content),
        filename="checked_model.pt",
        expected_sha256=real_sha256,
    )
    assert result.sha256 == real_sha256


def test_expected_sha256_verification_mismatch(
    temp_config: ModelIntegrityConfig, tmp_path: Path
) -> None:
    """Verifies that ingestion raises ModelDigestMismatchError on hash mismatch."""
    mgr = ModelArtifactManager(temp_config)
    content = b"pytorch checkpoint bytes simulation"

    with pytest.raises(ModelDigestMismatchError) as exc_info:
        mgr.ingest_model(
            io.BytesIO(content),
            filename="checked_model.pt",
            expected_sha256="0" * 64,
        )
    assert "mismatch" in exc_info.value.message
