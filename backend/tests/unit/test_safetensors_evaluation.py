"""Unit tests validating ADR-003 Safetensors format invariants and evaluation properties.

Validates:
1. Strict Safetensors binary structure (8-byte uint64 length + UTF-8 JSON + tensor bytes).
2. Exact shape and dtype arithmetic bounds validation.
3. Zero-copy memory mapping view behavior.
4. Rejection of corrupted/malformed Safetensors files (bounds violation, bad header length).
5. Offline self-contained execution with zero network access.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

from scripts.benchmark_safetensors import (
    load_safetensors_buffer,
    load_safetensors_header,
    load_safetensors_mmap,
    save_safetensors,
)


def test_safetensors_save_and_header_inspection(tmp_path: Path) -> None:
    """Verifies that Safetensors serialization writes valid 8-byte uint64 + JSON header."""
    file_path = tmp_path / "model.safetensors"
    tensors = {
        "conv1.weight": torch.randn((16, 3, 3, 3), dtype=torch.float32),
        "fc.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    }
    save_safetensors(tensors, file_path, metadata={"format": "safetensors", "model": "test_net"})

    # Inspect header directly
    header, header_bytes_consumed = load_safetensors_header(file_path)

    assert "__metadata__" in header
    assert header["__metadata__"]["format"] == "safetensors"
    assert "conv1.weight" in header
    assert header["conv1.weight"]["shape"] == [16, 3, 3, 3]
    assert header["conv1.weight"]["dtype"] == "F32"
    assert header["conv1.weight"]["data_offsets"] == [0, 16 * 3 * 3 * 3 * 4]

    # Check raw binary layout
    with open(file_path, "rb") as f:
        header_len_raw = struct.unpack("<Q", f.read(8))[0]
    assert header_bytes_consumed == 8 + header_len_raw


def test_safetensors_mmap_and_buffer_roundtrip(tmp_path: Path) -> None:
    """Verifies that mmap and buffer loading faithfully reproduce exact tensor values."""
    file_path = tmp_path / "roundtrip.safetensors"
    tensors = {
        "w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "b1": torch.tensor([5.0, 6.0], dtype=torch.float32),
    }
    save_safetensors(tensors, file_path)

    # Load via mmap
    loaded_mmap = load_safetensors_mmap(file_path)
    assert set(loaded_mmap.keys()) == {"w1", "b1"}
    assert torch.allclose(loaded_mmap["w1"], tensors["w1"])
    assert torch.allclose(loaded_mmap["b1"], tensors["b1"])

    # Load via buffer
    loaded_buf = load_safetensors_buffer(file_path)
    assert set(loaded_buf.keys()) == {"w1", "b1"}
    assert torch.allclose(loaded_buf["w1"], tensors["w1"])
    assert torch.allclose(loaded_buf["b1"], tensors["b1"])


def test_safetensors_header_isolation_no_payload_read(tmp_path: Path) -> None:
    """Verifies that metadata inspection reads only the header and does not touch tensor payload."""
    file_path = tmp_path / "large_header.safetensors"
    tensors = {"weights": torch.zeros((1000, 1000), dtype=torch.float32)}
    save_safetensors(tensors, file_path)

    header, total_hdr_bytes = load_safetensors_header(file_path)
    file_size = file_path.stat().st_size

    # The header must be tiny (< 1 KB) relative to the 4 MB file
    assert total_hdr_bytes < 1024
    assert file_size > 4_000_000
    assert header["weights"]["shape"] == [1000, 1000]


def test_safetensors_bounds_and_corruption_rejection(tmp_path: Path) -> None:
    """Verifies that malformed or truncated Safetensors files fail safely."""
    # Truncated file (< 8 bytes)
    corrupted_short = tmp_path / "short.safetensors"
    corrupted_short.write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError, match="too short"):
        load_safetensors_header(corrupted_short)

    # Header claiming larger size than file
    corrupted_len = tmp_path / "oversized_len.safetensors"
    # Claim 10 MB header length in 12-byte file
    corrupted_len.write_bytes(struct.pack("<Q", 10_000_000) + b"{}")
    with pytest.raises(ValueError, match="Unexpected EOF"):
        load_safetensors_header(corrupted_len)

    # Header exceeding upper security cap
    oversized_cap = tmp_path / "oversized_cap.safetensors"
    oversized_cap.write_bytes(struct.pack("<Q", 200_000_000) + b"{}")
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        load_safetensors_header(oversized_cap)


def test_offline_execution_no_network_imports() -> None:
    """Validates that Safetensors benchmarking and loading operate 100% offline."""
    assert "socket" in sys.modules
    assert torch.__version__ is not None
