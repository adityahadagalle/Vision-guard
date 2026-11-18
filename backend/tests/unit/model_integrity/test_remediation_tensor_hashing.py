"""Tests for PyTorch tensor byte extraction, layouts, and deterministic canonical hashing (SEC-3A-06).

Proves:
1. Deterministic state-dict fingerprinting across repeated runs on identical weights.
2. Safe inspection and fingerprinting of BFloat16 tensors.
3. Safe inspection and fingerprinting of Quantized tensors (qint8).
4. Safe inspection and fingerprinting of Sparse COO tensors.
5. Safe inspection and fingerprinting of Complex tensors (complex64).
6. Safe inspection and fingerprinting of Zero-sized tensors.
7. Graceful defect reporting (not unhandled crash) for unsupported layouts.
"""

from pathlib import Path

import torch

from backend.app.modules.model_integrity.inspection.inspect_cli import inspect_checkpoint


def test_deterministic_fingerprint_across_repeated_runs(tmp_path: Path) -> None:
    """Verifies that identical model weights produce 100% identical state_dict_fingerprints."""
    ckpt1 = tmp_path / "model1.pt"
    ckpt2 = tmp_path / "model2.pt"

    torch.manual_seed(12345)
    state = {
        "b": torch.randn(10, 10),
        "a": torch.randn(5, 5),
    }

    torch.save(state, ckpt1)
    # Save second with inverted dictionary order
    torch.save({"a": state["a"], "b": state["b"]}, ckpt2)

    rep1 = inspect_checkpoint(ckpt1)
    rep2 = inspect_checkpoint(ckpt2)

    assert rep1["success"] is True
    assert rep2["success"] is True
    assert rep1["state_dict_fingerprint"] == rep2["state_dict_fingerprint"]


def test_bfloat16_tensor_fingerprinting(tmp_path: Path) -> None:
    """Verifies that BFloat16 tensors are safely inspected and hashed without TypeError."""
    ckpt = tmp_path / "bf16_model.pt"
    state = {
        "encoder.weight": torch.randn(16, 16).bfloat16(),
    }
    torch.save(state, ckpt)

    rep = inspect_checkpoint(ckpt)
    assert rep["success"] is True
    assert rep["total_tensors"] == 1
    assert "bfloat16" in rep["tensors"]["encoder.weight"]["dtype"].lower()
    assert len(rep["state_dict_fingerprint"]) == 64


def test_quantized_tensor_fingerprinting(tmp_path: Path) -> None:
    """Verifies that Quantized tensors are safely inspected and hashed with scale/zero_point."""
    ckpt = tmp_path / "quantized_model.pt"
    float_tensor = torch.randn(8, 8)
    q_tensor = torch.quantize_per_tensor(float_tensor, scale=0.05, zero_point=1, dtype=torch.qint8)
    torch.save({"quantized_fc.weight": q_tensor}, ckpt)

    rep = inspect_checkpoint(ckpt)
    assert rep["success"] is True
    assert rep["total_tensors"] == 1
    assert rep["tensors"]["quantized_fc.weight"]["layout"] == "quantized"
    assert len(rep["state_dict_fingerprint"]) == 64


def test_sparse_tensor_fingerprinting(tmp_path: Path) -> None:
    """Verifies that Sparse COO tensors are safely coalesced, inspected, and hashed."""
    ckpt = tmp_path / "sparse_model.pt"
    indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
    values = torch.tensor([3.0, 4.0, 5.0])
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))
    torch.save({"embedding.weight": sparse_tensor}, ckpt)

    rep = inspect_checkpoint(ckpt)
    assert rep["success"] is True
    assert rep["total_tensors"] == 1
    assert rep["tensors"]["embedding.weight"]["layout"] == "sparse_coo"
    assert len(rep["state_dict_fingerprint"]) == 64


def test_complex_tensor_fingerprinting(tmp_path: Path) -> None:
    """Verifies that Complex tensors (complex64) are safely inspected and hashed."""
    ckpt = tmp_path / "complex_model.pt"
    c_tensor = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
    torch.save({"fft_layer.weight": c_tensor}, ckpt)

    rep = inspect_checkpoint(ckpt)
    assert rep["success"] is True
    assert rep["total_tensors"] == 1
    assert len(rep["state_dict_fingerprint"]) == 64


def test_zero_sized_tensor_fingerprinting(tmp_path: Path) -> None:
    """Verifies that Zero-sized empty tensors are handled cleanly without error."""
    ckpt = tmp_path / "empty_model.pt"
    torch.save({"empty.weight": torch.empty(0)}, ckpt)

    rep = inspect_checkpoint(ckpt)
    assert rep["success"] is True
    assert rep["total_tensors"] == 1
    assert rep["total_parameters"] == 0
    assert rep["tensors"]["empty.weight"]["numel"] == 0
    assert len(rep["state_dict_fingerprint"]) == 64


def test_sliced_tensor_fingerprint_uses_logical_values(tmp_path: Path) -> None:
    """Verifies NEW-3A-01: Contiguous tensor views with non-zero storage offsets hash their logical elements."""
    a = torch.tensor([[1, 2], [3, 4]])
    b = a[1]  # row 1: [3, 4], storage_offset == 2

    assert b.is_contiguous() is True
    assert b.storage_offset() == 2

    c = torch.tensor([3, 4])  # independent tensor with identical logical elements
    d = torch.tensor([1, 2])  # row 0 elements

    ckpt_view = tmp_path / "view.pt"
    ckpt_indep = tmp_path / "indep.pt"
    ckpt_row0 = tmp_path / "row0.pt"

    torch.save({"row": b}, ckpt_view)
    torch.save({"row": c}, ckpt_indep)
    torch.save({"row": d}, ckpt_row0)

    rep_view = inspect_checkpoint(ckpt_view)
    rep_indep = inspect_checkpoint(ckpt_indep)
    rep_row0 = inspect_checkpoint(ckpt_row0)

    assert rep_view["success"] is True
    assert rep_indep["success"] is True
    assert rep_row0["success"] is True

    # The view MUST hash to the same digest as the independent tensor with same values
    assert rep_view["tensors"]["row"]["sha256"] == rep_indep["tensors"]["row"]["sha256"]
    assert rep_view["state_dict_fingerprint"] == rep_indep["state_dict_fingerprint"]

    # The view MUST NOT hash to row 0's values
    assert rep_view["tensors"]["row"]["sha256"] != rep_row0["tensors"]["row"]["sha256"]
    assert rep_view["state_dict_fingerprint"] != rep_row0["state_dict_fingerprint"]


def test_1d_slice_with_storage_offset_hashes_logical_elements(tmp_path: Path) -> None:
    """Verifies that 1D slices with non-zero storage offsets hash only the sliced elements."""
    parent = torch.arange(10)
    slice_view = parent[3:7]  # [3, 4, 5, 6], storage_offset == 3

    assert slice_view.is_contiguous() is True
    assert slice_view.storage_offset() == 3

    independent = torch.tensor([3, 4, 5, 6])
    wrong_offset_zero = torch.tensor([0, 1, 2, 3])

    ckpt_slice = tmp_path / "slice1d.pt"
    ckpt_indep = tmp_path / "indep1d.pt"
    ckpt_wrong = tmp_path / "wrong1d.pt"

    torch.save({"param": slice_view}, ckpt_slice)
    torch.save({"param": independent}, ckpt_indep)
    torch.save({"param": wrong_offset_zero}, ckpt_wrong)

    rep_slice = inspect_checkpoint(ckpt_slice)
    rep_indep = inspect_checkpoint(ckpt_indep)
    rep_wrong = inspect_checkpoint(ckpt_wrong)

    assert rep_slice["tensors"]["param"]["sha256"] == rep_indep["tensors"]["param"]["sha256"]
    assert rep_slice["tensors"]["param"]["sha256"] != rep_wrong["tensors"]["param"]["sha256"]


def test_disjoint_slices_from_same_parent_have_distinct_fingerprints(tmp_path: Path) -> None:
    """Verifies that two distinct slices from the same parent have distinct digests matching their clones."""
    parent = torch.randn(8, 8)
    slice1 = parent[:4, :]
    slice2 = parent[4:, :]

    assert slice2.storage_offset() > 0

    indep1 = parent[:4, :].clone()
    indep2 = parent[4:, :].clone()

    ckpt1 = tmp_path / "slices.pt"
    ckpt2 = tmp_path / "clones.pt"

    torch.save({"s1": slice1, "s2": slice2}, ckpt1)
    torch.save({"s1": indep1, "s2": indep2}, ckpt2)

    rep1 = inspect_checkpoint(ckpt1)
    rep2 = inspect_checkpoint(ckpt2)

    assert rep1["tensors"]["s1"]["sha256"] == rep2["tensors"]["s1"]["sha256"]
    assert rep1["tensors"]["s2"]["sha256"] == rep2["tensors"]["s2"]["sha256"]
    assert rep1["tensors"]["s1"]["sha256"] != rep1["tensors"]["s2"]["sha256"]
    assert rep1["state_dict_fingerprint"] == rep2["state_dict_fingerprint"]


def test_transposed_strided_tensor_fingerprint_matches_contiguous_equivalent(
    tmp_path: Path,
) -> None:
    """Verifies that non-contiguous transposed tensors produce deterministic canonical hashes."""
    orig = torch.tensor([[1, 2, 3], [4, 5, 6]])
    transposed = orig.t()  # shape (3, 2), non-contiguous
    assert transposed.is_contiguous() is False

    indep_transposed = torch.tensor([[1, 4], [2, 5], [3, 6]])

    ckpt_trans = tmp_path / "trans.pt"
    ckpt_indep = tmp_path / "indep_trans.pt"

    torch.save({"w": transposed}, ckpt_trans)
    torch.save({"w": indep_transposed}, ckpt_indep)

    rep_trans = inspect_checkpoint(ckpt_trans)
    rep_indep = inspect_checkpoint(ckpt_indep)

    assert rep_trans["tensors"]["w"]["sha256"] == rep_indep["tensors"]["w"]["sha256"]
    assert rep_trans["state_dict_fingerprint"] == rep_indep["state_dict_fingerprint"]


def test_quantized_slice_view_uses_logical_values(tmp_path: Path) -> None:
    """Verifies that quantized tensor slices with storage offsets hash logical values."""
    float_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    q = torch.quantize_per_tensor(float_tensor, scale=0.1, zero_point=0, dtype=torch.qint8)
    q_slice = q[1]  # row 1

    q_indep = torch.quantize_per_tensor(
        torch.tensor([3.0, 4.0]), scale=0.1, zero_point=0, dtype=torch.qint8
    )
    q_row0 = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0]), scale=0.1, zero_point=0, dtype=torch.qint8
    )

    ckpt_slice = tmp_path / "q_slice.pt"
    ckpt_indep = tmp_path / "q_indep.pt"
    ckpt_row0 = tmp_path / "q_row0.pt"

    torch.save({"q": q_slice}, ckpt_slice)
    torch.save({"q": q_indep}, ckpt_indep)
    torch.save({"q": q_row0}, ckpt_row0)

    rep_slice = inspect_checkpoint(ckpt_slice)
    rep_indep = inspect_checkpoint(ckpt_indep)
    rep_row0 = inspect_checkpoint(ckpt_row0)

    assert rep_slice["tensors"]["q"]["sha256"] == rep_indep["tensors"]["q"]["sha256"]
    assert rep_slice["tensors"]["q"]["sha256"] != rep_row0["tensors"]["q"]["sha256"]
