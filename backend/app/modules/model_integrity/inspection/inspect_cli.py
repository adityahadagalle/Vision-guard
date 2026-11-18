"""Isolated PyTorch checkpoint inspection CLI.

Invoked inside an isolated subprocess worker. Hardened execution:
1. Strictly invokes torch.load(..., weights_only=True, map_location="cpu").
2. Weights_only=False is strictly prohibited and never called.
3. Deserializes strictly in CPU memory.
4. Traverses root dictionary and state-dict tensors.
5. Computes tensor metadata, NaN/Inf counts, and deterministic state-dict fingerprint.
6. Emits structured JSON to stdout.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


def inspect_checkpoint(
    checkpoint_path: Path,
    max_parameters: int = 200_000_000,
    max_tensor_memory_bytes: int = 2 * 1024 * 1024 * 1024,
    max_tensors: int = 5000,
) -> dict[str, Any]:
    """Safely inspects a PyTorch checkpoint file under weights_only=True.

    Returns:
        Structured dictionary containing inspection results and tensor metadata.
    """
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        return {
            "success": False,
            "error": f"Checkpoint file '{checkpoint_path}' does not exist.",
            "error_type": "FileNotFoundError",
        }

    # MANDATORY SECURITY INVARIANT: weights_only=True is strictly enforced.
    # No fallback to weights_only=False is permitted under any condition.
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to load checkpoint safely: {exc}",
            "error_type": type(exc).__name__,
        }

    # Verify root structure
    if not isinstance(loaded, dict):
        return {
            "success": False,
            "error": f"Checkpoint root object must be a dict, got {type(loaded).__name__}.",
            "error_type": "UnsupportedCheckpointFormatError",
        }

    # Locate the state dictionary
    candidate_state_dict: dict[Any, Any]
    format_type = "raw_state_dict"

    if all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in loaded.values()) and loaded:
        candidate_state_dict = loaded
        format_type = "raw_state_dict"
    elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        candidate_state_dict = loaded["state_dict"]
        format_type = "nested_state_dict"
    elif "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
        candidate_state_dict = loaded["model_state_dict"]
        format_type = "nested_model_state_dict"
    elif "model" in loaded and isinstance(loaded["model"], dict):
        candidate_state_dict = loaded["model"]
        format_type = "nested_model"
    else:
        # Check if dict contains tensors
        tensor_keys = [
            k for k, v in loaded.items() if isinstance(v, (torch.Tensor, torch.nn.Parameter))
        ]
        if tensor_keys:
            candidate_state_dict = {k: loaded[k] for k in tensor_keys}
            format_type = "partial_state_dict"
        else:
            return {
                "success": False,
                "error": "No valid state dictionary or tensor mappings found in checkpoint.",
                "error_type": "UnsupportedCheckpointFormatError",
            }

    # Validate state-dict keys and values
    non_str_keys: list[str] = []
    non_tensor_entries: list[str] = []
    tensor_records: dict[str, dict[str, Any]] = {}
    total_params = 0
    total_memory_bytes = 0
    total_nan = 0
    total_inf = 0
    affected_non_finite_tensors: list[str] = []

    defects: list[str] = []
    fingerprint_lines: list[str] = []

    for key, value in candidate_state_dict.items():
        if not isinstance(key, str):
            non_str_keys.append(str(key))
            continue

        if not isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            non_tensor_entries.append(key)
            continue

        tensor = value
        # Determine layout and extract canonical tensor digest
        try:
            if tensor.is_sparse:
                s = tensor.coalesce()
                indices = s.indices().detach().cpu().clone().contiguous()
                values = s.values().detach().cpu().clone().contiguous()
                idx_bytes = (
                    bytes(indices.untyped_storage()[: indices.numel() * indices.element_size()])
                    if indices.numel() > 0
                    else b""
                )
                val_bytes = (
                    bytes(values.untyped_storage()[: values.numel() * values.element_size()])
                    if values.numel() > 0
                    else b""
                )
                shape = list(s.shape)
                dtype_str = str(s.dtype)
                numel = s._nnz()
                elem_size = values.element_size()
                tensor_bytes = len(idx_bytes) + len(val_bytes)
                layout_str = "sparse_coo"

                hasher = hashlib.sha256()
                hasher.update(b"sparse_coo:")
                hasher.update(str(shape).encode("utf-8") + b":")
                hasher.update(dtype_str.encode("utf-8") + b":")
                hasher.update(idx_bytes)
                hasher.update(b":")
                hasher.update(val_bytes)
                tensor_sha256 = hasher.hexdigest()

                nan_count = (
                    int(torch.isnan(values).sum().item())
                    if values.is_floating_point() or values.is_complex()
                    else 0
                )
                inf_count = (
                    int(torch.isinf(values).sum().item())
                    if values.is_floating_point() or values.is_complex()
                    else 0
                )

            elif tensor.is_quantized:
                qscheme_str = str(tensor.qscheme())
                int_repr = tensor.int_repr().detach().cpu().clone().contiguous()
                nb = int_repr.numel() * int_repr.element_size()
                int_bytes = bytes(int_repr.untyped_storage()[:nb]) if nb > 0 else b""
                shape = list(tensor.shape)
                dtype_str = str(tensor.dtype)
                numel = tensor.numel()
                elem_size = tensor.element_size()
                tensor_bytes = nb
                layout_str = "quantized"

                hasher = hashlib.sha256()
                hasher.update(b"quantized:")
                hasher.update(qscheme_str.encode("utf-8") + b":")
                hasher.update(str(shape).encode("utf-8") + b":")
                if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                    hasher.update(
                        f"{float(tensor.q_scale())}:{int(tensor.q_zero_point())}:".encode()
                    )
                elif hasattr(tensor, "q_per_channel_scales"):
                    scales = [float(x) for x in tensor.q_per_channel_scales().tolist()[:10]]
                    zero_points = [int(x) for x in tensor.q_per_channel_zero_points().tolist()[:10]]
                    hasher.update(
                        f"{scales}:{zero_points}:{int(tensor.q_per_channel_axis())}:".encode()
                    )
                hasher.update(int_bytes)
                tensor_sha256 = hasher.hexdigest()

                nan_count = 0
                inf_count = 0

            else:
                if tensor.layout != torch.strided:
                    defects.append(
                        f"Unsupported tensor layout '{tensor.layout}' in tensor '{key}'."
                    )
                    continue

                t = tensor.detach().cpu().clone().contiguous()
                shape = list(t.shape)
                dtype_str = str(t.dtype)
                numel = t.numel()
                elem_size = t.element_size()
                tensor_bytes = numel * elem_size
                layout_str = "strided"

                raw_bytes = bytes(t.untyped_storage()[:tensor_bytes]) if tensor_bytes > 0 else b""
                tensor_sha256 = hashlib.sha256(raw_bytes).hexdigest()

                is_float = t.is_floating_point() or t.is_complex()
                nan_count = int(torch.isnan(t).sum().item()) if is_float else 0
                inf_count = int(torch.isinf(t).sum().item()) if is_float else 0

        except Exception as exc:
            defects.append(f"Failed to process tensor '{key}': {exc}")
            continue

        total_params += numel
        total_memory_bytes += tensor_bytes

        if nan_count > 0 or inf_count > 0:
            total_nan += nan_count
            total_inf += inf_count
            affected_non_finite_tensors.append(key)

        tensor_records[key] = {
            "shape": shape,
            "dtype": dtype_str,
            "layout": layout_str,
            "numel": numel,
            "size_bytes": tensor_bytes,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "sha256": tensor_sha256,
        }

    # Deterministic canonical state_dict_fingerprint
    for key in sorted(tensor_records.keys()):
        rec = tensor_records[key]
        line = f"{key}:{rec['dtype']}:{rec['shape']}:{rec.get('layout', 'strided')}:{rec['sha256']}"
        fingerprint_lines.append(line)

    canonical_repr = "\n".join(fingerprint_lines).encode("utf-8")
    state_dict_fingerprint = hashlib.sha256(canonical_repr).hexdigest()

    # Structural defect checks
    defects = list(defects)
    if non_str_keys:
        defects.append(f"Non-string keys found in state dict: {non_str_keys[:5]}")
    if non_tensor_entries:
        defects.append(f"Non-tensor values found in state dict: {non_tensor_entries[:5]}")
    if len(tensor_records) > max_tensors:
        defects.append(f"Tensor count {len(tensor_records)} exceeds limit of {max_tensors}.")
    if total_params > max_parameters:
        defects.append(f"Total parameter count {total_params} exceeds limit of {max_parameters}.")
    if total_memory_bytes > max_tensor_memory_bytes:
        defects.append(
            f"Total tensor memory {total_memory_bytes} bytes exceeds limit of {max_tensor_memory_bytes} bytes."
        )

    return {
        "success": True,
        "format": format_type,
        "weights_only_verified": True,
        "state_dict_fingerprint": state_dict_fingerprint,
        "total_tensors": len(tensor_records),
        "total_parameters": total_params,
        "total_memory_bytes": total_memory_bytes,
        "defects": defects,
        "non_finite_summary": {
            "has_nan": total_nan > 0,
            "has_inf": total_inf > 0,
            "total_nan": total_nan,
            "total_inf": total_inf,
            "affected_tensors": affected_non_finite_tensors,
        },
        "tensors": tensor_records,
    }


def main() -> None:
    """CLI entrypoint for isolated worker checkpoint inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoint inside isolated worker."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to candidate checkpoint file.")
    parser.add_argument("--max-parameters", type=int, default=200_000_000)
    parser.add_argument("--max-tensor-memory-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--max-tensors", type=int, default=5000)

    args = parser.parse_args()
    report = inspect_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        max_parameters=args.max_parameters,
        max_tensor_memory_bytes=args.max_tensor_memory_bytes,
        max_tensors=args.max_tensors,
    )

    print(json.dumps(report))
    if not report.get("success", False):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
