"""Reproducible benchmark runner for ADR-003: Safetensors vs. PyTorch Checkpoints (.pt).

Measures:
1. Serialization size and SHA-256 computation speed.
2. Cold and warm loading latency (median and p95) for:
   - PyTorch .pt (weights_only=True)
   - PyTorch .pt (weights_only=False)
   - Safetensors (mmap zero-copy loading)
   - Safetensors (standard buffer loading)
3. Header-only metadata inspection latency:
   - Safetensors header extraction (reading 8-byte uint64 + JSON metadata without tensor bytes)
   - PyTorch .pt metadata inspection (ZIP Central Directory / state-dict unpickling)
4. Peak process working set memory (RSS) on Windows / POSIX.

Zero network access. Uses existing packages in .venv: torch, numpy, standard library.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import mmap
import platform
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from ctypes import wintypes
from pathlib import Path
from typing import Any

import numpy as np
import torch

# -----------------------------------------------------------------------------
# OS & Hardware Profiling Utilities
# -----------------------------------------------------------------------------


def get_system_metadata() -> dict[str, Any]:
    """Collects host machine hardware and operating system parameters."""
    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
        avail_ram_gb = round(stat.ullAvailPhys / (1024**3), 2)

    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
    }


def get_process_rss_bytes() -> int:
    """Returns current process Resident Set Size (RSS) working set in bytes."""
    if sys.platform == "win32":

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if fn(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize)
        return 0
    else:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# -----------------------------------------------------------------------------
# Safetensors Pure Binary Format Implementation (Zero External Dependencies)
# -----------------------------------------------------------------------------

TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

SAFETENSORS_TO_NUMPY_DTYPE = {
    "F32": np.float32,
    "F64": np.float64,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def save_safetensors(
    tensors: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Serializes a state dictionary to exact Safetensors binary format."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    data_offset = 0
    tensor_bytes_list: list[bytes] = []

    for name, tensor in tensors.items():
        t_contig = tensor.contiguous().cpu()
        np_arr = t_contig.numpy()
        raw_b = np_arr.tobytes()
        length = len(raw_b)

        st_dtype = TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype)
        if not st_dtype:
            raise ValueError(f"Unsupported dtype for Safetensors: {tensor.dtype}")

        header[name] = {
            "dtype": st_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [data_offset, data_offset + length],
        }
        data_offset += length
        tensor_bytes_list.append(raw_b)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for chunk in tensor_bytes_list:
            f.write(chunk)


def load_safetensors_header(
    path: Path, max_header_bytes: int = 100_000_000
) -> tuple[dict[str, Any], int]:
    """Reads ONLY the 8-byte length prefix + JSON header. Zero tensor bytes read."""
    with open(path, "rb") as f:
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise ValueError("File too short for Safetensors header")
        header_len = struct.unpack("<Q", len_bytes)[0]
        if header_len > max_header_bytes:
            raise ValueError(
                f"Header length {header_len} exceeds maximum allowed {max_header_bytes} bytes"
            )
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError(
                f"Unexpected EOF: expected {header_len} header bytes, got {len(header_bytes)}"
            )
        header_json = header_bytes.decode("utf-8")
        header = json.loads(header_json)
        return header, 8 + header_len


def load_safetensors_mmap(path: Path) -> dict[str, torch.Tensor]:
    """Loads tensors via memory-mapping (zero-copy OS page views)."""
    with open(path, "rb") as f:
        fileno = f.fileno()
        mm = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)

    header_len = struct.unpack("<Q", mm[:8])[0]
    header = json.loads(mm[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len

    loaded: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start_off, end_off = info["data_offsets"]
        np_dtype = SAFETENSORS_TO_NUMPY_DTYPE[info["dtype"]]
        shape = info["shape"]

        buf_slice = mm[data_start + start_off : data_start + end_off]
        np_arr = np.frombuffer(buf_slice, dtype=np_dtype).reshape(shape)
        loaded[name] = torch.from_numpy(np_arr)

    return loaded


def load_safetensors_buffer(path: Path) -> dict[str, torch.Tensor]:
    """Loads tensors by reading the file into a memory buffer and constructing tensors."""
    with open(path, "rb") as f:
        data = f.read()

    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len

    loaded: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start_off, end_off = info["data_offsets"]
        np_dtype = SAFETENSORS_TO_NUMPY_DTYPE[info["dtype"]]
        shape = info["shape"]

        buf_slice = data[data_start + start_off : data_start + end_off]
        np_arr = np.frombuffer(buf_slice, dtype=np_dtype).reshape(shape)
        loaded[name] = torch.from_numpy(np_arr.copy())

    return loaded


# -----------------------------------------------------------------------------
# Synthetic Checkpoint Generators
# -----------------------------------------------------------------------------


def generate_synthetic_state_dict(num_params: int) -> dict[str, torch.Tensor]:
    """Generates a synthetic state dictionary resembling a convolutional neural network."""
    state_dict: dict[str, torch.Tensor] = {}
    remaining = num_params
    layer_idx = 0

    while remaining > 0:
        if remaining > 200_000:
            out_c = min(256, max(64, int((remaining / 9) ** 0.5)))
            in_c = out_c
            k_size = 3
            p_count = out_c * in_c * k_size * k_size
            if p_count > remaining:
                out_c = max(1, remaining // (in_c * k_size * k_size))
                p_count = out_c * in_c * k_size * k_size
            shape = (out_c, in_c, k_size, k_size)
        elif remaining > 50_000:
            out_f = 512
            in_f = min(remaining // out_f, 1024)
            p_count = out_f * in_f
            shape = (out_f, in_f)
        else:
            p_count = remaining
            shape = (p_count,)

        if p_count == 0:
            break

        state_dict[f"layer_{layer_idx}.weight"] = torch.randn(shape, dtype=torch.float32)
        remaining -= p_count
        layer_idx += 1

    return state_dict


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------


def compute_sha256(path: Path) -> tuple[str, float]:
    """Streams file through SHA-256 and measures throughput."""
    t0 = time.perf_counter()
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    dt = time.perf_counter() - t0
    return hasher.hexdigest(), dt


def run_benchmark_for_size(
    name: str,
    target_params: int,
    work_dir: Path,
    iterations: int = 5,
) -> dict[str, Any]:
    """Executes a full comparative benchmark run for a specific checkpoint size."""
    print(
        f"\n[{name.upper()}] Generating synthetic model (~{target_params:,} float32 parameters)..."
    )
    state_dict = generate_synthetic_state_dict(target_params)
    actual_params = sum(t.numel() for t in state_dict.values())
    raw_tensor_mb = (actual_params * 4) / (1024 * 1024)
    print(
        f"[{name.upper()}] Actual parameter count: {actual_params:,} ({raw_tensor_mb:.2f} MB raw tensor data)"
    )

    pt_path = work_dir / f"{name}_checkpoint.pt"
    st_path = work_dir / f"{name}_model.safetensors"

    # 1. Save .pt
    t0 = time.perf_counter()
    torch.save(state_dict, pt_path)
    pt_save_time = time.perf_counter() - t0
    pt_size_mb = pt_path.stat().st_size / (1024 * 1024)

    # 2. Save .safetensors
    t0 = time.perf_counter()
    save_safetensors(state_dict, st_path, metadata={"format": "safetensors", "model_size": name})
    st_save_time = time.perf_counter() - t0
    st_size_mb = st_path.stat().st_size / (1024 * 1024)

    # 3. SHA-256 Hashing Benchmark
    _, pt_hash_time = compute_sha256(pt_path)
    _, st_hash_time = compute_sha256(st_path)

    # 4. Latency & Memory: PyTorch weights_only=True
    pt_wo_times = []
    gc.collect()
    rss_before = get_process_rss_bytes()
    for _i in range(iterations):
        t0 = time.perf_counter()
        loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
        pt_wo_times.append(time.perf_counter() - t0)
        del loaded
    gc.collect()
    rss_peak_pt_wo = get_process_rss_bytes()

    # 5. Latency & Memory: PyTorch weights_only=False (legacy unpickler)
    pt_legacy_times = []
    gc.collect()
    for _i in range(iterations):
        t0 = time.perf_counter()
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        pt_legacy_times.append(time.perf_counter() - t0)
        del loaded
    gc.collect()

    # 6. Latency & Memory: Safetensors MMAP (zero-copy)
    st_mmap_times = []
    gc.collect()
    rss_before_mmap = get_process_rss_bytes()
    for _i in range(iterations):
        t0 = time.perf_counter()
        loaded = load_safetensors_mmap(st_path)
        st_mmap_times.append(time.perf_counter() - t0)
        del loaded
    gc.collect()
    rss_peak_st_mmap = get_process_rss_bytes()

    # 7. Latency & Memory: Safetensors standard buffer load
    st_buf_times = []
    gc.collect()
    for _i in range(iterations):
        t0 = time.perf_counter()
        loaded = load_safetensors_buffer(st_path)
        st_buf_times.append(time.perf_counter() - t0)
        del loaded
    gc.collect()

    # 8. Header-Only Metadata Inspection Latency
    st_hdr_times = []
    for _ in range(iterations * 10):
        t0 = time.perf_counter()
        _, _ = load_safetensors_header(st_path)
        st_hdr_times.append(time.perf_counter() - t0)

    pt_hdr_times = []
    for _ in range(iterations * 5):
        t0 = time.perf_counter()
        with zipfile.ZipFile(pt_path, "r") as zf:
            namelist = zf.namelist()
            zf.read(f"{namelist[0].split('/')[0]}/data.pkl")
        pt_hdr_times.append(time.perf_counter() - t0)

    def stats(arr: list[float]) -> dict[str, float]:
        arr_sorted = sorted(arr)
        median = arr_sorted[len(arr_sorted) // 2]
        p95_idx = int(len(arr_sorted) * 0.95)
        p95 = arr_sorted[min(p95_idx, len(arr_sorted) - 1)]
        return {"median_ms": median * 1000, "p95_ms": p95 * 1000, "cold_ms": arr[0] * 1000}

    return {
        "tier": name,
        "actual_params": actual_params,
        "raw_tensor_mb": raw_tensor_mb,
        "pt_size_mb": pt_size_mb,
        "st_size_mb": st_size_mb,
        "pt_save_s": pt_save_time,
        "st_save_s": st_save_time,
        "pt_hash_s": pt_hash_time,
        "st_hash_s": st_hash_time,
        "pt_wo_stats": stats(pt_wo_times),
        "pt_legacy_stats": stats(pt_legacy_times),
        "st_mmap_stats": stats(st_mmap_times),
        "st_buf_stats": stats(st_buf_times),
        "st_hdr_stats": stats(st_hdr_times),
        "pt_hdr_stats": stats(pt_hdr_times),
        "rss_delta_pt_mb": max(0.0, (rss_peak_pt_wo - rss_before) / (1024 * 1024)),
        "rss_delta_st_mmap_mb": max(0.0, (rss_peak_st_mmap - rss_before_mmap) / (1024 * 1024)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADR-003 Safetensors vs PyTorch Empirical Benchmark"
    )
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per test")
    parser.add_argument(
        "--keep-artifacts", action="store_true", help="Retain benchmark checkpoint files"
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["small", "medium", "large"],
        help="Tiers to run (small, medium, large)",
    )
    args = parser.parse_args()

    sys_meta = get_system_metadata()
    print("=" * 80)
    print("VISIONGUARD ADR-003: SAFETENSORS VS. PYTORCH CHECKPOINT BENCHMARK")
    print("=" * 80)
    print(f"OS: {sys_meta['os']}")
    print(f"Processor: {sys_meta['processor']}")
    print(f"RAM: {sys_meta['total_ram_gb']} GB total ({sys_meta['avail_ram_gb']} GB available)")
    print(
        f"Python: {sys_meta['python_version']} | PyTorch: {sys_meta['torch_version']} | NumPy: {sys_meta['numpy_version']}"
    )
    print(f"Iterations per measurement: {args.iterations}")
    print("=" * 80)

    tier_params = {
        "small": 2_500_000,  # ~10 MB float32
        "medium": 25_000_000,  # ~100 MB float32
        "large": 105_000_000,  # ~420 MB float32 (matches VisionGuard's 500 MB upload ceiling)
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="vg_adr003_bench_"))
    results = []

    try:
        for tier in args.tiers:
            if tier not in tier_params:
                print(f"Skipping unknown tier '{tier}'")
                continue
            res = run_benchmark_for_size(
                tier, tier_params[tier], temp_dir, iterations=args.iterations
            )
            results.append(res)

        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK RESULTS (RAW DATA)")
        print("=" * 80)

        # Print Markdown Table 1: Load Latency
        print("\n### Table 1: Checkpoint Load Latency Comparison (Cold & Median p50 / p95)\n")
        print(
            "| Checkpoint Tier | Raw Size | Format | Cold Load (ms) | Median Load (ms) | p95 Load (ms) |"
        )
        print("|---|---|---|---|---|---|")
        for r in results:
            raw_sz = f"{r['raw_tensor_mb']:.1f} MB"
            print(
                f"| {r['tier'].capitalize()} | {raw_sz} | PyTorch `weights_only=True` | {r['pt_wo_stats']['cold_ms']:.2f} | {r['pt_wo_stats']['median_ms']:.2f} | {r['pt_wo_stats']['p95_ms']:.2f} |"
            )
            print(
                f"| {r['tier'].capitalize()} | {raw_sz} | PyTorch `weights_only=False` | {r['pt_legacy_stats']['cold_ms']:.2f} | {r['pt_legacy_stats']['median_ms']:.2f} | {r['pt_legacy_stats']['p95_ms']:.2f} |"
            )
            print(
                f"| {r['tier'].capitalize()} | {raw_sz} | Safetensors `mmap` (zero-copy) | {r['st_mmap_stats']['cold_ms']:.2f} | {r['st_mmap_stats']['median_ms']:.2f} | {r['st_mmap_stats']['p95_ms']:.2f} |"
            )
            print(
                f"| {r['tier'].capitalize()} | {raw_sz} | Safetensors Standard Buffer | {r['st_buf_stats']['cold_ms']:.2f} | {r['st_buf_stats']['median_ms']:.2f} | {r['st_buf_stats']['p95_ms']:.2f} |"
            )

        # Print Markdown Table 2: Header-Only Metadata Inspection Latency
        print(
            "\n### Table 2: Header-Only Metadata Inspection Latency (Reading Shapes/Dtypes/Parameters)\n"
        )
        print(
            "| Checkpoint Tier | File Size (PT / ST) | PyTorch Metadata Zip/Pkl (ms) | Safetensors Header Read (ms) | Speedup Factor |"
        )
        print("|---|---|---|---|---|")
        for r in results:
            pt_sz = f"{r['pt_size_mb']:.1f} MB"
            st_sz = f"{r['st_size_mb']:.1f} MB"
            pt_hdr = r["pt_hdr_stats"]["median_ms"]
            st_hdr = r["st_hdr_stats"]["median_ms"]
            speedup = pt_hdr / max(0.0001, st_hdr)
            print(
                f"| {r['tier'].capitalize()} | {pt_sz} / {st_sz} | {pt_hdr:.3f} ms | {st_hdr:.3f} ms | **{speedup:.1f}x faster** |"
            )

        # Print Markdown Table 3: Serialization & Storage Characteristics
        print("\n### Table 3: File Size, Serialization & SHA-256 Hashing Throughput\n")
        print(
            "| Checkpoint Tier | Parameters | PyTorch Size | Safetensors Size | PT Hash (s) | ST Hash (s) |"
        )
        print("|---|---|---|---|---|---|")
        for r in results:
            print(
                f"| {r['tier'].capitalize()} | {r['actual_params']:,} | {r['pt_size_mb']:.2f} MB | {r['st_size_mb']:.2f} MB | {r['pt_hash_s']:.3f} s | {r['st_hash_s']:.3f} s |"
            )

        # Save JSON output for reproducibility
        out_json = Path("reports") / "adr003_benchmark_results.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"system": sys_meta, "results": results}, f, indent=2)
        print(f"\nBenchmark raw results serialized to: {out_json}")

    finally:
        if not args.keep_artifacts:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temporary benchmark artifacts cleaned up ({temp_dir}).")
        else:
            print(f"Benchmark artifacts retained at: {temp_dir}")


if __name__ == "__main__":
    main()
