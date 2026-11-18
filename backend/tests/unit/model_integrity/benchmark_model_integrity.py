"""Benchmark suite for PyTorch model ingestion and isolated inspection performance."""

import time
from pathlib import Path

import torch
import torchvision.models as models

from backend.app.modules.model_integrity.architecture.registry import (
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.artifact import ModelArtifactManager
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.inspection.worker import SubprocessModelWorker


def run_benchmarks(tmp_path: Path) -> dict[str, float]:
    """Runs performance benchmarks on realistic model artifacts."""
    print("=" * 60)
    print("VisionGuard Phase 3, Slice 3A — Performance Benchmarks")
    print("=" * 60)

    # 1. Generate realistic ResNet-18 checkpoint (~45 MB)
    print("\n[1/3] Generating realistic ResNet-18 checkpoint (~44.7 MB)...")
    t0 = time.perf_counter()
    resnet = models.resnet18(weights=None)
    model_path = tmp_path / "resnet18_benchmark.pt"
    torch.save(resnet.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Generated {model_size_mb:.2f} MB in {time.perf_counter() - t0:.3f}s")

    # 2. Benchmark streaming ingestion & SHA-256 throughput
    print("\n[2/3] Benchmarking streaming artifact ingestion...")
    config = ModelIntegrityConfig(
        models_dir=tmp_path / "bench_models",
        snapshots_dir=tmp_path / "bench_snapshots",
    )
    mgr = ModelArtifactManager(config)

    t0 = time.perf_counter()
    result = mgr.ingest_model(model_path, filename="resnet18_benchmark.pt")
    ingest_time = time.perf_counter() - t0
    throughput_mb_s = model_size_mb / ingest_time
    print(f"Ingested {model_size_mb:.2f} MB in {ingest_time:.3f}s ({throughput_mb_s:.2f} MB/s)")
    print(f"Computed SHA-256: {result.sha256[:16]}...")
    print(f"Snapshot created: {result.snapshot.snapshot_id}")

    # 3. Benchmark isolated subprocess checkpoint inspection
    print("\n[3/3] Benchmarking isolated subprocess checkpoint inspection...")
    worker = SubprocessModelWorker(config)
    t0 = time.perf_counter()
    report = worker.inspect(result.storage_path)
    inspect_time = time.perf_counter() - t0
    print(f"Subprocess inspection completed in {inspect_time:.3f}s")
    print(f"Tensors extracted: {report.total_tensors}")
    print(f"Total parameters: {report.total_parameters:,}")
    print(f"State-dict fingerprint: {report.state_dict_fingerprint[:16]}...")
    print(f"Finite weights verified: {not report.has_nan and not report.has_inf}")

    # 4. Architecture compatibility check
    print("\n[4/4] Benchmarking architecture compatibility validation...")
    registry = TrustedArchitectureRegistry()
    t0 = time.perf_counter()
    compat = registry.validate_compatibility("resnet18", report.tensors)
    compat_time = time.perf_counter() - t0
    print(f"Compatibility evaluated in {compat_time * 1000:.2f} ms")
    print(f"Compatible: {compat.compatible}, Score: {compat.compatibility_score}")

    print("\n" + "=" * 60)
    print("Benchmark Summary:")
    print(f"  Artifact Size:           {model_size_mb:.2f} MB")
    print(f"  Ingestion Throughput:    {throughput_mb_s:.2f} MB/s (Time: {ingest_time:.3f}s)")
    print(f"  Subprocess Inspection:   {inspect_time:.3f}s")
    print(f"  Architecture Validation: {compat_time * 1000:.2f} ms")
    print("=" * 60)

    return {
        "model_size_mb": model_size_mb,
        "ingest_time_s": ingest_time,
        "throughput_mb_s": throughput_mb_s,
        "inspect_time_s": inspect_time,
        "compat_time_ms": compat_time * 1000,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        run_benchmarks(Path(td))
