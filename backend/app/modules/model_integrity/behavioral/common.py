"""Common evaluation utilities for Model Integrity behavioral detectors."""

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.domain.ports.detector import AssessmentContext, DetectorResult
from backend.app.modules.model_integrity.architecture.registry import (
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.behavioral.worker import (
    BehavioralReport,
    SubprocessBehavioralWorker,
)


def compute_behavioral_cache_identity(
    candidate_sha256: str,
    reference_sha256: str,
    architecture_id: str,
    assessment_id: str,
    job_id: str,
    lease_generation: int,
    probe_count: int,
    probe_batch_size: int,
    probe_seed: int,
    input_shape: list[int],
    sensitivity_epsilon: float,
    sensitivity_flip_threshold: float,
    consistency_repetitions: int,
    backdoor_trigger_size: int,
    backdoor_concentration_threshold: float,
    backdoor_confidence_threshold: float,
    backdoor_delta_threshold: float,
    reference_agreement_threshold: float,
) -> tuple[str, dict[str, Any]]:
    """Derives deterministic canonical cache key and metadata covering complete assessment context."""
    metadata = {
        "architecture_id": architecture_id,
        "assessment_id": assessment_id,
        "backdoor_concentration_threshold": backdoor_concentration_threshold,
        "backdoor_confidence_threshold": backdoor_confidence_threshold,
        "backdoor_delta_threshold": backdoor_delta_threshold,
        "backdoor_trigger_size": backdoor_trigger_size,
        "candidate_sha256": candidate_sha256,
        "consistency_repetitions": consistency_repetitions,
        "input_shape": list(input_shape),
        "job_id": job_id,
        "lease_generation": lease_generation,
        "probe_batch_size": probe_batch_size,
        "probe_count": probe_count,
        "probe_seed": probe_seed,
        "reference_agreement_threshold": reference_agreement_threshold,
        "reference_sha256": reference_sha256,
        "sensitivity_epsilon": sensitivity_epsilon,
        "sensitivity_flip_threshold": sensitivity_flip_threshold,
    }
    canonical_repr = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(canonical_repr).hexdigest()
    return cache_key, metadata


def seed_behavioral_cache_entry(
    context: AssessmentContext,
    report: BehavioralReport,
    candidate_model_file: Path | None = None,
    reference_model_file: Path | None = None,
    reference_sha256: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Seeds the compound behavioral cache store in context with a verified report entry.

    Ensures that mock reports injected in tests conform to the exact compound-keyed
    isolation identity and metadata validation schema enforced by get_or_run_behavioral_report.
    """
    model_file: Path | None = candidate_model_file
    if model_file is None:
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()
        target_path_str = context.snapshot.metadata.get("storage_path")
        if target_path_str and Path(target_path_str).exists():
            model_file = Path(target_path_str).resolve()
        elif snapshot_dir.exists():
            for p in snapshot_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".pt", ".pth"):
                    model_file = p
                    break

    if not model_file or not model_file.exists():
        raise ValueError("Cannot seed behavioral cache: model file not found in context.")

    ref_checkpoint: Path | None = reference_model_file
    if ref_checkpoint is None:
        ref_path_str = context.config.get("reference_model_path") or context.snapshot.metadata.get(
            "reference_model_path"
        )
        if ref_path_str and Path(ref_path_str).exists():
            ref_checkpoint = Path(ref_path_str).resolve()

    candidate_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if ref_checkpoint and ref_checkpoint.exists():
        ref_digest = reference_sha256 or hashlib.sha256(ref_checkpoint.read_bytes()).hexdigest()
    else:
        ref_digest = reference_sha256 or "NO_REFERENCE"

    architecture_id = (
        context.config.get("architecture_id")
        or context.snapshot.metadata.get("architecture_id")
        or "simple_cnn"
    )
    registry = TrustedArchitectureRegistry()
    input_shape = (
        registry.get_default_input_shape(architecture_id)
        if registry.has(architecture_id)
        else [3, 32, 32]
    )

    assessment_id = str(context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}"))
    job_id = str(context.config.get("job_id", "none"))
    lease_generation = int(context.config.get("lease_generation", 0) or 0)
    probe_count = int(context.config.get("probe_count", context.config.get("probes_count", 30)))
    probe_batch_size = int(context.config.get("probe_batch_size", 10))
    probe_seed = int(context.config.get("probe_seed", context.config.get("seed", 42)))
    sensitivity_epsilon = float(
        context.config.get("sensitivity_epsilon", context.config.get("epsilon", 0.02))
    )
    sensitivity_flip_threshold = float(context.config.get("sensitivity_flip_threshold", 0.40))
    consistency_repetitions = int(context.config.get("consistency_repetitions", 3))
    backdoor_trigger_size = int(context.config.get("backdoor_trigger_size", 4))
    backdoor_concentration_threshold = float(
        context.config.get("backdoor_concentration_threshold", 0.85)
    )
    backdoor_confidence_threshold = float(context.config.get("backdoor_confidence_threshold", 0.80))
    backdoor_delta_threshold = float(context.config.get("backdoor_delta_threshold", 0.50))
    reference_agreement_threshold = float(context.config.get("reference_agreement_threshold", 0.95))

    cache_key, cache_metadata = compute_behavioral_cache_identity(
        candidate_sha256=candidate_sha256,
        reference_sha256=ref_digest,
        architecture_id=architecture_id,
        assessment_id=assessment_id,
        job_id=job_id,
        lease_generation=lease_generation,
        probe_count=probe_count,
        probe_batch_size=probe_batch_size,
        probe_seed=probe_seed,
        input_shape=input_shape,
        sensitivity_epsilon=sensitivity_epsilon,
        sensitivity_flip_threshold=sensitivity_flip_threshold,
        consistency_repetitions=consistency_repetitions,
        backdoor_trigger_size=backdoor_trigger_size,
        backdoor_concentration_threshold=backdoor_concentration_threshold,
        backdoor_confidence_threshold=backdoor_confidence_threshold,
        backdoor_delta_threshold=backdoor_delta_threshold,
        reference_agreement_threshold=reference_agreement_threshold,
    )

    cache_store = context.config.setdefault("_behavioral_cache_store", {})
    cache_store[cache_key] = {
        "report": report,
        "metadata": cache_metadata,
    }
    return cache_key, cache_metadata


def get_or_run_behavioral_report(
    detector_id: str,
    detector_version: str,
    context: AssessmentContext,
    reference_checkpoint_path: Path | None = None,
    reference_sha256: str | None = None,
) -> tuple[Path, str, BehavioralReport] | DetectorResult:
    """Retrieves cached BehavioralReport from context or executes SubprocessBehavioralWorker.

    Enforces fail-closed capability gating, cryptographic checkpoint resolution,
    and compound-keyed cache isolation across all behavioral detectors.

    Returns:
        tuple of (model_file_path, architecture_id, behavioral_report) if successful,
        or DetectorResult with UNAVAILABLE / FAILED / NOT_APPLICABLE status.
    """
    snapshot_dir = Path(
        context.config.get(
            "snapshot_root",
            context.snapshot.metadata.get("snapshot_path", "."),
        )
    ).resolve()

    model_file: Path | None = None
    target_path_str = context.snapshot.metadata.get("storage_path")
    if target_path_str and Path(target_path_str).exists():
        model_file = Path(target_path_str).resolve()
    else:
        for p in snapshot_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".pt", ".pth"):
                model_file = p
                break

    if not model_file or not model_file.exists():
        return DetectorResult(
            detector_id=detector_id,
            version=detector_version,
            status=CapabilityStatus.NOT_APPLICABLE,
            status_reason="No PyTorch model checkpoint (.pt/.pth) found in snapshot.",
        )

    architecture_id = (
        context.config.get("architecture_id")
        or context.snapshot.metadata.get("architecture_id")
        or "simple_cnn"
    )

    ref_checkpoint: Path | None = reference_checkpoint_path
    if ref_checkpoint is None:
        ref_path_str = context.config.get("reference_model_path") or context.snapshot.metadata.get(
            "reference_model_path"
        )
        if ref_path_str and Path(ref_path_str).exists():
            ref_checkpoint = Path(ref_path_str).resolve()

    # Calculate content SHA-256 digests
    candidate_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if ref_checkpoint and ref_checkpoint.exists():
        ref_digest = reference_sha256 or hashlib.sha256(ref_checkpoint.read_bytes()).hexdigest()
    else:
        ref_digest = "NO_REFERENCE"

    registry = TrustedArchitectureRegistry()
    input_shape = (
        registry.get_default_input_shape(architecture_id)
        if registry.has(architecture_id)
        else [3, 32, 32]
    )

    assessment_id = str(context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}"))
    job_id = str(context.config.get("job_id", "none"))
    lease_generation = int(context.config.get("lease_generation", 0) or 0)
    probe_count = int(context.config.get("probe_count", context.config.get("probes_count", 30)))
    probe_batch_size = int(context.config.get("probe_batch_size", 10))
    probe_seed = int(context.config.get("probe_seed", context.config.get("seed", 42)))
    sensitivity_epsilon = float(
        context.config.get("sensitivity_epsilon", context.config.get("epsilon", 0.02))
    )
    sensitivity_flip_threshold = float(context.config.get("sensitivity_flip_threshold", 0.40))
    consistency_repetitions = int(context.config.get("consistency_repetitions", 3))
    backdoor_trigger_size = int(context.config.get("backdoor_trigger_size", 4))
    backdoor_concentration_threshold = float(
        context.config.get("backdoor_concentration_threshold", 0.85)
    )
    backdoor_confidence_threshold = float(context.config.get("backdoor_confidence_threshold", 0.80))
    backdoor_delta_threshold = float(context.config.get("backdoor_delta_threshold", 0.50))
    reference_agreement_threshold = float(context.config.get("reference_agreement_threshold", 0.95))

    cache_key, cache_metadata = compute_behavioral_cache_identity(
        candidate_sha256=candidate_sha256,
        reference_sha256=ref_digest,
        architecture_id=architecture_id,
        assessment_id=assessment_id,
        job_id=job_id,
        lease_generation=lease_generation,
        probe_count=probe_count,
        probe_batch_size=probe_batch_size,
        probe_seed=probe_seed,
        input_shape=input_shape,
        sensitivity_epsilon=sensitivity_epsilon,
        sensitivity_flip_threshold=sensitivity_flip_threshold,
        consistency_repetitions=consistency_repetitions,
        backdoor_trigger_size=backdoor_trigger_size,
        backdoor_concentration_threshold=backdoor_concentration_threshold,
        backdoor_confidence_threshold=backdoor_confidence_threshold,
        backdoor_delta_threshold=backdoor_delta_threshold,
        reference_agreement_threshold=reference_agreement_threshold,
    )

    cache_store = context.config.setdefault("_behavioral_cache_store", {})
    report: BehavioralReport | None = None

    if isinstance(cache_store, dict) and cache_key in cache_store:
        entry = cache_store[cache_key]
        if (
            isinstance(entry, dict)
            and entry.get("metadata") == cache_metadata
            and isinstance(entry.get("report"), BehavioralReport)
        ):
            report = entry["report"]

    if report is None:
        behavioral_config = BehavioralConfig(
            probe_count=probe_count,
            probe_batch_size=probe_batch_size,
            probe_seed=probe_seed,
            probe_input_shape=input_shape,
            sensitivity_epsilon=sensitivity_epsilon,
            sensitivity_flip_threshold=sensitivity_flip_threshold,
            consistency_repetitions=consistency_repetitions,
            backdoor_trigger_size=backdoor_trigger_size,
            backdoor_concentration_threshold=backdoor_concentration_threshold,
            backdoor_confidence_threshold=backdoor_confidence_threshold,
            backdoor_delta_threshold=backdoor_delta_threshold,
            reference_agreement_threshold=reference_agreement_threshold,
        )
        worker = SubprocessBehavioralWorker(config=behavioral_config)
        try:
            report = worker.evaluate_behavior(
                checkpoint_path=model_file,
                architecture_id=architecture_id,
                reference_checkpoint_path=ref_checkpoint,
            )
            if isinstance(cache_store, dict):
                cache_store[cache_key] = {
                    "report": report,
                    "metadata": cache_metadata,
                }
                context.config["_cached_behavioral_report"] = report
        except Exception as exc:
            return DetectorResult(
                detector_id=detector_id,
                version=detector_version,
                status=CapabilityStatus.FAILED,
                status_reason=f"Behavioral evaluation worker execution failed: {exc}",
            )

    if report.capability_status == CapabilityStatus.UNAVAILABLE:
        return DetectorResult(
            detector_id=detector_id,
            version=detector_version,
            status=CapabilityStatus.UNAVAILABLE,
            status_reason=report.error_message
            or "Behavioral inference capability is UNAVAILABLE due to lack of OS sandbox isolation.",
            metadata={"isolation_level": report.isolation_level},
        )

    if not report.success or report.capability_status == CapabilityStatus.FAILED:
        return DetectorResult(
            detector_id=detector_id,
            version=detector_version,
            status=CapabilityStatus.FAILED,
            status_reason=report.error_message
            or "Model behavioral evaluation failed inside worker.",
            metadata={"error_type": report.error_type},
        )

    return model_file, architecture_id, report
