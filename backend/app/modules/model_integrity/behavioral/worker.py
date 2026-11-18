"""Subprocess worker boundary for isolated PyTorch model behavioral evaluation.

Enforces fail-closed capability gating when genuine OS sandbox isolation cannot be
proven, and dispatches behavioral probing to a disposable child subprocess under
verified container boundaries.
"""

import base64
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.exceptions import (
    ModelWorkerExecutionError,
    ModelWorkerTimeoutError,
)
from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


class BehavioralReport(BaseModel):
    """Structured report returned across the isolated behavioral worker boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    success: bool
    architecture_id: str = ""
    probes_count: int = 0
    probe_seed: int = 42
    input_shape: list[int] = Field(default_factory=list)
    behavioral_fingerprint: str = ""
    clean_class_distribution: dict[str, int] = Field(default_factory=dict)
    clean_concentration: float = 0.0
    consistency: dict[str, Any] = Field(default_factory=dict)
    sensitivity: dict[str, Any] = Field(default_factory=dict)
    backdoor: dict[str, Any] = Field(default_factory=dict)
    reference_comparison: dict[str, Any] | None = None
    sample_records: list[dict[str, Any]] = Field(default_factory=list)
    latent_representations: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    isolation_level: str = IsolationStatus.UNHARDENED_HOST_SUBPROCESS.value
    capability_status: CapabilityStatus = CapabilityStatus.UNAVAILABLE
    error_message: str | None = None
    error_type: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubprocessBehavioralWorker:
    """Manages isolated subprocess execution for model behavioral inference and backdoor probing."""

    def __init__(
        self,
        config: BehavioralConfig | None = None,
        isolation_assessor: HostIsolationAssessor | None = None,
    ) -> None:
        self.config = config or BehavioralConfig()
        self.assessor = isolation_assessor or HostIsolationAssessor()

    def evaluate_behavior(
        self,
        checkpoint_path: Path,
        architecture_id: str,
        reference_checkpoint_path: Path | None = None,
    ) -> BehavioralReport:
        """Executes behavioral inference inside an isolated subprocess if isolation is verified.

        Args:
            checkpoint_path: Path to candidate model checkpoint (.pt/.pth).
            architecture_id: Verified architecture identifier (e.g. 'simple_cnn').
            reference_checkpoint_path: Optional path to trusted reference model checkpoint.

        Returns:
            BehavioralReport detailing probe outputs or UNAVAILABLE capability status.
        """
        resolved_path = checkpoint_path.resolve()
        isolation_status, details, reason = self.assessor.assess()

        # STRICT CAPABILITY GATING: Fail closed unless hardened isolation is positively verified
        if isolation_status != IsolationStatus.HARDENED_ISOLATION_VERIFIED:
            return BehavioralReport(
                success=False,
                isolation_level=isolation_status.value,
                capability_status=CapabilityStatus.UNAVAILABLE,
                error_message=reason,
                error_type="IsolationUnavailableError",
            )

        cmd = [
            sys.executable,
            "-m",
            "backend.app.modules.model_integrity.behavioral.probe_cli",
            "--checkpoint",
            str(resolved_path),
            "--architecture",
            architecture_id,
            "--probes",
            str(self.config.probe_count),
            "--seed",
            str(self.config.probe_seed),
            "--batch-size",
            str(self.config.probe_batch_size),
            "--epsilon",
            str(self.config.sensitivity_epsilon),
        ]

        if reference_checkpoint_path and reference_checkpoint_path.exists():
            cmd.extend(["--reference-checkpoint", str(reference_checkpoint_path.resolve())])

        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path.cwd()),
            "PYTHONHASHSEED": str(self.config.probe_seed),
        }

        start_time = time.perf_counter()
        proc = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=clean_env,
            )

            stdout, stderr = proc.communicate(timeout=self.config.worker_timeout_seconds)
            duration = time.perf_counter() - start_time

            if proc.returncode != 0:
                error_msg = f"Behavioral worker exited with code {proc.returncode}."
                error_type = "BEHAVIORAL_WORKER_NONZERO_EXIT"
                if stdout.strip():
                    try:
                        err_payload = json.loads(stdout)
                        error_msg = err_payload.get("error", error_msg)
                        error_type = err_payload.get("error_type", error_type)
                    except json.JSONDecodeError:
                        error_msg = f"{error_msg} Stderr: {stderr.strip()}"

                return BehavioralReport(
                    success=False,
                    duration_seconds=round(duration, 3),
                    isolation_level=isolation_status.value,
                    capability_status=CapabilityStatus.FAILED,
                    error_message=error_msg,
                    error_type=error_type,
                )

            payload = json.loads(stdout)
            if not payload.get("success", False):
                return BehavioralReport(
                    success=False,
                    duration_seconds=round(duration, 3),
                    isolation_level=isolation_status.value,
                    capability_status=CapabilityStatus.FAILED,
                    error_message=payload.get(
                        "error", "Behavioral evaluation failed inside worker."
                    ),
                    error_type=payload.get("error_type", "BehavioralEvaluationError"),
                )

            clean_dist = {
                str(k): int(v) for k, v in payload.get("clean_class_distribution", {}).items()
            }

            # Safely decode latent representations payload if present
            latent_representations: dict[str, np.ndarray] = {}
            raw_latents = payload.get("latent_representations")
            if isinstance(raw_latents, dict):
                for class_id, entry in raw_latents.items():
                    if not isinstance(entry, dict):
                        continue
                    data_b64 = entry.get("data_b64")
                    shape = entry.get("shape")
                    dtype_str = entry.get("dtype", "float32")
                    if not data_b64 or not shape or not isinstance(shape, list):
                        continue
                    try:
                        raw_bytes = base64.b64decode(data_b64)
                        dt = np.dtype(dtype_str)
                        target_shape = tuple(shape)
                        expected_bytes = int(np.prod(target_shape)) * dt.itemsize
                        if len(raw_bytes) != expected_bytes:
                            continue
                        arr = np.frombuffer(raw_bytes, dtype=dt).reshape(target_shape).copy()
                        latent_representations[str(class_id)] = arr
                    except Exception:
                        continue

            return BehavioralReport(
                success=True,
                architecture_id=payload.get("architecture_id", architecture_id),
                probes_count=payload.get("probes_count", 0),
                probe_seed=payload.get("probe_seed", self.config.probe_seed),
                input_shape=payload.get("input_shape", []),
                behavioral_fingerprint=payload.get("behavioral_fingerprint", ""),
                clean_class_distribution=clean_dist,
                clean_concentration=payload.get("clean_concentration", 0.0),
                consistency=payload.get("consistency", {}),
                sensitivity=payload.get("sensitivity", {}),
                backdoor=payload.get("backdoor", {}),
                reference_comparison=payload.get("reference_comparison"),
                sample_records=payload.get("sample_records", []),
                latent_representations=latent_representations,
                duration_seconds=round(duration, 3),
                isolation_level=isolation_status.value,
                capability_status=CapabilityStatus.COMPLETED,
            )

        except subprocess.TimeoutExpired as err:
            duration = time.perf_counter() - start_time
            if proc is not None:
                proc.kill()
                proc.communicate()
            raise ModelWorkerTimeoutError(
                f"Behavioral worker subprocess exceeded execution timeout of {self.config.worker_timeout_seconds}s.",
                details={
                    "timeout_seconds": self.config.worker_timeout_seconds,
                    "elapsed": duration,
                },
            ) from err

        except Exception as err:
            raise ModelWorkerExecutionError(
                f"Unexpected error running behavioral worker subprocess: {err}",
                details={"error": str(err)},
            ) from err
