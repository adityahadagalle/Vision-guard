"""Subprocess worker boundary for isolated PyTorch model inspection.

Executes checkpoint deserialization and state-dict analysis inside a disposable
child worker process, enforcing execution timeouts, memory limits, and capability
gating when hardened sandbox isolation cannot be established.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.exceptions import (
    ModelWorkerExecutionError,
    ModelWorkerTimeoutError,
)


class IsolationStatus(StrEnum):
    """Explicit lifecycle classification of host OS isolation guarantees."""

    HARDENED_ISOLATION_VERIFIED = "HARDENED_ISOLATION_VERIFIED"
    UNHARDENED_HOST_SUBPROCESS = "UNHARDENED_HOST_SUBPROCESS"
    ISOLATION_UNAVAILABLE = "ISOLATION_UNAVAILABLE"
    ISOLATION_FAILED = "ISOLATION_FAILED"


class HostIsolationAssessor:
    """Probes the host operating system to determine whether genuine sandbox isolation can be proven."""

    def assess(self) -> tuple[IsolationStatus, dict[str, Any], str]:
        """Assesses platform-specific process, filesystem, and network isolation controls."""
        try:
            current_os = platform.system()
            if current_os == "Windows":
                # On Windows, standard desktop/server processes inherit full host token, network, and filesystem.
                details = {
                    "platform": "Windows",
                    "appcontainer": False,
                    "network_severed": False,
                    "filesystem_restricted": False,
                    "job_object_limits": False,
                    "no_new_privs": False,
                    "non_root_user": False,
                }
                reason = (
                    "Host platform 'Windows' lacks proven OS AppContainer sandboxing, network severance, "
                    "read-only root mounts, and filesystem isolation controls required for untrusted checkpoint deserialization."
                )
                return IsolationStatus.UNHARDENED_HOST_SUBPROCESS, details, reason

            elif current_os == "Linux":
                in_container = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()

                # 1. cgroups v2 memory quota
                has_cgroup_mem_limit = False
                cgroup_mem_file = Path("/sys/fs/cgroup/memory.max")
                if cgroup_mem_file.exists():
                    try:
                        val = cgroup_mem_file.read_text().strip()
                        has_cgroup_mem_limit = val != "max"
                    except Exception:
                        has_cgroup_mem_limit = False

                # 2. cgroups v2 process/PID limit (fork-bomb defense)
                has_cgroup_pids_limit = False
                cgroup_pids_file = Path("/sys/fs/cgroup/pids.max")
                if cgroup_pids_file.exists():
                    try:
                        val = cgroup_pids_file.read_text().strip()
                        has_cgroup_pids_limit = val != "max"
                    except Exception:
                        has_cgroup_pids_limit = False

                # 3. network isolation (loopback-only or empty routing table)
                net_severed = False
                route_file = Path("/proc/net/route")
                if route_file.exists():
                    try:
                        routes = [
                            line.split()[0]
                            for line in route_file.read_text().splitlines()[1:]
                            if line.strip()
                        ]
                        net_severed = len(routes) == 0 or all(r == "lo" for r in routes)
                    except Exception:
                        net_severed = False

                # 4. NoNewPrivileges enforcement
                no_new_privs = False
                status_file = Path("/proc/self/status")
                if status_file.exists():
                    try:
                        for line in status_file.read_text().splitlines():
                            if line.startswith("NoNewPrivs:"):
                                no_new_privs = line.split(":", 1)[1].strip() == "1"
                                break
                    except Exception:
                        no_new_privs = False

                # 5. Non-root unprivileged execution
                non_root_user = False
                try:
                    if hasattr(os, "getuid"):
                        non_root_user = os.getuid() != 0
                except Exception:
                    non_root_user = False

                # 6. Read-only filesystem / mount restrictions
                filesystem_restricted = False
                mounts_file = Path("/proc/mounts")
                if mounts_file.exists():
                    try:
                        for line in mounts_file.read_text().splitlines():
                            parts = line.split()
                            if len(parts) >= 4 and parts[1] == "/":
                                opts = parts[3].split(",")
                                if "ro" in opts:
                                    filesystem_restricted = True
                                break
                    except Exception:
                        filesystem_restricted = False

                missing_controls: list[str] = []
                if not in_container:
                    missing_controls.append("container_boundary")
                if not has_cgroup_mem_limit:
                    missing_controls.append("cgroup_memory_limit")
                if not has_cgroup_pids_limit:
                    missing_controls.append("cgroup_pids_limit")
                if not net_severed:
                    missing_controls.append("network_isolation")
                if not no_new_privs:
                    missing_controls.append("no_new_privileges")
                if not non_root_user:
                    missing_controls.append("non_root_user")
                if not filesystem_restricted:
                    missing_controls.append("read_only_root_filesystem")

                details = {
                    "platform": "Linux",
                    "container": in_container,
                    "cgroup_memory_max": has_cgroup_mem_limit,
                    "cgroup_pids_max": has_cgroup_pids_limit,
                    "network_severed": net_severed,
                    "no_new_privs": no_new_privs,
                    "non_root_user": non_root_user,
                    "filesystem_restricted": filesystem_restricted,
                    "missing_controls": missing_controls,
                }

                if not missing_controls:
                    return (
                        IsolationStatus.HARDENED_ISOLATION_VERIFIED,
                        details,
                        "Hardened container sandbox verified with active network severance, cgroup quotas, and privilege restrictions.",
                    )

                reason = (
                    f"Host platform 'Linux' lacks verified OS sandbox controls: {', '.join(missing_controls)}. "
                    "Untrusted checkpoint deserialization requires proven container boundaries, cgroup memory/pid quotas, "
                    "severed network, no-new-privileges, unprivileged user, and read-only mount controls."
                )
                return IsolationStatus.UNHARDENED_HOST_SUBPROCESS, details, reason

            else:
                details = {"platform": current_os}
                reason = f"Operating system '{current_os}' is not supported for hardened model isolation."
                return IsolationStatus.ISOLATION_UNAVAILABLE, details, reason

        except Exception as exc:
            return (
                IsolationStatus.ISOLATION_FAILED,
                {"error": str(exc)},
                f"Isolation assessment failed: {exc}",
            )


class CheckpointInspectionReport(BaseModel):
    """Structured report returned across the isolated model worker boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    format: str = "unknown"
    weights_only_verified: bool = False
    state_dict_fingerprint: str = ""
    total_tensors: int = 0
    total_parameters: int = 0
    total_memory_bytes: int = 0
    defects: list[str] = Field(default_factory=list)
    has_nan: bool = False
    has_inf: bool = False
    total_nan: int = 0
    total_inf: int = 0
    affected_non_finite_tensors: list[str] = Field(default_factory=list)
    tensors: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    isolation_level: str = IsolationStatus.UNHARDENED_HOST_SUBPROCESS.value
    capability_status: CapabilityStatus = CapabilityStatus.UNAVAILABLE
    error_message: str | None = None
    error_type: str | None = None
    inspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubprocessModelWorker:
    """Manages isolated subprocess execution for candidate PyTorch checkpoints."""

    def __init__(
        self,
        config: ModelIntegrityConfig | None = None,
        isolation_assessor: HostIsolationAssessor | None = None,
    ) -> None:
        self.config = config or ModelIntegrityConfig()
        self.assessor = isolation_assessor or HostIsolationAssessor()

    def inspect(self, checkpoint_path: Path) -> CheckpointInspectionReport:
        """Inspects an untrusted model checkpoint inside an isolated child subprocess.

        Args:
            checkpoint_path: Path to candidate .pt/.pth checkpoint file.

        Returns:
            CheckpointInspectionReport detailing tensor metadata or UNAVAILABLE status.
        """
        resolved_path = checkpoint_path.resolve()
        isolation_status, details, reason = self.assessor.assess()

        # STRICT CAPABILITY GATING: Fail closed unless hardened isolation is positively verified
        if isolation_status != IsolationStatus.HARDENED_ISOLATION_VERIFIED:
            return CheckpointInspectionReport(
                success=False,
                isolation_level=isolation_status.value,
                capability_status=CapabilityStatus.UNAVAILABLE,
                error_message=reason,
                error_type="IsolationUnavailableError",
            )

        cmd = [
            sys.executable,
            "-m",
            "backend.app.modules.model_integrity.inspection.inspect_cli",
            "--checkpoint",
            str(resolved_path),
            "--max-parameters",
            str(self.config.max_parameter_count),
            "--max-tensor-memory-bytes",
            str(self.config.max_tensor_memory_bytes),
            "--max-tensors",
            str(self.config.max_tensor_count),
        ]

        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path.cwd()),
            "PYTHONHASHSEED": "42",
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
                error_msg = f"Model worker exited with code {proc.returncode}."
                error_type = "MODEL_WORKER_NONZERO_EXIT"
                if stdout.strip():
                    try:
                        err_payload = json.loads(stdout)
                        error_msg = err_payload.get("error", error_msg)
                        error_type = err_payload.get("error_type", error_type)
                    except json.JSONDecodeError:
                        error_msg = f"{error_msg} Stderr: {stderr.strip()}"

                return CheckpointInspectionReport(
                    success=False,
                    duration_seconds=round(duration, 3),
                    isolation_level=isolation_status.value,
                    capability_status=CapabilityStatus.FAILED,
                    error_message=error_msg,
                    error_type=error_type,
                )

            payload = json.loads(stdout)
            if not payload.get("success", False):
                return CheckpointInspectionReport(
                    success=False,
                    duration_seconds=round(duration, 3),
                    isolation_level=isolation_status.value,
                    capability_status=CapabilityStatus.FAILED,
                    error_message=payload.get("error", "Inspection failed inside worker."),
                    error_type=payload.get("error_type", "InspectionError"),
                )

            non_finite = payload.get("non_finite_summary", {})
            return CheckpointInspectionReport(
                success=True,
                format=payload.get("format", "unknown"),
                weights_only_verified=payload.get("weights_only_verified", False),
                state_dict_fingerprint=payload.get("state_dict_fingerprint", ""),
                total_tensors=payload.get("total_tensors", 0),
                total_parameters=payload.get("total_parameters", 0),
                total_memory_bytes=payload.get("total_memory_bytes", 0),
                defects=payload.get("defects", []),
                has_nan=non_finite.get("has_nan", False),
                has_inf=non_finite.get("has_inf", False),
                total_nan=non_finite.get("total_nan", 0),
                total_inf=non_finite.get("total_inf", 0),
                affected_non_finite_tensors=non_finite.get("affected_tensors", []),
                tensors=payload.get("tensors", {}),
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
                f"Model worker subprocess exceeded execution timeout of {self.config.worker_timeout_seconds}s.",
                details={
                    "timeout_seconds": self.config.worker_timeout_seconds,
                    "elapsed": duration,
                },
            ) from err

        except Exception as err:
            raise ModelWorkerExecutionError(
                f"Unexpected error running model worker subprocess: {err}",
                details={"error": str(err)},
            ) from err
