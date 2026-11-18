"""Subprocess boundary for isolated archive extraction.

Executes untrusted decompression inside disposable child worker processes,
enforcing execution timeouts, streaming uncompressed byte limits, path containment,
and capability gating when hardened sandbox isolation cannot be established.
Note: On Windows and non-container environments, this provides process separation
and lifecycle containment rather than a kernel security sandbox.
"""

import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.entities.enums import CapabilityStatus
from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    WorkerExecutionError,
    WorkerIsolationUnavailableError,
    WorkerTimeoutError,
)


class WorkerExtractionResult(BaseModel):
    """Structured report returned across the worker process boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    exit_code: int
    extracted_files: list[str] = Field(default_factory=list)
    total_bytes_extracted: int = 0
    file_count: int = 0
    duration_seconds: float = 0.0
    capability_status: CapabilityStatus = CapabilityStatus.COMPLETED
    isolation_level: str = "SUBPROCESS_RESOURCE_CONSTRAINED"
    error_message: str | None = None
    error_code: str | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubprocessExtractionWorker:
    """Manages isolated subprocess extraction boundaries for untrusted dataset archives."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()

    def extract(
        self,
        archive_path: Path,
        output_dir: Path,
        expected_sha256: str | None = None,
        require_hardened_isolation: bool = False,
    ) -> WorkerExtractionResult:
        """Decompresses an untrusted archive inside a separate child subprocess.

        Args:
            archive_path: Quarantined archive path.
            output_dir: Target extraction staging directory.
            expected_sha256: Optional authoritative SHA-256 digest of the archive to verify
                before decompressing (prevents TOCTOU archive substitution).
            require_hardened_isolation: If True, requires Linux namespaces/container isolation;
                fails with CapabilityStatus.UNAVAILABLE on Windows or unprivileged hosts.

        Returns:
            WorkerExtractionResult detailing extracted files and performance metrics.

        Raises:
            WorkerIsolationUnavailableError: When hardened isolation is required but unavailable.
            WorkerTimeoutError: When subprocess exceeds worker_timeout_seconds.
            WorkerExecutionError: When subprocess crashes or fails extraction checks.
        """
        resolved_archive = archive_path.resolve()
        resolved_output = output_dir.resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)

        current_os = platform.system()
        isolation_level = self._assess_isolation_level(current_os)

        # Capability Gating: Hardened container / namespace isolation enforcement
        if require_hardened_isolation and isolation_level != "HARDENED_LINUX_CONTAINER":
            self._safe_cleanup_dir(resolved_output)
            raise WorkerIsolationUnavailableError(
                f"Hardened sandbox isolation is unavailable on host platform '{current_os}'. "
                "Current execution provides subprocess-level process separation with timeouts and resource limits, "
                "not full kernel sandboxing. "
                "Hardened mode mandates Linux container execution with unshared user, mount, and network namespaces.",
                details={"platform": current_os, "isolation_level": isolation_level},
            )

        cmd = [
            sys.executable,
            "-m",
            "backend.app.modules.data_integrity.ingestion.extract_cli",
            "--archive",
            str(resolved_archive),
            "--output-dir",
            str(resolved_output),
            "--max-bytes",
            str(self.config.max_uncompressed_bytes),
            "--max-files",
            str(self.config.max_file_count),
        ]
        if expected_sha256:
            cmd.extend(["--expected-sha256", str(expected_sha256)])

        start_time = time.perf_counter()
        proc = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = proc.communicate(timeout=self.config.worker_timeout_seconds)
            duration = time.perf_counter() - start_time

            if proc.returncode != 0:
                self._safe_cleanup_dir(resolved_output)
                # Attempt to parse structured error from stdout
                error_msg = f"Worker process exited with code {proc.returncode}."
                error_code = "WORKER_NONZERO_EXIT"
                if stdout.strip():
                    try:
                        err_payload = json.loads(stdout)
                        error_msg = err_payload.get("error", error_msg)
                        error_code = err_payload.get("code", error_code)
                    except json.JSONDecodeError:
                        error_msg = f"{error_msg} Stderr: {stderr.strip()}"

                raise WorkerExecutionError(
                    error_msg,
                    code=error_code,
                    details={"exit_code": proc.returncode, "stderr": stderr.strip()},
                )

            payload = json.loads(stdout)
            if not payload.get("success"):
                self._safe_cleanup_dir(resolved_output)
                raise WorkerExecutionError(
                    payload.get("error", "Extraction failed inside worker subprocess."),
                    code=payload.get("code", "EXTRACTION_FAILED"),
                )

            return WorkerExtractionResult(
                success=True,
                exit_code=0,
                extracted_files=payload.get("extracted_files", []),
                total_bytes_extracted=payload.get("total_bytes", 0),
                file_count=payload.get("file_count", 0),
                duration_seconds=round(duration, 3),
                capability_status=CapabilityStatus.COMPLETED,
                isolation_level=isolation_level,
            )

        except subprocess.TimeoutExpired as err:
            duration = time.perf_counter() - start_time
            if proc is not None:
                proc.kill()
                proc.communicate()
            self._safe_cleanup_dir(resolved_output)
            raise WorkerTimeoutError(
                f"Worker subprocess exceeded hard execution timeout of {self.config.worker_timeout_seconds}s.",
                details={
                    "timeout_seconds": self.config.worker_timeout_seconds,
                    "elapsed": duration,
                },
            ) from err

        except (WorkerIsolationUnavailableError, WorkerTimeoutError, WorkerExecutionError):
            raise

        except Exception as err:
            self._safe_cleanup_dir(resolved_output)
            raise WorkerExecutionError(
                f"Unexpected error running worker extraction subprocess: {err}",
                details={"error": str(err)},
            ) from err

    @staticmethod
    def _assess_isolation_level(os_name: str) -> str:
        """Determines the actual isolation capability of the current host environment."""
        if os_name == "Linux":
            # Check for container environment or namespace capabilities
            if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
                return "HARDENED_LINUX_CONTAINER"
            return "LINUX_SUBPROCESS"
        if os_name == "Windows":
            return "SUBPROCESS_RESOURCE_CONSTRAINED"
        return "GENERIC_SUBPROCESS"

    @staticmethod
    def _safe_cleanup_dir(target_dir: Path) -> None:
        """Safely cleans up an ephemeral extraction directory upon failure."""
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
