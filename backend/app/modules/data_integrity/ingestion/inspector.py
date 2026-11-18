"""Defensive pre-inspection for untrusted dataset archive containers.

Enforces deep container validation prior to extraction: defends against
ZIP Slip, decompression bombs, symlink escapes, Windows device names,
NTFS alternate data streams, and case-collision attacks.
"""

import re
import zipfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.modules.data_integrity.ingestion.config import IngestionConfig
from backend.app.modules.data_integrity.ingestion.exceptions import (
    AlternateDataStreamError,
    CompressionRatioExceededError,
    DecompressionBombError,
    DuplicateEntryError,
    EntryCountLimitExceededError,
    MalformedArchiveError,
    PathTraversalError,
    ReservedDeviceNameError,
    SymlinkNotPermittedError,
)

_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_DRIVE_LETTER_PATTERN = re.compile(r"^[a-zA-Z]:")


class ArchiveInspectionResult(BaseModel):
    """Structured inspection summary certifying container safety prior to extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_entries: int
    total_uncompressed_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    entries: list[str]
    warnings: list[str] = Field(default_factory=list)


class DefensiveArchiveInspector:
    """Performs strict defensive inspection of untrusted ZIP archive files."""

    @classmethod
    def inspect_zip(
        cls,
        archive_path: Path,
        config: IngestionConfig | None = None,
    ) -> ArchiveInspectionResult:
        """Inspects and validates an untrusted ZIP container against all security invariants.

        Args:
            archive_path: Filesystem path to the quarantined archive file.
            config: Optional custom IngestionConfig (uses defaults if None).

        Returns:
            ArchiveInspectionResult certifying that the archive meets all safety limits.

        Raises:
            MalformedArchiveError: If the file is not a valid ZIP container.
            EntryCountLimitExceededError: If entry count exceeds config.max_file_count.
            DecompressionBombError: If uncompressed size or single file exceeds bounds.
            CompressionRatioExceededError: If compression ratio exceeds config.max_compression_ratio.
            PathTraversalError: On any directory traversal attempt (ZIP Slip, absolute paths).
            ReservedDeviceNameError: If paths target Windows device names.
            AlternateDataStreamError: If paths contain Alternate Data Stream syntax (:).
            SymlinkNotPermittedError: If archive entries declare symlink or device attributes.
            DuplicateEntryError: On duplicate paths or case-insensitive collisions.
        """
        cfg = config or IngestionConfig()
        resolved_path = archive_path.resolve()

        if not resolved_path.exists() or not resolved_path.is_file():
            raise MalformedArchiveError(
                f"Archive file does not exist or is not a regular file: {archive_path}",
                details={"path": str(archive_path)},
            )

        compressed_size = resolved_path.stat().st_size
        if compressed_size == 0:
            raise MalformedArchiveError(
                "Archive file contains 0 bytes.",
                details={"path": str(archive_path)},
            )

        try:
            with zipfile.ZipFile(resolved_path, "r") as zf:
                # Validate archive structure and integrity
                bad_crc = zf.testzip()
                if bad_crc is not None:
                    raise MalformedArchiveError(
                        f"Archive CRC check failed on entry: '{bad_crc}'.",
                        details={"bad_entry": bad_crc},
                    )

                infolist = zf.infolist()
                total_entries = len(infolist)

                if total_entries > cfg.max_file_count:
                    raise EntryCountLimitExceededError(
                        f"Archive entry count ({total_entries}) exceeds maximum permissible "
                        f"limit of {cfg.max_file_count}.",
                        details={"entry_count": total_entries, "max_entries": cfg.max_file_count},
                    )

                total_uncompressed = 0
                seen_lower_paths: set[str] = set()
                normalized_entries: list[str] = []
                warnings: list[str] = []

                for info in infolist:
                    filename = info.filename

                    # 1. Inspect Unix mode attributes for symlinks, hardlinks, and special devices
                    mode = info.external_attr >> 16
                    if mode != 0:
                        file_type = mode & 0o170000
                        if file_type == 0o120000:
                            raise SymlinkNotPermittedError(
                                f"Symlink entries are strictly forbidden: '{filename}'.",
                                details={"entry": filename, "mode": oct(mode)},
                            )
                        if file_type in (0o020000, 0o060000, 0o010000):
                            raise SymlinkNotPermittedError(
                                f"Special character, block device, or FIFO entries are forbidden: '{filename}'.",
                                details={"entry": filename, "mode": oct(mode)},
                            )

                    # 2. Strict path inspection and normalization
                    clean_path = cls._validate_entry_path(filename)

                    # Skip directory marker entries from duplicate collision checks
                    if info.is_dir() or clean_path.endswith("/"):
                        continue

                    # 3. Case-insensitive collision detection
                    lower_path = clean_path.lower()
                    if lower_path in seen_lower_paths:
                        raise DuplicateEntryError(
                            f"Duplicate or case-colliding entry detected in archive: '{clean_path}'.",
                            details={"entry": clean_path},
                        )
                    seen_lower_paths.add(lower_path)
                    normalized_entries.append(clean_path)

                    # 4. Individual file size check
                    if info.file_size > cfg.max_file_bytes:
                        raise DecompressionBombError(
                            f"Archive entry '{clean_path}' uncompressed size ({info.file_size} bytes) "
                            f"exceeds single file limit ({cfg.max_file_bytes} bytes).",
                            details={
                                "entry": clean_path,
                                "file_size": info.file_size,
                                "max_file_bytes": cfg.max_file_bytes,
                            },
                        )

                    total_uncompressed += info.file_size

                    # 5. Running total uncompressed size limit
                    if total_uncompressed > cfg.max_uncompressed_bytes:
                        raise DecompressionBombError(
                            f"Archive uncompressed size exceeds maximum permissible limit of "
                            f"{cfg.max_uncompressed_bytes} bytes (reached {total_uncompressed} bytes).",
                            details={
                                "total_bytes": total_uncompressed,
                                "max_uncompressed": cfg.max_uncompressed_bytes,
                            },
                        )

                # 6. Global compression ratio check
                effective_compressed = max(1, compressed_size)
                ratio = total_uncompressed / effective_compressed

                # Enforce ratio checks only when uncompressed content exceeds 10 MB (ignoring trivial text files)
                if total_uncompressed > 10 * 1024 * 1024 and ratio > cfg.max_compression_ratio:
                    raise CompressionRatioExceededError(
                        f"Archive compression ratio ({ratio:.1f}:1) exceeds safety threshold "
                        f"of {cfg.max_compression_ratio:.1f}:1 (decompression bomb defense).",
                        details={
                            "compression_ratio": ratio,
                            "max_ratio": cfg.max_compression_ratio,
                        },
                    )

                return ArchiveInspectionResult(
                    total_entries=total_entries,
                    total_uncompressed_bytes=total_uncompressed,
                    compressed_size_bytes=compressed_size,
                    compression_ratio=round(ratio, 2),
                    entries=normalized_entries,
                    warnings=warnings,
                )

        except (
            EntryCountLimitExceededError,
            DecompressionBombError,
            CompressionRatioExceededError,
            PathTraversalError,
            ReservedDeviceNameError,
            AlternateDataStreamError,
            SymlinkNotPermittedError,
            DuplicateEntryError,
            MalformedArchiveError,
        ):
            raise
        except zipfile.BadZipFile as err:
            raise MalformedArchiveError(
                f"File is not a valid ZIP archive: {err}",
                details={"error": str(err)},
            ) from err
        except Exception as err:
            raise MalformedArchiveError(
                f"Unexpected error inspecting archive '{archive_path.name}': {err}",
                details={"error": str(err)},
            ) from err

    @classmethod
    def _validate_entry_path(cls, raw_path: str) -> str:
        """Validates that an archive entry relative path does not escape the sandbox root."""
        clean = raw_path.strip()
        if not clean:
            raise PathTraversalError("Archive contains an empty entry path.")

        if "\0" in clean:
            raise PathTraversalError(f"Entry path contains prohibited null byte: '{raw_path}'.")

        # Reject absolute paths (Unix or Windows drive letters)
        if clean.startswith("/") or clean.startswith("\\") or _DRIVE_LETTER_PATTERN.match(clean):
            raise PathTraversalError(
                f"Absolute entry paths are strictly forbidden: '{raw_path}'.",
                details={"entry": raw_path},
            )

        # Reject Alternate Data Streams (colons)
        if ":" in clean:
            raise AlternateDataStreamError(
                f"Entry path contains prohibited Alternate Data Stream syntax (colon): '{raw_path}'.",
                details={"entry": raw_path},
            )

        # Normalize backslashes to forward slashes
        normalized = clean.replace("\\", "/")

        # Strip redundant leading './' prefixes
        while normalized.startswith("./"):
            normalized = normalized[2:]

        # Split components and inspect each segment
        segments = [seg for seg in normalized.split("/") if seg and seg != "."]
        if not segments:
            raise PathTraversalError(f"Entry path resolves to root: '{raw_path}'.")

        for segment in segments:
            if segment == "..":
                raise PathTraversalError(
                    f"Path traversal component ('..') detected in archive entry: '{raw_path}'.",
                    details={"entry": raw_path},
                )

            # Reject segments ending with dots or spaces (NTFS aliasing vulnerabilities)
            if segment.endswith(".") or segment.endswith(" "):
                raise PathTraversalError(
                    f"Entry path segment with trailing space or dot is prohibited on NTFS: '{segment}'.",
                    details={"entry": raw_path, "segment": segment},
                )

            # Check for Windows reserved device names
            base_name = segment.split(".")[0].lower()
            if base_name in _WINDOWS_DEVICE_NAMES:
                raise ReservedDeviceNameError(
                    f"Entry path uses reserved Windows device name: '{segment}'.",
                    details={"entry": raw_path, "device_name": base_name},
                )

        return "/".join(segments)
