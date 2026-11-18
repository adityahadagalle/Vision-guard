"""CLI entrypoint executed inside an isolated worker subprocess to extract ZIP archives.

Performs defensive streaming extraction, verifying resolved path containment,
monitoring uncompressed byte limits, and outputting structured JSON results.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from backend.app.modules.data_integrity.ingestion.inspector import DefensiveArchiveInspector


def safe_extract(
    archive_path: Path,
    output_dir: Path,
    max_bytes: int,
    max_files: int,
    expected_sha256: str | None = None,
) -> dict:
    """Safely extracts a ZIP archive into output_dir within specified resource bounds.

    Enforces streaming byte limits, entry count caps, symlink mode checks, path traversal
    containment, case-insensitive collision prevention, and optional archive SHA-256 verification.
    """
    resolved_archive = archive_path.resolve()
    resolved_out = output_dir.resolve()
    resolved_out.mkdir(parents=True, exist_ok=True)

    # 1. Verify archive SHA-256 if expected_sha256 anchor is provided (anti-TOCTOU)
    if expected_sha256:
        hasher = hashlib.sha256()
        with resolved_archive.open("rb") as f:
            while chunk := f.read(64 * 1024):
                hasher.update(chunk)
        actual_digest = hasher.hexdigest()
        if actual_digest != expected_sha256.lower().strip():
            return {
                "success": False,
                "error": (
                    f"Archive digest mismatch before extraction: computed '{actual_digest}' != "
                    f"expected '{expected_sha256}'. Archive substitution or TOCTOU detected."
                ),
                "code": "ARCHIVE_DIGEST_MISMATCH",
            }

    extracted_files: list[str] = []
    seen_lower_paths: set[str] = set()
    total_bytes = 0

    with zipfile.ZipFile(resolved_archive, "r") as zf:
        infolist = zf.infolist()
        if len(infolist) > max_files:
            return {
                "success": False,
                "error": f"Entry count ({len(infolist)}) exceeds limit ({max_files}).",
                "code": "ENTRY_COUNT_EXCEEDED",
            }

        for info in infolist:
            # Skip directory markers
            if info.is_dir():
                continue

            # 2. Re-check Unix external attributes for symlinks and special files
            mode = info.external_attr >> 16
            if mode != 0:
                file_type = mode & 0o170000
                if file_type == 0o120000:
                    return {
                        "success": False,
                        "error": f"Symlink entries are strictly forbidden: '{info.filename}'.",
                        "code": "SYMLINK_NOT_PERMITTED",
                    }
                if file_type in (0o020000, 0o060000, 0o010000):
                    return {
                        "success": False,
                        "error": f"Special character, block device, or FIFO entries are forbidden: '{info.filename}'.",
                        "code": "DEVICE_FILE_NOT_PERMITTED",
                    }

            clean_path = DefensiveArchiveInspector._validate_entry_path(info.filename)

            # 3. Case-insensitive collision detection during extraction
            lower_path = clean_path.lower()
            if lower_path in seen_lower_paths:
                return {
                    "success": False,
                    "error": f"Duplicate or case-colliding entry detected during extraction: '{clean_path}'.",
                    "code": "DUPLICATE_ENTRY_DETECTED",
                }
            seen_lower_paths.add(lower_path)

            dest_path = (resolved_out / clean_path).resolve()

            # 4. Strict path containment verification
            try:
                dest_path.relative_to(resolved_out)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Path traversal escape detected: '{clean_path}'.",
                    "code": "PATH_TRAVERSAL_DETECTED",
                }

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 5. Defensive streaming read to enforce cumulative uncompressed byte limits
            entry_bytes = 0
            with zf.open(info, "r") as src, dest_path.open("wb") as dst:
                while True:
                    chunk = src.read(64 * 1024)
                    if not chunk:
                        break
                    entry_bytes += len(chunk)
                    total_bytes += len(chunk)

                    if total_bytes > max_bytes:
                        return {
                            "success": False,
                            "error": f"Extracted bytes exceeded limit of {max_bytes} bytes.",
                            "code": "DECOMPRESSION_SIZE_EXCEEDED",
                        }

                    dst.write(chunk)

            extracted_files.append(clean_path)

    return {
        "success": True,
        "extracted_files": extracted_files,
        "total_bytes": total_bytes,
        "file_count": len(extracted_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated archive extraction CLI.")
    parser.add_argument("--archive", required=True, type=Path, help="Path to archive")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Target extraction directory"
    )
    parser.add_argument("--max-bytes", default=10 * 1024 * 1024 * 1024, type=int)
    parser.add_argument("--max-files", default=50_000, type=int)
    parser.add_argument(
        "--expected-sha256", default=None, type=str, help="Expected archive SHA-256"
    )

    args = parser.parse_args()

    try:
        result = safe_extract(
            args.archive,
            args.output_dir,
            args.max_bytes,
            args.max_files,
            expected_sha256=args.expected_sha256,
        )
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        if not result.get("success"):
            sys.exit(2)
        sys.exit(0)
    except Exception as err:
        sys.stdout.write(json.dumps({"success": False, "error": str(err), "code": "WORKER_ERROR"}))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
