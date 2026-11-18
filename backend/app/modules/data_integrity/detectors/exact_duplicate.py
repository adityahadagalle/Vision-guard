"""Exact duplicate image and cross-split data leakage detector.

Groups image files by their SHA-256 cryptographic digests, flags redundant duplicate
samples within splits, and elevates cross-split data leakage to CRITICAL severity findings.
"""

import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities.enums import (
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_SPLIT_NAMES = frozenset({"train", "val", "validation", "test", "holdout"})


class ExactDuplicateDetector(BaseDetector):
    """Detects identical image byte copies and cross-split evaluation leakage."""

    @property
    def detector_id(self) -> str:
        return "data_integrity.exact_duplicate"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_labels=False, requires_embeddings=False)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Analyzes snapshot manifest digests to group duplicates and identify cross-split leakage."""
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()

        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=f"Manifest file missing from snapshot directory: '{snapshot_dir}'.",
            )

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest_data.get("files", [])
        except Exception as err:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.FAILED,
                status_reason=f"Failed to read snapshot manifest: {err}",
            )

        # Filter image entries
        image_entries = [
            f for f in files if Path(f.get("relative_path", "")).suffix.lower() in _IMAGE_EXTS
        ]

        if not image_entries:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.NOT_APPLICABLE,
                status_reason="No image files found in snapshot manifest to inspect.",
                metadata={"total_files": len(files)},
            )

        # Group by SHA-256
        hash_groups: dict[str, list[dict]] = defaultdict(list)
        for entry in image_entries:
            sha = entry.get("sha256")
            if sha:
                hash_groups[sha].append(entry)

        # Identify duplicate groups
        duplicate_groups = [
            (sha, entries) for sha, entries in hash_groups.items() if len(entries) > 1
        ]

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        now = datetime.now(UTC)
        cross_split_count = 0

        for sha, entries in duplicate_groups:
            paths = [e["relative_path"] for e in entries]

            # Detect split associations for each duplicate copy
            splits_detected: set[str] = set()
            for rel in paths:
                for part in rel.split("/"):
                    if part.lower() in _SPLIT_NAMES:
                        splits_detected.add(part.lower())
                        break

            is_cross_split = len(splits_detected) > 1
            uid = uuid.uuid4().hex[:12]
            fnd_id = f"fnd_dup_{uid}"
            ev_id = f"ev_dup_{uid}"

            if is_cross_split:
                cross_split_count += 1
                category = FindingCategory.SPLIT_LEAKAGE
                severity = SeverityLevel.CRITICAL
                disposition = SuggestedDisposition.QUARANTINE
                splits_str = ", ".join(sorted(splits_detected))
                title = f"Critical Cross-Split Exact Duplicate Leakage ({len(entries)} copies across: {splits_str})"
                description = (
                    f"Exact SHA-256 byte collision detected across distinct evaluation splits ({splits_str}): "
                    f"{paths}. Cross-split leakage compromises evaluation integrity and risks false benchmark metrics."
                )
            else:
                category = FindingCategory.DUPLICATE_SAMPLE
                severity = SeverityLevel.MEDIUM
                disposition = SuggestedDisposition.REVIEW
                title = f"Intra-Split Exact Duplicate Group ({len(entries)} identical copies)"
                description = (
                    f"Identical byte content (SHA-256: {sha[:16]}...) detected across multiple "
                    f"file paths in the dataset: {paths}."
                )

            ev = EvidenceRecord(
                evidence_id=ev_id,
                evidence_type=EvidenceType.HASH_COLLISION,
                finding_id=fnd_id,
                signal_name="exact_sha256_collision",
                rationale=(f"Found {len(entries)} identical files sharing SHA-256 digest '{sha}'."),
                created_at=now,
                sample_reference=paths[0],
                evidence_digest=sha,
                observed_value=float(len(entries)),
                threshold_value=1.0,
                metadata={
                    "sha256": sha,
                    "duplicate_paths": paths,
                    "splits": sorted(splits_detected),
                    "is_cross_split": is_cross_split,
                    "copy_count": len(entries),
                },
            )

            affected = [
                AffectedAssetReference(
                    asset_id=context.snapshot.asset_id,
                    sample_path=rel,
                    sha256=sha,
                )
                for rel in paths
            ]

            fnd = UniversalFinding(
                finding_id=fnd_id,
                assessment_id=context.config.get(
                    "assessment_id", f"asm_{context.snapshot.snapshot_id}"
                ),
                detector_id=self.detector_id,
                detector_version=self.version,
                category=category,
                severity=severity,
                title=title,
                description=description,
                confidence=1.0,
                reproducibility=ReproducibilityMetadata(
                    execution_timestamp=now,
                    software_versions={"detector": self.version},
                    detector_config={"algorithm": "sha256_hash_grouping"},
                ),
                created_at=now,
                suggested_disposition=disposition,
                anomaly_score=1.0,
                affected_assets=affected,
                evidence_ids=[ev_id],
                details={
                    "sha256": sha,
                    "duplicate_paths": paths,
                    "is_cross_split": is_cross_split,
                },
            )

            findings.append(fnd)
            evidence_list.append(ev)

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "scanned_images": len(image_entries),
                "unique_hashes": len(hash_groups),
                "duplicate_groups_count": len(duplicate_groups),
                "cross_split_leakage_groups": cross_split_count,
            },
        )
