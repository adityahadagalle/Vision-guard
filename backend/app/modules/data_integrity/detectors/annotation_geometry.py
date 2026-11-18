"""Annotation geometry validator for object detection datasets.

Validates bounding box geometry invariants for COCO and YOLO formats:
1. Negative coordinates or dimensions.
2. Non-positive or zero width/height.
3. COCO bounding boxes exceeding actual image boundary dimensions (x + w > W, y + h > H).
4. YOLO normalized center coordinates and extent corners outside [0.0, 1.0].
5. Missing image/annotation references.
"""

import json
import uuid
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


class AnnotationGeometryValidator(BaseDetector):
    """Validates structural and spatial geometry invariants of detection bounding boxes."""

    @property
    def detector_id(self) -> str:
        return "data_integrity.annotation_geometry"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(requires_labels=True, requires_embeddings=False)

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Inspects bounding box coordinates against image spatial boundaries."""
        snapshot_dir = Path(
            context.config.get(
                "snapshot_root",
                context.snapshot.metadata.get("snapshot_path", "."),
            )
        ).resolve()

        metadata_path = snapshot_dir / "metadata.json"
        dataset_format = context.snapshot.metadata.get("format")

        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                dataset_format = meta.get("original_format", dataset_format)
            except Exception as err:
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.FAILED,
                    status_reason=f"Failed to read snapshot metadata: {err}",
                )

        # Classification datasets do not declare bounding box geometries
        if dataset_format == "FOLDER_PER_CLASS":
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.NOT_APPLICABLE,
                status_reason="Annotation geometry validation is not applicable to classification datasets.",
                metadata={"format": dataset_format},
            )

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        now = datetime.now(UTC)
        total_boxes_checked = 0

        # 1. COCO Detection Validation
        if dataset_format == "COCO_DETECTION" or (snapshot_dir / "annotations").exists():
            self._validate_coco(snapshot_dir, context, findings, evidence_list, now)
            total_boxes_checked += len(evidence_list)

        # 2. YOLO Detection Validation
        elif dataset_format == "YOLO_DETECTION" or (snapshot_dir / "labels").exists():
            self._validate_yolo(snapshot_dir, context, findings, evidence_list, now)
            total_boxes_checked += len(evidence_list)

        else:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.NOT_APPLICABLE,
                status_reason=f"Unsupported format '{dataset_format}' for annotation geometry inspection.",
            )

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "dataset_format": dataset_format,
                "defective_boxes_count": len(findings),
            },
        )

    def _validate_coco(
        self,
        snapshot_dir: Path,
        context: AssessmentContext,
        findings: list[UniversalFinding],
        evidence_list: list[EvidenceRecord],
        now: datetime,
    ) -> None:
        """Validates COCO annotations against absolute image pixel boundaries."""
        candidates = list((snapshot_dir / "annotations").glob("*.json"))
        if not candidates:
            candidates = list(snapshot_dir.glob("*.json"))

        for cand in candidates:
            if cand.name in ("manifest.json", "metadata.json"):
                continue
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
                continue

            image_map = {
                img["id"]: img
                for img in data.get("images", [])
                if isinstance(img, dict) and "id" in img
            }

            for anno in data.get("annotations", []):
                if not isinstance(anno, dict):
                    continue

                anno_id = anno.get("id", "unknown")
                img_id = anno.get("image_id")

                # Check missing image reference
                if img_id not in image_map:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"COCO Annotation {anno_id}: Dangling Image Reference",
                        description=f"Annotation references non-existent image_id '{img_id}'.",
                        sample_path=None,
                        defect_type="MISSING_IMAGE_REFERENCE",
                        details={"annotation_id": anno_id, "image_id": img_id},
                    )
                    continue

                img_info = image_map[img_id]
                img_w = img_info.get("width")
                img_h = img_info.get("height")
                sample_file = img_info.get("file_name")

                bbox = anno.get("bbox")
                if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"COCO Annotation {anno_id}: Malformed Bounding Box Structure",
                        description=f"Expected [x, y, w, h] list of 4 numbers, got {bbox}.",
                        sample_path=sample_file,
                        defect_type="MALFORMED_BBOX_STRUCTURE",
                        details={"annotation_id": anno_id, "bbox": bbox},
                    )
                    continue

                x, y, w, h = bbox

                # Check non-positive dimensions
                if w <= 0 or h <= 0:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"COCO Annotation {anno_id}: Non-Positive Bounding Box Dimension",
                        description=f"Bounding box dimensions must be strictly positive: [w={w}, h={h}].",
                        sample_path=sample_file,
                        defect_type="NON_POSITIVE_DIMENSION",
                        details={"annotation_id": anno_id, "bbox": bbox},
                    )
                    continue

                # Check negative coordinates
                if x < 0 or y < 0:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"COCO Annotation {anno_id}: Negative Coordinate Origin",
                        description=f"Bounding box top-left corner is outside image canvas: [x={x}, y={y}].",
                        sample_path=sample_file,
                        defect_type="NEGATIVE_COORDINATES",
                        details={"annotation_id": anno_id, "bbox": bbox},
                    )

                # Check image boundary containment if dimensions are declared
                if (
                    img_w is not None
                    and img_h is not None
                    and img_w > 0
                    and img_h > 0
                    and ((x + w) > img_w or (y + h) > img_h)
                ):
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"COCO Annotation {anno_id}: Bounding Box Exceeds Image Canvas",
                        description=(
                            f"Box extent [x+w={x + w}, y+h={y + h}] exceeds image canvas dimensions "
                            f"[W={img_w}, H={img_h}]."
                        ),
                        sample_path=sample_file,
                        defect_type="COCO_BOUNDING_BOX_OUT_OF_BOUNDS",
                        observed_val=max((x + w) / img_w, (y + h) / img_h),
                        threshold_val=1.0,
                        details={
                            "annotation_id": anno_id,
                            "bbox": bbox,
                            "image_width": img_w,
                            "image_height": img_h,
                        },
                    )

    def _validate_yolo(
        self,
        snapshot_dir: Path,
        context: AssessmentContext,
        findings: list[UniversalFinding],
        evidence_list: list[EvidenceRecord],
        now: datetime,
    ) -> None:
        """Validates YOLO normalized center coordinates and extent bounds in [0.0, 1.0]."""
        label_files = list(snapshot_dir.rglob("*.txt"))
        for lbl_path in label_files:
            if lbl_path.name in ("manifest.json", "metadata.json", "labels.txt"):
                continue

            rel_lbl = str(lbl_path.relative_to(snapshot_dir)).replace("\\", "/")
            try:
                lines = [
                    line.strip()
                    for line in lbl_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception:
                continue

            for idx, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) < 5:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"YOLO Label '{lbl_path.name}': Malformed Line {idx}",
                        description=f"Expected class_id and 4 normalized coordinates, got '{line}'.",
                        sample_path=rel_lbl,
                        defect_type="MALFORMED_YOLO_LINE",
                        details={"label_file": rel_lbl, "line": line, "line_number": idx},
                    )
                    continue

                try:
                    xc, yc, w, h = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"YOLO Label '{lbl_path.name}': Non-Numeric Coordinates on Line {idx}",
                        description=f"Coordinates cannot be parsed as floats: '{line}'.",
                        sample_path=rel_lbl,
                        defect_type="NON_NUMERIC_COORDINATES",
                        details={"label_file": rel_lbl, "line": line, "line_number": idx},
                    )
                    continue

                # Dimensions check
                if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"YOLO Label '{lbl_path.name}': Invalid Box Dimensions [w={w}, h={h}]",
                        description=f"Normalized width and height must be in range (0.0, 1.0], got w={w}, h={h}.",
                        sample_path=rel_lbl,
                        defect_type="YOLO_DIMENSION_OUT_OF_RANGE",
                        details={"label_file": rel_lbl, "bbox": [xc, yc, w, h]},
                    )
                    continue

                # Center check
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"YOLO Label '{lbl_path.name}': Center Coordinate Out of [0, 1]",
                        description=f"Box center [xc={xc}, yc={yc}] is outside normalized unit boundary [0, 1].",
                        sample_path=rel_lbl,
                        defect_type="YOLO_CENTER_OUT_OF_RANGE",
                        details={"label_file": rel_lbl, "bbox": [xc, yc, w, h]},
                    )

                # Extent corners check
                x_min = xc - (w / 2.0)
                y_min = yc - (h / 2.0)
                x_max = xc + (w / 2.0)
                y_max = yc + (h / 2.0)

                if x_min < 0.0 or y_min < 0.0 or x_max > 1.0 or y_max > 1.0:
                    excess = max(0.0 - x_min, 0.0 - y_min, x_max - 1.0, y_max - 1.0)
                    self._add_geometry_finding(
                        context,
                        findings,
                        evidence_list,
                        now,
                        title=f"YOLO Label '{lbl_path.name}': Bounding Box Extent Outside [0, 1]",
                        description=(
                            f"Box extent [xmin={x_min:.3f}, ymin={y_min:.3f}, xmax={x_max:.3f}, ymax={y_max:.3f}] "
                            f"exceeds normalized unit canvas boundary [0.0, 1.0]."
                        ),
                        sample_path=rel_lbl,
                        defect_type="YOLO_BOUNDING_BOX_EXTENT_OUT_OF_RANGE",
                        observed_val=excess,
                        threshold_val=0.0,
                        details={
                            "label_file": rel_lbl,
                            "bbox": [xc, yc, w, h],
                            "corners": [x_min, y_min, x_max, y_max],
                        },
                    )

    def _add_geometry_finding(
        self,
        context: AssessmentContext,
        findings: list[UniversalFinding],
        evidence_list: list[EvidenceRecord],
        now: datetime,
        title: str,
        description: str,
        sample_path: str | None,
        defect_type: str,
        details: dict,
        observed_val: float | None = 1.0,
        threshold_val: float | None = 0.0,
    ) -> None:
        """Helper to assemble structured finding and evidence pairs."""
        uid = uuid.uuid4().hex[:12]
        fnd_id = f"fnd_geom_{uid}"
        ev_id = f"ev_geom_{uid}"

        clean_sample_path = sample_path.replace("\\", "/").lstrip("/") if sample_path else None

        ev = EvidenceRecord(
            evidence_id=ev_id,
            evidence_type=EvidenceType.METADATA_MISMATCH,
            finding_id=fnd_id,
            signal_name="annotation_spatial_geometry",
            rationale=description,
            created_at=now,
            sample_reference=clean_sample_path,
            observed_value=observed_val,
            threshold_value=threshold_val,
            metadata=details,
        )

        affected = (
            [
                AffectedAssetReference(
                    asset_id=context.snapshot.asset_id,
                    sample_path=clean_sample_path,
                )
            ]
            if clean_sample_path
            else []
        )

        fnd = UniversalFinding(
            finding_id=fnd_id,
            assessment_id=context.config.get(
                "assessment_id", f"asm_{context.snapshot.snapshot_id}"
            ),
            detector_id=self.detector_id,
            detector_version=self.version,
            category=FindingCategory.STRUCTURAL_DEFECT,
            severity=SeverityLevel.HIGH,
            title=title,
            description=description,
            confidence=1.0,
            reproducibility=ReproducibilityMetadata(
                execution_timestamp=now,
                software_versions={"detector": self.version},
                detector_config={"validator": "geometry_boundary_assertion"},
            ),
            created_at=now,
            suggested_disposition=SuggestedDisposition.REVIEW,
            anomaly_score=1.0,
            affected_assets=affected,
            evidence_ids=[ev_id],
            details=details,
        )

        findings.append(fnd)
        evidence_list.append(ev)
