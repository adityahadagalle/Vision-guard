"""COCO Detection dataset format adapter.

Parses instances_*.json annotations, verifies absolute pixel bounding boxes [x, y, w, h],
links annotations to image records, and tracks annotation validation defects.
"""

import json
from pathlib import Path

from backend.app.modules.data_integrity.ingestion.exceptions import MalformedAnnotationError
from backend.app.modules.data_integrity.ingestion.formats.base import (
    BaseFormatAdapter,
    NormalizedBBox,
    NormalizedSample,
    ParsedDataset,
)


class COCODatasetAdapter(BaseFormatAdapter):
    """Adapter for COCO-formatted object detection datasets."""

    @property
    def format_name(self) -> str:
        return "COCO_DETECTION"

    def can_handle(self, root_dir: Path) -> bool:
        """Determines if root_dir contains valid COCO annotation candidates."""
        candidates = self._find_annotation_files(root_dir)
        for cand in candidates:
            try:
                with cand.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(
                    data.keys()
                ):
                    return True
            except Exception:
                continue
        return False

    def parse_and_validate(self, root_dir: Path) -> ParsedDataset:
        """Parses COCO annotations and validates bounding box geometries."""
        candidates = self._find_annotation_files(root_dir)
        annotation_file = None

        for cand in candidates:
            try:
                with cand.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(
                    data.keys()
                ):
                    annotation_file = cand
                    break
            except Exception:
                continue

        if not annotation_file:
            raise MalformedAnnotationError(
                f"No valid COCO JSON annotation file found in '{root_dir}'.",
                details={"root_dir": str(root_dir)},
            )

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Parse categories
        raw_cats = data.get("categories", [])
        cat_map: dict[int, str] = {}
        for cat in raw_cats:
            if isinstance(cat, dict) and "id" in cat and "name" in cat:
                cat_map[cat["id"]] = str(cat["name"])

        categories = sorted(cat_map.values())

        # 2. Parse images
        raw_images = data.get("images", [])
        image_map: dict[int, dict] = {}
        for img in raw_images:
            if isinstance(img, dict) and "id" in img and "file_name" in img:
                image_map[img["id"]] = img

        # 3. Parse annotations
        raw_annos = data.get("annotations", [])
        image_annos: dict[int, list[dict]] = {img_id: [] for img_id in image_map}
        warnings: list[str] = []
        errors: list[str] = []

        for anno in raw_annos:
            if not isinstance(anno, dict):
                errors.append(f"Invalid annotation entry: not a JSON object: {anno}")
                continue

            img_id = anno.get("image_id")
            if img_id not in image_map:
                errors.append(
                    f"Annotation {anno.get('id')} references non-existent image_id: {img_id}"
                )
                continue

            image_annos[img_id].append(anno)

        # 4. Construct normalized samples
        samples: list[NormalizedSample] = []
        for img_id, img_info in image_map.items():
            file_name = img_info.get("file_name", "")
            img_w = img_info.get("width")
            img_h = img_info.get("height")

            # Resolve image relative path (check images/ or root)
            rel_path = self._resolve_image_path(root_dir, file_name)

            sample_bboxes: list[NormalizedBBox] = []
            sample_labels: set[str] = set()

            for anno in image_annos.get(img_id, []):
                cat_id = anno.get("category_id")
                cat_name = cat_map.get(cat_id, f"unknown_class_{cat_id}")
                sample_labels.add(cat_name)

                bbox = anno.get("bbox")
                if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"Image {file_name}: malformed bbox structure: {bbox}")
                    continue

                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    errors.append(
                        f"Image {file_name}: invalid non-positive bbox dimension [w={w}, h={h}]"
                    )
                    continue

                if x < 0 or y < 0:
                    warnings.append(f"Image {file_name}: negative bbox coordinates [x={x}, y={y}]")

                sample_bboxes.append(
                    NormalizedBBox(
                        category=cat_name,
                        category_id=cat_id,
                        bbox=[float(x), float(y), float(w), float(h)],
                        coordinate_system="ABSOLUTE_PIXELS",
                        area=float(anno.get("area", w * h)),
                    )
                )

            samples.append(
                NormalizedSample(
                    sample_id=f"coco_{img_id}",
                    image_relative_path=rel_path,
                    width=img_w,
                    height=img_h,
                    labels=sorted(sample_labels),
                    bounding_boxes=sample_bboxes,
                    metadata={"coco_image_id": img_id, "original_file_name": file_name},
                )
            )

        return ParsedDataset(
            format_name=self.format_name,
            samples=samples,
            categories=categories,
            metadata={
                "annotation_file": str(annotation_file.relative_to(root_dir)).replace("\\", "/"),
                "total_images": len(samples),
                "total_annotations": len(raw_annos),
            },
            validation_warnings=warnings,
            validation_errors=errors,
        )

    @staticmethod
    def _find_annotation_files(root_dir: Path) -> list[Path]:
        """Locates candidate JSON annotation files in annotations/ or root."""
        candidates: list[Path] = []
        anno_dir = root_dir / "annotations"
        if anno_dir.exists() and anno_dir.is_dir():
            candidates.extend(sorted(anno_dir.glob("*.json")))
        candidates.extend(sorted(root_dir.glob("*.json")))
        return candidates

    @staticmethod
    def _resolve_image_path(root_dir: Path, file_name: str) -> str:
        """Finds whether the image is in images/, data/, or directly in root."""
        clean_name = file_name.replace("\\", "/").lstrip("/")
        for prefix in ["images", "data", "train", "val", "test", ""]:
            cand = root_dir / prefix / clean_name if prefix else root_dir / clean_name
            if cand.exists():
                rel = cand.relative_to(root_dir)
                return str(rel).replace("\\", "/")
        # Default to images/ prefix or clean_name
        if (root_dir / "images").exists():
            return f"images/{clean_name}"
        return clean_name
