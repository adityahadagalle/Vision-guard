"""YOLO Detection dataset format adapter.

Parses dataset.yaml or images/ + labels/ paired directory structures,
validates normalized [x_center, y_center, w, h] coordinates in [0.0, 1.0],
and distinguishes empty label files (valid negatives) from missing label files.
"""

import contextlib
from pathlib import Path

from backend.app.modules.data_integrity.ingestion.exceptions import MalformedAnnotationError
from backend.app.modules.data_integrity.ingestion.formats.base import (
    BaseFormatAdapter,
    NormalizedBBox,
    NormalizedSample,
    ParsedDataset,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODatasetAdapter(BaseFormatAdapter):
    """Adapter for YOLO-formatted object detection datasets."""

    @property
    def format_name(self) -> str:
        return "YOLO_DETECTION"

    def can_handle(self, root_dir: Path) -> bool:
        """Determines if root_dir matches YOLO directory conventions."""
        # 1. Check for YAML configuration file
        for yaml_name in ["data.yaml", "dataset.yaml", "yolo.yaml"]:
            if (root_dir / yaml_name).exists():
                return True

        # 2. Check for paired images/ and labels/ directories
        img_dir = root_dir / "images"
        lbl_dir = root_dir / "labels"
        return bool(img_dir.exists() and img_dir.is_dir() and lbl_dir.exists() and lbl_dir.is_dir())

    def parse_and_validate(self, root_dir: Path) -> ParsedDataset:
        """Parses YOLO label files and pairs them with image assets."""
        cat_names = self._parse_yaml_classes(root_dir)

        # Locate images
        image_files: list[Path] = []
        for ext in _IMAGE_EXTS:
            image_files.extend(root_dir.rglob(f"*{ext}"))
            image_files.extend(root_dir.rglob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        if not image_files:
            raise MalformedAnnotationError(
                f"No image files found in YOLO dataset directory '{root_dir}'.",
                details={"root_dir": str(root_dir)},
            )

        samples: list[NormalizedSample] = []
        warnings: list[str] = []
        errors: list[str] = []
        discovered_classes: set[str] = set(cat_names.values())

        for img_path in image_files:
            rel_img = str(img_path.relative_to(root_dir)).replace("\\", "/")
            label_path = self._find_label_for_image(root_dir, img_path)

            sample_bboxes: list[NormalizedBBox] = []
            sample_labels: set[str] = set()

            if label_path is None or not label_path.exists():
                warnings.append(
                    f"Image '{rel_img}' has no corresponding label file (unannotated/orphan)."
                )
            else:
                # Read YOLO label lines
                with label_path.open("r", encoding="utf-8") as lf:
                    lines = [line.strip() for line in lf if line.strip()]

                # An empty file is a valid negative sample
                if not lines:
                    pass  # Valid negative sample

                for line_idx, line in enumerate(lines, start=1):
                    parts = line.split()
                    if len(parts) < 5:
                        errors.append(
                            f"Label '{label_path.name}' line {line_idx}: malformed entry '{line}'."
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        errors.append(
                            f"Label '{label_path.name}' line {line_idx}: non-numeric values in '{line}'."
                        )
                        continue

                    class_name = cat_names.get(class_id, f"class_{class_id}")
                    sample_labels.add(class_name)
                    discovered_classes.add(class_name)

                    # Bounding box range validation in [0.0, 1.0]
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                        warnings.append(
                            f"Label '{label_path.name}' line {line_idx}: center coordinates outside [0, 1]: ({xc}, {yc})"
                        )
                    if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                        errors.append(
                            f"Label '{label_path.name}' line {line_idx}: invalid width/height [w={w}, h={h}]"
                        )
                        continue

                    sample_bboxes.append(
                        NormalizedBBox(
                            category=class_name,
                            category_id=class_id,
                            bbox=[xc, yc, w, h],
                            coordinate_system="NORMALIZED_0_1",
                            area=w * h,
                        )
                    )

            # Derive deterministic, unique sample_id from normalized dataset-relative image path
            clean_rel = rel_img.rsplit(".", 1)[0]
            sample_id = f"yolo_{clean_rel.replace('/', '_')}"

            # Detect split (e.g., train, val, test) if present in directory hierarchy
            detected_split: str | None = None
            for part in clean_rel.split("/"):
                if part.lower() in ("train", "val", "validation", "test"):
                    detected_split = part.lower()
                    break

            samples.append(
                NormalizedSample(
                    sample_id=sample_id,
                    image_relative_path=rel_img,
                    labels=sorted(sample_labels),
                    bounding_boxes=sample_bboxes,
                    metadata={
                        "has_label_file": label_path is not None and label_path.exists(),
                        "split": detected_split,
                        "relative_image_path": rel_img,
                    },
                )
            )

        categories = sorted(discovered_classes)

        return ParsedDataset(
            format_name=self.format_name,
            samples=samples,
            categories=categories,
            metadata={"total_samples": len(samples), "classes": categories},
            validation_warnings=warnings,
            validation_errors=errors,
        )

    @staticmethod
    def _parse_yaml_classes(root_dir: Path) -> dict[int, str]:
        """Simple safe YAML parsing for class names without external pyyaml dependencies."""
        cat_map: dict[int, str] = {}
        for yname in ["data.yaml", "dataset.yaml", "yolo.yaml"]:
            ypath = root_dir / yname
            if ypath.exists():
                with ypath.open("r", encoding="utf-8") as f:
                    in_names = False
                    for line in f:
                        line = line.strip()
                        if line.startswith("names:"):
                            in_names = True
                            # Check if names is a list on same line: names: ['cat', 'dog']
                            after = line[len("names:") :].strip()
                            if after.startswith("[") and after.endswith("]"):
                                items = [
                                    item.strip(" '\"")
                                    for item in after[1:-1].split(",")
                                    if item.strip()
                                ]
                                for idx, name in enumerate(items):
                                    cat_map[idx] = name
                                break
                            continue
                        if in_names:
                            if not line or line.startswith("#"):
                                continue
                            if ":" in line:
                                k, v = line.split(":", 1)
                                with contextlib.suppress(ValueError):
                                    cat_map[int(k.strip())] = v.strip(" '\"")
                            elif line.startswith("- "):
                                cat_map[len(cat_map)] = line[2:].strip(" '\"")
                            else:
                                break
        return cat_map

    @staticmethod
    def _find_label_for_image(root_dir: Path, img_path: Path) -> Path | None:
        """Finds matching .txt label file for an image across YOLO folder conventions."""
        stem = img_path.stem
        # 1. Directly alongside image: images/001.jpg -> images/001.txt
        same_dir_txt = img_path.with_suffix(".txt")
        if same_dir_txt.exists():
            return same_dir_txt

        # 2. Swapping 'images' directory component for 'labels': images/train/001.jpg -> labels/train/001.txt
        rel_str = str(img_path.relative_to(root_dir)).replace("\\", "/")
        if "images/" in rel_str:
            lbl_rel = rel_str.replace("images/", "labels/", 1)
            lbl_path = (root_dir / lbl_rel).with_suffix(".txt")
            if lbl_path.exists():
                return lbl_path

        # 3. labels/ directory flat search
        lbl_flat = root_dir / "labels" / f"{stem}.txt"
        if lbl_flat.exists():
            return lbl_flat

        return None
