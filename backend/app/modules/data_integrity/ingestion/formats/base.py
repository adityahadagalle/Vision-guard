"""Abstract base classes and canonical schemas for dataset format adapters.

Provides a unified representation for bounding boxes, samples, categories,
and annotations across COCO, YOLO, and Folder-per-Class datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NormalizedBBox(BaseModel):
    """Canonical representation of an object detection bounding box."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str
    category_id: int
    bbox: list[float]  # [x, y, width, height]
    coordinate_system: str  # "ABSOLUTE_PIXELS" or "NORMALIZED_0_1"
    area: float | None = None


class NormalizedSample(BaseModel):
    """Canonical representation of a single dataset sample (image + annotations)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    image_relative_path: str
    width: int | None = None
    height: int | None = None
    labels: list[str] = Field(default_factory=list)
    bounding_boxes: list[NormalizedBBox] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDataset(BaseModel):
    """Structured result of parsing and validating an extracted dataset layout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_name: str
    samples: list[NormalizedSample]
    categories: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_warnings: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class BaseFormatAdapter(ABC):
    """Interface for dataset format parsing and validation adapters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Stable uppercase identifier for the format (e.g. 'COCO_DETECTION')."""
        ...

    @abstractmethod
    def can_handle(self, root_dir: Path) -> bool:
        """Heuristically inspects root_dir to determine if it matches this format's layout."""
        ...

    @abstractmethod
    def parse_and_validate(self, root_dir: Path) -> ParsedDataset:
        """Parses annotations, validates structural invariants, and returns a ParsedDataset."""
        ...
