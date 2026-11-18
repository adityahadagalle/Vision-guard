"""Dataset normalization engine.

Standardizes heterogeneous computer-vision datasets into a canonical internal
representation while preserving sample identifiers, relative file paths,
bounding box geometries, and complete validation error traces.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.modules.data_integrity.ingestion.formats.base import (
    NormalizedSample,
    ParsedDataset,
)


class NormalizedDataset(BaseModel):
    """Canonical normalized dataset representation output by ingestion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_format: str
    normalization_version: str = "1.0.0"
    categories: list[str]
    samples: list[NormalizedSample]
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_warnings: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class DatasetNormalizer:
    """Normalizes parsed dataset structures into canonical internal records."""

    @classmethod
    def normalize(
        cls, parsed: ParsedDataset, extra_metadata: dict[str, Any] | None = None
    ) -> NormalizedDataset:
        """Converts ParsedDataset into a NormalizedDataset while preserving all original references.

        Args:
            parsed: The ParsedDataset returned by a format adapter.
            extra_metadata: Optional ingestion context (e.g., upload timestamps, ingestion IDs).

        Returns:
            An immutable NormalizedDataset instance.
        """
        combined_meta = dict(parsed.metadata)
        if extra_metadata:
            combined_meta.update(extra_metadata)

        return NormalizedDataset(
            original_format=parsed.format_name,
            normalization_version="1.0.0",
            categories=sorted(set(parsed.categories)),
            samples=parsed.samples,
            metadata=combined_meta,
            validation_warnings=parsed.validation_warnings,
            validation_errors=parsed.validation_errors,
        )
