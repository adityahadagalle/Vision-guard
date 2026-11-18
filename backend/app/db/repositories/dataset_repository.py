"""SQLite repository adapter for datasets and immutable snapshots."""

import json
from datetime import UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import AssetSnapshotModel, DatasetModel
from backend.app.db.session import now_utc
from backend.app.domain.entities.asset import AssetSnapshot
from backend.app.domain.entities.enums import AssetType, IngestionStatus


class SqliteDatasetRepository:
    """Repository managing dataset records and content-addressed asset snapshots in SQLite."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_dataset(
        self,
        dataset_id: str,
        name: str,
        dataset_format: str,
        storage_path: str,
        sha256: str,
        size_bytes: int,
        status: IngestionStatus = IngestionStatus.RECEIVED,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetModel:
        """Persists a new dataset metadata record."""
        now = now_utc()
        model = DatasetModel(
            dataset_id=dataset_id,
            name=name,
            format=dataset_format,
            status=status,
            storage_path=storage_path,
            sha256=sha256,
            size_bytes=size_bytes,
            metadata_json=json.dumps(metadata or {}, sort_keys=True),
            created_at=now,
            updated_at=now,
        )
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return model

    def get_dataset(self, dataset_id: str) -> DatasetModel | None:
        """Retrieves a dataset record by its identifier."""
        stmt = select(DatasetModel).where(DatasetModel.dataset_id == dataset_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def update_dataset_status(
        self,
        dataset_id: str,
        status: IngestionStatus,
    ) -> None:
        """Updates the ingestion status of a dataset."""
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        dataset.status = status
        dataset.updated_at = now_utc()
        self._session.commit()

    def create_snapshot(
        self,
        snapshot_id: str,
        dataset_id: str,
        manifest_sha256: str,
        storage_path: str,
        file_count: int,
        total_size_bytes: int,
        asset_type: AssetType = AssetType.DATASET_SNAPSHOT,
        schema_version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> AssetSnapshot:
        """Persists an immutable asset snapshot record or returns existing record idempotently."""
        existing = self.get_snapshot(snapshot_id)
        if existing is not None:
            return existing

        now = now_utc()
        meta = metadata or {}
        model = AssetSnapshotModel(
            snapshot_id=snapshot_id,
            dataset_id=dataset_id,
            asset_type=asset_type,
            manifest_sha256=manifest_sha256,
            storage_path=storage_path,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            schema_version=schema_version,
            metadata_json=json.dumps(meta, sort_keys=True),
            created_at=now,
        )
        self._session.add(model)
        self._session.commit()

        created_at = (
            model.created_at if model.created_at.tzinfo else model.created_at.replace(tzinfo=UTC)
        )
        return AssetSnapshot(
            snapshot_id=model.snapshot_id,
            asset_id=model.dataset_id,
            asset_type=AssetType(model.asset_type),
            manifest_sha256=model.manifest_sha256,
            created_at=created_at,
            file_count=model.file_count,
            total_size_bytes=model.total_size_bytes,
            schema_version=model.schema_version,
            metadata=meta,
        )

    def get_snapshot(self, snapshot_id: str) -> AssetSnapshot | None:
        """Retrieves and reconstructs an immutable AssetSnapshot domain entity."""
        stmt = select(AssetSnapshotModel).where(AssetSnapshotModel.snapshot_id == snapshot_id)
        model = self._session.execute(stmt).scalar_one_or_none()
        if model is None:
            return None

        metadata = json.loads(model.metadata_json) if model.metadata_json else {}
        created_at = (
            model.created_at if model.created_at.tzinfo else model.created_at.replace(tzinfo=UTC)
        )
        asset_id = model.dataset_id or model.model_id or ""
        return AssetSnapshot(
            snapshot_id=model.snapshot_id,
            asset_id=asset_id,
            asset_type=AssetType(model.asset_type),
            manifest_sha256=model.manifest_sha256,
            created_at=created_at,
            file_count=model.file_count,
            total_size_bytes=model.total_size_bytes,
            schema_version=model.schema_version,
            metadata=metadata,
        )

    def get_snapshot_storage_path(self, snapshot_id: str) -> str | None:
        """Retrieves the physical filesystem path for an asset snapshot."""
        stmt = select(AssetSnapshotModel.storage_path).where(
            AssetSnapshotModel.snapshot_id == snapshot_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_authoritative_manifest_sha256(self, snapshot_id: str) -> str | None:
        """Retrieves the authoritative manifest SHA-256 digest recorded in the database."""
        stmt = select(AssetSnapshotModel.manifest_sha256).where(
            AssetSnapshotModel.snapshot_id == snapshot_id
        )
        return self._session.execute(stmt).scalar_one_or_none()
