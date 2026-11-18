"""VisionGuard database persistence module.

Provides SQLAlchemy models, SQLite connection lifecycle, and repository implementations.
"""

from backend.app.db.models import (
    AssessmentModel,
    AssetSnapshotModel,
    Base,
    DatasetModel,
    DetectorRunModel,
    EvidenceRecordModel,
    FindingAffectedAssetModel,
    FindingModel,
    JobModel,
)
from backend.app.db.session import (
    get_db,
    get_engine,
    get_session_factory,
    now_utc,
)

__all__ = [
    "Base",
    "DatasetModel",
    "AssetSnapshotModel",
    "AssessmentModel",
    "DetectorRunModel",
    "FindingModel",
    "EvidenceRecordModel",
    "FindingAffectedAssetModel",
    "JobModel",
    "get_engine",
    "get_session_factory",
    "get_db",
    "now_utc",
]
