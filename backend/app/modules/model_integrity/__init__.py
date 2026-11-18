"""Model Integrity module for VisionGuard.

Provides PyTorch artifact ingestion, safe isolated checkpoint inspection,
state-dict structural validation, and trusted architecture compatibility checks.
"""

from backend.app.modules.model_integrity.architecture.registry import (
    ArchitectureSpec,
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.artifact import (
    ModelArtifactManager,
    ModelIngestionResult,
)
from backend.app.modules.model_integrity.behavioral import (
    BehavioralConfig,
    BehavioralReport,
    SubprocessBehavioralWorker,
)
from backend.app.modules.model_integrity.config import ModelIntegrityConfig
from backend.app.modules.model_integrity.conflict import (
    ConflictType,
    ModelConflictEngine,
)
from backend.app.modules.model_integrity.detectors import (
    ArchitectureCompatibilityDetector,
    BackdoorIndicatorDetector,
    BehavioralFingerprintDetector,
    CheckpointStructureDetector,
    InputSensitivityDetector,
    ModelArtifactHashDetector,
    PredictionConsistencyDetector,
    ReferenceComparisonDetector,
    StateDictIntegrityDetector,
    register_model_integrity_detectors,
)
from backend.app.modules.model_integrity.inspection.worker import (
    CheckpointInspectionReport,
    HostIsolationAssessor,
    IsolationStatus,
    SubprocessModelWorker,
)

__all__ = [
    "ModelIntegrityConfig",
    "ModelArtifactManager",
    "ModelIngestionResult",
    "SubprocessModelWorker",
    "CheckpointInspectionReport",
    "IsolationStatus",
    "HostIsolationAssessor",
    "TrustedArchitectureRegistry",
    "ArchitectureSpec",
    "ModelArtifactHashDetector",
    "CheckpointStructureDetector",
    "StateDictIntegrityDetector",
    "ArchitectureCompatibilityDetector",
    "BehavioralFingerprintDetector",
    "PredictionConsistencyDetector",
    "InputSensitivityDetector",
    "BackdoorIndicatorDetector",
    "ReferenceComparisonDetector",
    "BehavioralConfig",
    "BehavioralReport",
    "SubprocessBehavioralWorker",
    "ConflictType",
    "ModelConflictEngine",
    "register_model_integrity_detectors",
]
