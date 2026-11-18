"""Model integrity detectors package.

Registers Model Integrity Engine detectors:
1. model_integrity.artifact_hash
2. model_integrity.checkpoint_structure
3. model_integrity.state_dict_integrity
4. model_integrity.architecture_compatibility
5. model_integrity.behavioral_fingerprint
6. model_integrity.prediction_consistency
7. model_integrity.input_sensitivity
8. model_integrity.backdoor_indicator
9. model_integrity.reference_comparison
10. model_integrity.spectral_signatures
11. model_integrity.activation_clustering
"""

from backend.app.domain.registry.detector_registry import DetectorRegistry
from backend.app.modules.model_integrity.detectors.activation_clustering import (
    ActivationClusteringDetector,
)
from backend.app.modules.model_integrity.detectors.architecture_compatibility import (
    ArchitectureCompatibilityDetector,
)
from backend.app.modules.model_integrity.detectors.artifact_hash import (
    ModelArtifactHashDetector,
)
from backend.app.modules.model_integrity.detectors.backdoor_indicator import (
    BackdoorIndicatorDetector,
)
from backend.app.modules.model_integrity.detectors.behavioral_fingerprint import (
    BehavioralFingerprintDetector,
)
from backend.app.modules.model_integrity.detectors.checkpoint_structure import (
    CheckpointStructureDetector,
)
from backend.app.modules.model_integrity.detectors.input_sensitivity import (
    InputSensitivityDetector,
)
from backend.app.modules.model_integrity.detectors.prediction_consistency import (
    PredictionConsistencyDetector,
)
from backend.app.modules.model_integrity.detectors.reference_comparison import (
    ReferenceComparisonDetector,
)
from backend.app.modules.model_integrity.detectors.spectral_signatures import (
    SpectralSignaturesDetector,
)
from backend.app.modules.model_integrity.detectors.state_dict_integrity import (
    StateDictIntegrityDetector,
)


def register_model_integrity_detectors(registry: DetectorRegistry) -> None:
    """Registers all Model Integrity detectors into the central DetectorRegistry."""
    detectors = [
        ModelArtifactHashDetector(),
        CheckpointStructureDetector(),
        StateDictIntegrityDetector(),
        ArchitectureCompatibilityDetector(),
        BehavioralFingerprintDetector(),
        PredictionConsistencyDetector(),
        InputSensitivityDetector(),
        BackdoorIndicatorDetector(),
        ReferenceComparisonDetector(),
        SpectralSignaturesDetector(),
        ActivationClusteringDetector(),
    ]
    for d in detectors:
        if not registry.has(d.detector_id):
            registry.register(d)


__all__ = [
    "ModelArtifactHashDetector",
    "CheckpointStructureDetector",
    "StateDictIntegrityDetector",
    "ArchitectureCompatibilityDetector",
    "BehavioralFingerprintDetector",
    "PredictionConsistencyDetector",
    "InputSensitivityDetector",
    "BackdoorIndicatorDetector",
    "ReferenceComparisonDetector",
    "SpectralSignaturesDetector",
    "ActivationClusteringDetector",
    "register_model_integrity_detectors",
]
