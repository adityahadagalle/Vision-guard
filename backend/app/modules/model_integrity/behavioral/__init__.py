"""Model behavioral analysis and backdoor indicator detection package.

Exports configuration, probe utilities, subprocess worker, and report schemas.
"""

from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.behavioral.probes import (
    apply_bounded_perturbation,
    apply_trigger,
    compute_behavioral_fingerprint,
    generate_deterministic_probes,
    get_candidate_triggers,
)
from backend.app.modules.model_integrity.behavioral.worker import (
    BehavioralReport,
    SubprocessBehavioralWorker,
)

__all__ = [
    "BehavioralConfig",
    "BehavioralReport",
    "SubprocessBehavioralWorker",
    "generate_deterministic_probes",
    "apply_bounded_perturbation",
    "get_candidate_triggers",
    "apply_trigger",
    "compute_behavioral_fingerprint",
]
