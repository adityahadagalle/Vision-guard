"""Configuration settings for secure model behavioral probing and backdoor analysis.

Enforces resource limits, probe batch sizes, perturbation bounds, and execution timeouts.
"""

from pydantic import BaseModel, ConfigDict, Field


class BehavioralConfig(BaseModel):
    """Configuration governing model behavioral inference, sensitivity, and backdoor probing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Probe suite settings
    probe_count: int = Field(
        default=100,
        description="Number of deterministic synthetic input probes evaluated.",
    )
    probe_batch_size: int = Field(
        default=10,
        description="Batch size for worker inference forward passes.",
    )
    probe_seed: int = Field(
        default=42,
        description="Deterministic PRNG seed for isolated probe generation.",
    )
    probe_input_shape: list[int] = Field(
        default_factory=lambda: [3, 32, 32],
        description="Default input tensor shape (C, H, W) for reference probes.",
    )

    # Sensitivity probing
    sensitivity_epsilon: float = Field(
        default=0.02,
        description="Bounded perturbation magnitude (L_inf norm in [0, 1] range).",
    )
    sensitivity_flip_threshold: float = Field(
        default=0.40,
        description="Prediction flip rate threshold above which model is flagged as highly sensitive.",
    )

    # Consistency evaluation
    consistency_repetitions: int = Field(
        default=3,
        description="Number of repeated inference evaluations under fixed eval conditions.",
    )
    consistency_tolerance: float = Field(
        default=1e-4,
        description="Numerical tolerance for floating-point logit comparison.",
    )

    # Backdoor trigger evaluation
    backdoor_trigger_size: int = Field(
        default=4,
        description="Width and height in pixels for candidate trigger patches.",
    )
    backdoor_concentration_threshold: float = Field(
        default=0.85,
        description="Target-class concentration ratio threshold indicating suspicious trigger hijacking.",
    )
    backdoor_confidence_threshold: float = Field(
        default=0.80,
        description="Average confidence threshold on the concentrated target class.",
    )
    backdoor_delta_threshold: float = Field(
        default=0.50,
        description="Minimum concentration increase over clean baseline to flag suspicious trigger behavior.",
    )

    # Reference model comparison
    reference_agreement_threshold: float = Field(
        default=0.95,
        description="Minimum acceptable top-1 prediction agreement with trusted reference model.",
    )
    reference_divergence_threshold: float = Field(
        default=1e-4,
        description="Maximum permissible logit divergence before noting numerical deviation.",
    )

    # Subprocess execution timeout
    worker_timeout_seconds: float = Field(
        default=60.0,
        description="Hard execution timeout for behavioral worker subprocess in seconds.",
    )
