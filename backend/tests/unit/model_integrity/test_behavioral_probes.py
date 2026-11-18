"""Unit tests for deterministic probe generation, perturbations, and trigger synthesis."""

import torch

from backend.app.modules.model_integrity.behavioral.config import BehavioralConfig
from backend.app.modules.model_integrity.behavioral.probes import (
    apply_bounded_perturbation,
    apply_trigger,
    compute_behavioral_fingerprint,
    generate_deterministic_probes,
    get_candidate_triggers,
)


def test_deterministic_probe_generation() -> None:
    """Verifies that probe generation is 100% reproducible with fixed seeds."""
    p1 = generate_deterministic_probes(10, [3, 32, 32], seed=42)
    p2 = generate_deterministic_probes(10, [3, 32, 32], seed=42)
    p3 = generate_deterministic_probes(10, [3, 32, 32], seed=999)

    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)
    assert p1.shape == (10, 3, 32, 32)
    assert (p1 >= 0.0).all() and (p1 <= 1.0).all()


def test_probe_generation_does_not_mutate_global_torch_seed() -> None:
    """Verifies that isolated torch.Generator prevents global PRNG contamination."""
    torch.manual_seed(12345)
    r1 = torch.rand(5)

    torch.manual_seed(12345)
    # Generate probes using isolated generator
    _ = generate_deterministic_probes(20, [3, 16, 16], seed=99)
    # Global stream should match r1 exactly
    r2 = torch.rand(5)

    assert torch.equal(r1, r2)


def test_apply_bounded_perturbation_bounds() -> None:
    """Verifies that perturbation remains bounded within [-epsilon, epsilon] and [0, 1]."""
    probes = generate_deterministic_probes(5, [3, 16, 16], seed=42)
    epsilon = 0.03
    perturbed, deltas = apply_bounded_perturbation(probes, epsilon=epsilon, seed=42)

    assert (perturbed >= 0.0).all() and (perturbed <= 1.0).all()
    assert (deltas.abs() <= epsilon + 1e-6).all()


def test_candidate_triggers_synthesis_and_application() -> None:
    """Verifies that candidate triggers are correctly constructed and applied."""
    input_shape = [3, 32, 32]
    triggers = get_candidate_triggers(input_shape, trigger_size=4)

    assert "corner_checkerboard" in triggers
    assert "corner_square" in triggers
    assert "center_cross" in triggers
    assert "blended_watermark" in triggers

    probes = torch.zeros((2, 3, 32, 32), dtype=torch.float32)

    for _name, spec in triggers.items():
        trig_out = apply_trigger(probes, spec)
        assert trig_out.shape == probes.shape
        assert (trig_out >= 0.0).all() and (trig_out <= 1.0).all()
        # Trigger should have non-zero elements
        assert trig_out.sum() > 0


def test_compute_behavioral_fingerprint_determinism() -> None:
    """Verifies canonical behavioral fingerprint formatting and order invariance."""
    records = [
        {"index": 1, "predicted_class": 0, "confidence": 0.954321, "logit_digest": "abc"},
        {"index": 0, "predicted_class": 3, "confidence": 0.812345, "logit_digest": "xyz"},
    ]
    # Invert input order
    records_inverted = list(reversed(records))

    fp1 = compute_behavioral_fingerprint(records)
    fp2 = compute_behavioral_fingerprint(records_inverted)

    assert len(fp1) == 64
    assert fp1 == fp2


def test_behavioral_config_default_probe_count_is_100() -> None:
    """Verifies that default probe count is scaled to M=100 while supporting explicit overrides."""
    default_config = BehavioralConfig()
    assert default_config.probe_count == 100

    custom_config = BehavioralConfig(probe_count=42)
    assert custom_config.probe_count == 42


def test_deterministic_probe_suite_scaling_m_100() -> None:
    """Verifies deterministic generation and bounded values for scaled M=100 probe suite."""
    rng_state_before = torch.random.get_rng_state()

    probes_a = generate_deterministic_probes(100, [3, 32, 32], seed=42)
    probes_b = generate_deterministic_probes(100, [3, 32, 32], seed=42)
    probes_different_seed = generate_deterministic_probes(100, [3, 32, 32], seed=1337)

    rng_state_after = torch.random.get_rng_state()

    assert probes_a.shape == (100, 3, 32, 32)
    assert torch.equal(probes_a, probes_b)
    assert not torch.equal(probes_a, probes_different_seed)
    assert (probes_a >= 0.0).all() and (probes_a <= 1.0).all()
    # Global RNG state must remain completely unaffected by isolated probe generation
    assert torch.equal(rng_state_before, rng_state_after)
