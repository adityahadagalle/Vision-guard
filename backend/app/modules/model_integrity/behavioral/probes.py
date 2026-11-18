"""Deterministic probe generation, perturbation transforms, and candidate trigger synthesis.

Provides isolated PRNG generators, bounded perturbations, and canonical fingerprinting
without mutating global process random seeds.
"""

import hashlib
from typing import Any

import torch


def generate_deterministic_probes(
    count: int,
    input_shape: list[int],
    seed: int = 42,
) -> torch.Tensor:
    """Generates synthetic reference input probes with values in [0.0, 1.0].

    Uses an isolated torch.Generator seeded with an explicit value, preventing
    global random state contamination.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(count, *input_shape, generator=gen, dtype=torch.float32)


def apply_bounded_perturbation(
    probes: torch.Tensor,
    epsilon: float,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies bounded uniform noise in [-epsilon, epsilon] clamped to [0.0, 1.0].

    Returns:
        tuple of (perturbed_probes, applied_deltas).
    """
    gen = torch.Generator().manual_seed(seed)
    # Uniform random in [-epsilon, epsilon]
    noise = (torch.rand(probes.shape, generator=gen, dtype=probes.dtype) * 2.0 - 1.0) * epsilon
    perturbed = torch.clamp(probes + noise, 0.0, 1.0)
    actual_delta = perturbed - probes
    return perturbed, actual_delta


def get_candidate_triggers(
    input_shape: list[int],
    trigger_size: int = 4,
) -> dict[str, dict[str, Any]]:
    """Synthesizes standard candidate backdoor trigger specifications for visual models.

    Args:
        input_shape: [C, H, W] of input tensors.
        trigger_size: Side length of candidate trigger patterns.

    Returns:
        Dictionary mapping trigger_name to trigger specification metadata.
    """
    channels, height, width = input_shape
    ts = min(trigger_size, height, width)

    triggers: dict[str, dict[str, Any]] = {}

    # 1. Corner Checkerboard (bottom-right)
    r_start = max(0, height - ts)
    c_start = max(0, width - ts)
    triggers["corner_checkerboard"] = {
        "type": "patch",
        "name": "corner_checkerboard",
        "r_start": r_start,
        "r_end": height,
        "c_start": c_start,
        "c_end": width,
        "pattern": "checkerboard",
    }

    # 2. Corner Square (top-left solid white patch)
    triggers["corner_square"] = {
        "type": "patch",
        "name": "corner_square",
        "r_start": 0,
        "r_end": ts,
        "c_start": 0,
        "c_end": ts,
        "pattern": "solid",
        "value": 1.0,
    }

    # 3. Center Cross (centered plus symbol)
    mid_r = height // 2
    mid_c = width // 2
    r_half = ts // 2
    c_half = ts // 2
    triggers["center_cross"] = {
        "type": "patch",
        "name": "center_cross",
        "r_start": max(0, mid_r - r_half),
        "r_end": min(height, mid_r + r_half),
        "c_start": max(0, mid_c - c_half),
        "c_end": min(width, mid_c + c_half),
        "pattern": "cross",
    }

    # 4. Blended Watermark (low-opacity periodic grid)
    triggers["blended_watermark"] = {
        "type": "blended",
        "name": "blended_watermark",
        "alpha": 0.15,
        "frequency": 4,
    }

    return triggers


def apply_trigger(probes: torch.Tensor, trigger_spec: dict[str, Any]) -> torch.Tensor:
    """Applies a candidate trigger pattern to a batch of probes."""
    triggered = probes.clone()
    t_type = trigger_spec.get("type", "patch")

    if t_type == "patch":
        r0 = trigger_spec["r_start"]
        r1 = trigger_spec["r_end"]
        c0 = trigger_spec["c_start"]
        c1 = trigger_spec["c_end"]
        pattern = trigger_spec.get("pattern", "solid")

        if pattern == "solid":
            val = trigger_spec.get("value", 1.0)
            triggered[:, :, r0:r1, c0:c1] = val

        elif pattern == "checkerboard":
            h = r1 - r0
            w = c1 - c0
            grid = (torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2 == 0
            grid_val = grid.float()
            triggered[:, :, r0:r1, c0:c1] = grid_val

        elif pattern == "cross":
            h = r1 - r0
            w = c1 - c0
            cross = torch.zeros((h, w), dtype=torch.float32)
            cross[h // 2, :] = 1.0
            cross[:, w // 2] = 1.0
            triggered[:, :, r0:r1, c0:c1] = cross

    elif t_type == "blended":
        alpha = trigger_spec.get("alpha", 0.15)
        freq = trigger_spec.get("frequency", 4)
        _, _, h, w = probes.shape
        grid = (
            torch.sin(torch.linspace(0, 3.14159 * freq, h))[:, None]
            * torch.sin(torch.linspace(0, 3.14159 * freq, w))[None, :]
        ).abs()
        grid = grid.unsqueeze(0).unsqueeze(0).to(probes.device)
        triggered = torch.clamp((1.0 - alpha) * triggered + alpha * grid, 0.0, 1.0)

    return torch.clamp(triggered, 0.0, 1.0)


def compute_behavioral_fingerprint(probe_records: list[dict[str, Any]]) -> str:
    """Builds a deterministic, canonical SHA-256 behavioral fingerprint from probe predictions.

    Args:
        probe_records: List of dictionaries with keys:
            'index', 'predicted_class', 'confidence', 'logit_digest'

    Returns:
        Hex-encoded SHA-256 digest representing canonical behavioral fingerprint.
    """
    sorted_records = sorted(probe_records, key=lambda x: int(x["index"]))
    lines: list[str] = []
    for r in sorted_records:
        idx = int(r["index"])
        pred_class = int(r["predicted_class"])
        conf = float(r["confidence"])
        ldigest = str(r.get("logit_digest", ""))
        lines.append(f"p_{idx:04d}:{pred_class}:{conf:.6f}:{ldigest}")

    canonical_repr = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(canonical_repr).hexdigest()
