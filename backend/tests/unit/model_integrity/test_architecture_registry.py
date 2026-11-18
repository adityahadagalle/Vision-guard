"""Unit tests for TrustedArchitectureRegistry and architecture compatibility checks."""

import pytest
import torch
import torch.nn as nn
import torchvision.models as models

from backend.app.modules.model_integrity.architecture.registry import (
    SimpleCNN,
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.exceptions import UnknownArchitectureError


@pytest.fixture
def registry() -> TrustedArchitectureRegistry:
    return TrustedArchitectureRegistry()


def test_registry_contains_default_architectures(
    registry: TrustedArchitectureRegistry,
) -> None:
    """Verifies that reference architectures are statically registered."""
    arch_ids = registry.list_ids()
    assert "simple_cnn" in arch_ids
    assert "resnet18" in arch_ids
    assert "mobilenet_v2" in arch_ids


def test_registry_unknown_architecture_raises(
    registry: TrustedArchitectureRegistry,
) -> None:
    """Verifies that requesting an unregistered architecture raises UnknownArchitectureError."""
    with pytest.raises(UnknownArchitectureError) as exc_info:
        registry.get("non_existent_transformer_v99")
    assert "non_existent_transformer_v99" in exc_info.value.message


def test_compatibility_exact_match(registry: TrustedArchitectureRegistry) -> None:
    """Verifies that an authentic SimpleCNN state-dict yields a 1.0 compatibility score."""
    model = SimpleCNN()
    candidate_tensors = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()
    }

    report = registry.validate_compatibility("simple_cnn", candidate_tensors)

    assert report.compatible is True
    assert report.compatibility_score == 1.0
    assert len(report.missing_layers) == 0
    assert len(report.unexpected_layers) == 0
    assert len(report.shape_mismatches) == 0
    assert len(report.dtype_mismatches) == 0


def test_compatibility_missing_layer(registry: TrustedArchitectureRegistry) -> None:
    """Verifies that missing layers are identified and degrade compatibility score."""
    model = SimpleCNN()
    candidate_tensors = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()
    }
    # Remove final classification layer
    del candidate_tensors["fc2.weight"]

    report = registry.validate_compatibility("simple_cnn", candidate_tensors)

    assert report.compatible is False
    assert "fc2.weight" in report.missing_layers
    assert report.compatibility_score < 1.0


def test_compatibility_unexpected_layer(registry: TrustedArchitectureRegistry) -> None:
    """Verifies that extraneous or Trojan backdoor layers are identified."""
    model = SimpleCNN()
    candidate_tensors = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()
    }
    # Inject unauthorized extra layer
    candidate_tensors["backdoor_trigger_probe.weight"] = {
        "shape": [1, 1],
        "dtype": "torch.float32",
    }

    report = registry.validate_compatibility("simple_cnn", candidate_tensors)

    assert report.compatible is False
    assert "backdoor_trigger_probe.weight" in report.unexpected_layers
    assert report.compatibility_score < 1.0


def test_compatibility_shape_mismatch(registry: TrustedArchitectureRegistry) -> None:
    """Verifies that parameter dimension alterations are detected."""
    model = SimpleCNN()
    candidate_tensors = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()
    }
    # Alter shape of fc2 (e.g. 100 classes instead of 10)
    candidate_tensors["fc2.weight"]["shape"] = [100, 64]

    report = registry.validate_compatibility("simple_cnn", candidate_tensors)

    assert report.compatible is False
    assert len(report.shape_mismatches) == 1
    assert report.shape_mismatches[0]["layer"] == "fc2.weight"


def test_compatibility_dtype_mismatch(registry: TrustedArchitectureRegistry) -> None:
    """Verifies that parameter data type changes (e.g. float16 vs float32) are detected."""
    model = SimpleCNN()
    candidate_tensors = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()
    }
    candidate_tensors["fc1.weight"]["dtype"] = "torch.float16"

    report = registry.validate_compatibility("simple_cnn", candidate_tensors)

    assert report.compatible is False
    assert len(report.dtype_mismatches) == 1
    assert report.dtype_mismatches[0]["layer"] == "fc1.weight"


def test_extract_features_simple_cnn(registry: TrustedArchitectureRegistry) -> None:
    """Verifies SimpleCNN latent feature extraction shape, head consistency, and registry access."""
    model = registry.create_instance("simple_cnn")
    assert not model.training

    x = torch.randn(4, 3, 32, 32)
    feats = model.extract_features(x)
    assert feats.shape == (4, 64)

    logits_direct = model(x)
    logits_from_head = model.fc2(feats)
    assert torch.allclose(logits_direct, logits_from_head, atol=1e-5)

    feats_from_registry = registry.extract_features(model, "simple_cnn", x)
    assert torch.equal(feats, feats_from_registry)
    assert registry.get_feature_dimension("simple_cnn") == 64


def test_extract_features_resnet18(registry: TrustedArchitectureRegistry) -> None:
    """Verifies ResNet-18 latent feature extraction shape, head consistency, and state_dict preservation."""
    model = registry.create_instance("resnet18")
    assert not model.training

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.extract_features(x)
        assert feats.shape == (2, 512)

        logits_direct = model(x)
        logits_from_head = model.fc(feats)
        assert torch.allclose(logits_direct, logits_from_head, atol=1e-5)

        feats_from_registry = registry.extract_features(model, "resnet18", x)
        assert torch.equal(feats, feats_from_registry)
        assert registry.get_feature_dimension("resnet18") == 512

    # Verify state_dict keys match vanilla ResNet-18 exactly
    vanilla_keys = set(models.resnet18(weights=None).state_dict().keys())
    assert set(model.state_dict().keys()) == vanilla_keys


def test_extract_features_mobilenet_v2(registry: TrustedArchitectureRegistry) -> None:
    """Verifies MobileNetV2 latent feature extraction shape, classifier consistency, and state_dict preservation."""
    model = registry.create_instance("mobilenet_v2")
    assert not model.training

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.extract_features(x)
        assert feats.shape == (2, 1280)

        logits_direct = model(x)
        # In eval mode dropout is identity, classifier[1] is the final Linear(1280, 1000)
        logits_from_head = model.classifier[1](feats)
        assert torch.allclose(logits_direct, logits_from_head, atol=1e-5)

        feats_from_registry = registry.extract_features(model, "mobilenet_v2", x)
        assert torch.equal(feats, feats_from_registry)
        assert registry.get_feature_dimension("mobilenet_v2") == 1280

    # Verify state_dict keys match vanilla MobileNetV2 exactly
    vanilla_keys = set(models.mobilenet_v2(weights=None).state_dict().keys())
    assert set(model.state_dict().keys()) == vanilla_keys


def test_feature_dimension_and_extraction_unknown_architecture_raises(
    registry: TrustedArchitectureRegistry,
) -> None:
    """Verifies unknown architecture IDs raise UnknownArchitectureError."""
    with pytest.raises(UnknownArchitectureError):
        registry.get_feature_dimension("nonexistent_arch_xyz")

    dummy = nn.Linear(10, 2)
    with pytest.raises(UnknownArchitectureError):
        registry.extract_features(dummy, "nonexistent_arch_xyz", torch.randn(2, 10))


def test_feature_extraction_leaves_eval_mode_and_running_stats_intact(
    registry: TrustedArchitectureRegistry,
) -> None:
    """Verifies feature extraction in eval mode does not mutate BatchNorm running stats."""
    model = registry.create_instance("simple_cnn")
    initial_running_mean = model.bn1.running_mean.clone()

    x = torch.randn(10, 3, 32, 32)
    with torch.no_grad():
        for _ in range(5):
            _ = model.extract_features(x)

    assert torch.equal(model.bn1.running_mean, initial_running_mean)
