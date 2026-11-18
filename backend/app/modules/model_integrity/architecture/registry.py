"""Trusted Architecture Registry for state-dict structural validation.

Provides static reference architecture specifications (ResNet-18, MobileNetV2,
SimpleCNN) originating from trusted code. Evaluates candidate checkpoint state-dicts
against declared layer names, shapes, and dtypes without executing untrusted code.
"""

import types
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from backend.app.modules.model_integrity.exceptions import UnknownArchitectureError


class LayerSpec(BaseModel):
    """Specification of an expected parameter layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    shape: list[int]
    dtype: str
    numel: int


class ArchitectureSpec(BaseModel):
    """Declared specification of a trusted model architecture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture_id: str
    name: str
    description: str
    layers: dict[str, LayerSpec] = Field(default_factory=dict)
    total_parameters: int = 0
    feature_dimension: int = 0


class CompatibilityReport(BaseModel):
    """Detailed structural compatibility evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compatible: bool
    architecture_id: str
    compatibility_score: float  # [0.0, 1.0]
    total_expected_layers: int
    total_candidate_layers: int
    missing_layers: list[str] = Field(default_factory=list)
    unexpected_layers: list[str] = Field(default_factory=list)
    shape_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    dtype_mismatches: list[dict[str, Any]] = Field(default_factory=list)


class SimpleCNN(nn.Module):
    """Reference lightweight CNN architecture for integrity verification and testing."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate latent feature representation (dim 64) before final classification."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.extract_features(x))


def _extract_resnet18_features(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extracts penultimate latent feature representation (dim 512) for ResNet-18."""
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    return torch.flatten(x, 1)


def _extract_mobilenet_v2_features(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extracts penultimate latent feature representation (dim 1280) for MobileNetV2."""
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


class TrustedArchitectureRegistry:
    """Registry maintaining approved reference model architecture contracts."""

    def __init__(self) -> None:
        self._architectures: dict[str, ArchitectureSpec] = {}
        self._register_default_architectures()

    def register(self, spec: ArchitectureSpec) -> None:
        """Registers a trusted architecture specification."""
        self._architectures[spec.architecture_id] = spec

    def get(self, architecture_id: str) -> ArchitectureSpec:
        """Retrieves a trusted architecture specification.

        Raises:
            UnknownArchitectureError: If architecture_id is not registered.
        """
        if architecture_id not in self._architectures:
            raise UnknownArchitectureError(
                f"Architecture '{architecture_id}' is not registered in the Trusted Architecture Registry. "
                f"Available: {sorted(self._architectures.keys())}.",
                details={
                    "architecture_id": architecture_id,
                    "available": sorted(self._architectures.keys()),
                },
            )
        return self._architectures[architecture_id]

    def has(self, architecture_id: str) -> bool:
        """Checks if an architecture is registered."""
        return architecture_id in self._architectures

    def list_ids(self) -> list[str]:
        """Returns sorted list of registered architecture IDs."""
        return sorted(self._architectures.keys())

    def create_instance(self, architecture_id: str) -> nn.Module:
        """Instantiates a fresh reference PyTorch model for the given trusted architecture.

        Args:
            architecture_id: Identifier of registered architecture.

        Returns:
            Configured nn.Module instance in evaluation mode.

        Raises:
            UnknownArchitectureError: If architecture_id is not registered.
        """
        if architecture_id == "simple_cnn":
            model = SimpleCNN()
            model.eval()
            return model
        elif architecture_id == "resnet18":
            import torchvision.models as models

            model = models.resnet18(weights=None)
            model.extract_features = types.MethodType(_extract_resnet18_features, model)
            model.eval()
            return model
        elif architecture_id == "mobilenet_v2":
            import torchvision.models as models

            model = models.mobilenet_v2(weights=None)
            model.extract_features = types.MethodType(_extract_mobilenet_v2_features, model)
            model.eval()
            return model
        raise UnknownArchitectureError(
            f"Architecture '{architecture_id}' is not registered in the Trusted Architecture Registry. "
            f"Available: {sorted(self._architectures.keys())}.",
            details={
                "architecture_id": architecture_id,
                "available": sorted(self._architectures.keys()),
            },
        )

    def get_feature_dimension(self, architecture_id: str) -> int:
        """Returns the penultimate latent feature dimension for the trusted architecture.

        Raises:
            UnknownArchitectureError: If architecture_id is not registered.
        """
        spec = self.get(architecture_id)
        if spec.feature_dimension > 0:
            return spec.feature_dimension
        default_dims = {
            "simple_cnn": 64,
            "resnet18": 512,
            "mobilenet_v2": 1280,
        }
        if architecture_id in default_dims:
            return default_dims[architecture_id]
        raise UnknownArchitectureError(
            f"Architecture '{architecture_id}' is not registered in the Trusted Architecture Registry. "
            f"Available: {sorted(self._architectures.keys())}.",
            details={
                "architecture_id": architecture_id,
                "available": sorted(self._architectures.keys()),
            },
        )

    def extract_features(
        self,
        model: nn.Module,
        architecture_id: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Extracts penultimate latent feature representation for a model given its architecture ID.

        Args:
            model: PyTorch module instance.
            architecture_id: Identifier of registered architecture.
            x: Input tensor batch (N, C, H, W).

        Returns:
            Latent feature tensor batch (N, feature_dimension).

        Raises:
            UnknownArchitectureError: If architecture_id is not registered.
        """
        if hasattr(model, "extract_features") and callable(model.extract_features):
            return model.extract_features(x)
        if architecture_id == "simple_cnn":
            if hasattr(model, "conv1") and hasattr(model, "fc1"):
                relu = getattr(model, "relu", nn.ReLU())
                bn1 = getattr(model, "bn1", nn.Identity())
                bn2 = getattr(model, "bn2", nn.Identity())
                pool = getattr(model, "pool", nn.AdaptiveAvgPool2d((4, 4)))
                f = relu(bn1(model.conv1(x)))
                f = relu(bn2(model.conv2(f)))
                f = pool(f)
                f = torch.flatten(f, 1)
                return relu(model.fc1(f))
        elif architecture_id == "resnet18":
            return _extract_resnet18_features(model, x)
        elif architecture_id == "mobilenet_v2":
            return _extract_mobilenet_v2_features(model, x)
        raise UnknownArchitectureError(
            f"Architecture '{architecture_id}' is not registered in the Trusted Architecture Registry. "
            f"Available: {sorted(self._architectures.keys())}.",
            details={
                "architecture_id": architecture_id,
                "available": sorted(self._architectures.keys()),
            },
        )

    def get_default_input_shape(self, architecture_id: str) -> list[int]:
        """Returns the canonical input shape (C, H, W) expected by the architecture."""
        if architecture_id == "simple_cnn":
            return [3, 32, 32]
        elif architecture_id in ("resnet18", "mobilenet_v2"):
            return [3, 224, 224]
        return [3, 32, 32]

    def validate_compatibility(
        self,
        architecture_id: str,
        candidate_tensors: dict[str, Any],
    ) -> CompatibilityReport:
        """Validates candidate state-dict tensors against the trusted architecture spec.

        Args:
            architecture_id: Identifier of expected architecture.
            candidate_tensors: Dictionary mapping tensor name to dict with 'shape' and 'dtype'.

        Returns:
            CompatibilityReport detailing matches, mismatches, and overall compatibility.
        """
        spec = self.get(architecture_id)
        expected_layers = spec.layers

        missing_layers: list[str] = []
        shape_mismatches: list[dict[str, Any]] = []
        dtype_mismatches: list[dict[str, Any]] = []

        for name, exp_layer in expected_layers.items():
            if name not in candidate_tensors:
                missing_layers.append(name)
                continue

            cand = candidate_tensors[name]
            cand_shape = list(cand.get("shape", []))
            cand_dtype = str(cand.get("dtype", ""))

            if cand_shape != exp_layer.shape:
                shape_mismatches.append(
                    {
                        "layer": name,
                        "expected_shape": exp_layer.shape,
                        "observed_shape": cand_shape,
                    }
                )

            # Compare normalized dtype string (e.g. torch.float32 vs float32)
            norm_exp = exp_layer.dtype.replace("torch.", "")
            norm_cand = cand_dtype.replace("torch.", "")
            if norm_cand != norm_exp:
                dtype_mismatches.append(
                    {
                        "layer": name,
                        "expected_dtype": exp_layer.dtype,
                        "observed_dtype": cand_dtype,
                    }
                )

        unexpected_layers = [k for k in candidate_tensors if k not in expected_layers]

        total_expected = len(expected_layers)
        total_candidate = len(candidate_tensors)

        total_defects = (
            len(missing_layers)
            + len(unexpected_layers)
            + len(shape_mismatches)
            + len(dtype_mismatches)
        )

        denom = max(1, total_expected + len(unexpected_layers))
        compatibility_score = max(0.0, round(1.0 - (total_defects / denom), 4))
        is_compatible = total_defects == 0

        return CompatibilityReport(
            compatible=is_compatible,
            architecture_id=architecture_id,
            compatibility_score=compatibility_score,
            total_expected_layers=total_expected,
            total_candidate_layers=total_candidate,
            missing_layers=missing_layers,
            unexpected_layers=unexpected_layers,
            shape_mismatches=shape_mismatches,
            dtype_mismatches=dtype_mismatches,
        )

    def _register_default_architectures(self) -> None:
        """Extracts reference architecture specifications offline from trusted definitions."""
        # 1. SimpleCNN
        simple_model = SimpleCNN()
        self.register(
            self._extract_spec_from_model(
                "simple_cnn",
                "Simple CNN (10 classes)",
                simple_model,
                feature_dimension=64,
            )
        )

        # 2. ResNet18
        try:
            import torchvision.models as models

            resnet = models.resnet18(weights=None)
            self.register(
                self._extract_spec_from_model(
                    "resnet18",
                    "TorchVision ResNet-18",
                    resnet,
                    feature_dimension=512,
                )
            )
        except Exception:
            pass

        # 3. MobileNetV2
        try:
            import torchvision.models as models

            mobilenet = models.mobilenet_v2(weights=None)
            self.register(
                self._extract_spec_from_model(
                    "mobilenet_v2",
                    "TorchVision MobileNetV2",
                    mobilenet,
                    feature_dimension=1280,
                )
            )
        except Exception:
            pass

    @staticmethod
    def _extract_spec_from_model(
        architecture_id: str,
        name: str,
        model: nn.Module,
        feature_dimension: int = 0,
    ) -> ArchitectureSpec:
        """Converts an instantiated PyTorch module into an immutable ArchitectureSpec."""
        layers: dict[str, LayerSpec] = {}
        total_params = 0

        for key, tensor in model.state_dict().items():
            shape = list(tensor.shape)
            dtype_str = str(tensor.dtype)
            numel = tensor.numel()
            total_params += numel
            layers[key] = LayerSpec(
                name=key,
                shape=shape,
                dtype=dtype_str,
                numel=numel,
            )

        return ArchitectureSpec(
            architecture_id=architecture_id,
            name=name,
            description=f"Standard reference specification for {name}.",
            layers=layers,
            total_parameters=total_params,
            feature_dimension=feature_dimension,
        )
