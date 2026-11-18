# ADR-001: PyTorch-only ML framework

- **Status:** Accepted / locked
- **Date:** 2026-11-10

## Context

The project needs a consistent computer-vision model framework for implementation and evaluation. The developer explicitly selected PyTorch only.

## Decision

Use PyTorch as the only deep-learning framework. Use TorchVision where useful. Do not use TensorFlow/Keras or ONNX Runtime/ONNX support in the initial scope.

Supporting utilities such as NumPy, SciPy, Pandas, and scikit-learn are allowed because they are not competing deep-learning frameworks.

## Consequences

- Initial model ingestion targets PyTorch-native `.pt` / `.pth` artifacts.
- Detector implementations and experiments must be compatible with the PyTorch workflow.
- Some published methods may have reference implementations in other frameworks; adapt/reimplement only when practical and verify against the paper.
- Framework-independent cryptographic and statistical components remain independent of PyTorch.
- Any future framework support requires a deliberate ADR and scope review.
