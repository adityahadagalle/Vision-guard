# ADR-003: Initial model artifact formats

- **Status:** Accepted / locked for initial scope
- **Date:** 2026-11-10

## Context

The original SIH concept mentioned ONNX, PyTorch, and TorchScript. The project owner subsequently selected PyTorch only and asked to remove ONNX Runtime/ONNX support.

## Decision

The initial system supports PyTorch-native `.pt` / `.pth` artifacts. Do not implement ONNX or ONNX Runtime support in the initial release. TensorFlow/Keras is also excluded.

## Consequences

- The model adapter interface may be designed cleanly, but only the PyTorch adapter is implemented initially.
- TorchScript is not a separate cross-framework target; support only if it is explicitly tested and justified within the PyTorch-native scope.
- Loading arbitrary checkpoints is a security risk and requires a safe-loading policy.
- Dataset formats and model serialization are separate concerns; COCO/YOLO dataset annotation support can remain even when model support is PyTorch-only.
