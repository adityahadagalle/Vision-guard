# VisionGuard Testing Strategy

## 1. Test layers

### Unit tests

Test pure functions and isolated services:
- SHA-256 digest computation.
- Canonical receipt serialization.
- Ed25519 sign/verify.
- Hash-chain append/verify.
- Dataset path and annotation validation.
- Duplicate matching primitives.
- Detector prerequisite checks and capability status.
- Finding aggregation and severity rules.

### Integration tests

Test boundaries:
- FastAPI route -> application service -> detector -> SQLite.
- Upload -> validation -> asset registration -> digest persistence.
- Detector run -> evidence persistence -> report assembly.
- Inference receipt -> signature -> chain -> verification endpoint.
- Frontend API contract against a local test server.

### Security tests

- Invalid file types and malformed images.
- Oversized files and excessive image dimensions.
- Zip/archive traversal and decompression-bomb behavior.
- Symlinks and path normalization.
- Untrusted/malformed PyTorch checkpoints.
- SQL injection and unsafe query parameters.
- XSS-safe rendering of user-provided metadata.
- Missing/wrong public keys and invalid signatures.
- Tampered, deleted, duplicated, reordered, and replayed audit events.
- Unauthorized access to local key files and sensitive paths.

### ML evaluation tests

These are controlled experiments, not ordinary unit tests. Store them under `experiments/` and produce reproducible result files. Do not make a flaky statistical test the only gate in the unit-test suite.

## 2. Test conventions

- Use pytest.
- Keep unit tests fast and deterministic.
- Use temporary directories and temporary SQLite databases.
- Never use production/private data in CI by default.
- Seed randomness where applicable.
- Separate expected-negative behavior from unexpected exceptions.
- Test capability statuses explicitly: available, unavailable, failed, not_run.

## 3. Definition of done

A change is complete only when:
1. Acceptance criteria are met.
2. Unit and relevant integration/security tests pass.
3. New behavior has tests for failure and boundary cases.
4. Config and API docs are updated.
5. No prohibited framework/dependency has been introduced.
6. The coding agent reports exact commands and actual results.
7. Known limitations are recorded.
