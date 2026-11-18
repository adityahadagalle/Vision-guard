# Start Here — First Coding-Agent Session

Do not ask the coding agent to implement VisionGuard immediately. Use this first session to verify that it understands the requirements and identify unresolved decisions.

## Prompt

```text
You are working on VisionGuard. Before writing code, read:
- AGENTS.md
- PROJECT_CONTEXT.md
- REQUIREMENTS.md
- TECH_STACK.md
- docs/ARCHITECTURE.md
- docs/THREAT_MODEL.md
- docs/IMPLEMENTATION_ROADMAP.md
- docs/research/literature_review.md
- docs/research/existing_solutions.md

Do not modify files yet.

1. Summarize the project in your own words.
2. List all locked technical decisions and prohibited technologies.
3. Identify contradictions, missing requirements, security risks, and unrealistic assumptions.
4. Separate locked decisions from proposed design choices and unresolved research questions.
5. Recommend the smallest Phase 1 implementation slice, with acceptance criteria and tests.
6. List the exact files you would create or modify for that slice.

Do not invent research results, datasets, detector performance, or implementation status.
Do not use TensorFlow, Keras, ONNX, or ONNX Runtime.
Wait for approval before writing code.
```

## Later coding prompt template

```text
Task: [one narrowly scoped feature]

Read AGENTS.md and the relevant module specification first. Inspect the current implementation and tests. Do not change unrelated architecture.

Acceptance criteria:
- [observable criterion 1]
- [observable criterion 2]
- [failure/edge case]

Constraints:
- PyTorch only as the deep-learning framework.
- Offline-first; no paid APIs or runtime downloads.
- Keep detector logic separate from API routes and persistence.
- Add tests and update documentation.

Before coding, state the implementation plan and files to change. Then implement the smallest complete slice, run relevant tests, and report exact commands/results plus limitations.
```
