# ADR-002: Offline-first deployment

- **Status:** Accepted / locked
- **Date:** 2026-11-10

## Context

The SIH concept targets deployment in offline/air-gapped environments and should not depend on cloud services or external APIs.

## Decision

Core processing, persistence, reporting, and UI shall run locally. Internet access may be used during development to obtain dependencies/research materials, but runtime operation must not require it after software/assets are provisioned.

## Consequences

- Dependencies and optional model weights must be provisioned before air-gapped deployment.
- No runtime download of weights or datasets.
- Docker and offline install instructions are required.
- Local key management, backup, and recovery need explicit design.
- Local offline operation does not by itself guarantee security against a compromised host.
