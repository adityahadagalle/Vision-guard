# VisionGuard API Specification (Draft)

**Status:** Draft contract for implementation planning. Confirm schemas and authentication/local-user model before treating endpoints as stable.

Base path proposal: `/api/v1`

## General conventions

- JSON request/response bodies unless file upload is explicitly required.
- Use generated UUID-like identifiers; never accept a user path as an internal storage path.
- Return structured errors with a stable `code`, safe `message`, and optional `details`.
- Long assessments may become background jobs; execution model is TBD.
- Do not return private key material in any API response.

## Proposed endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process health |
| GET | `/version` | App and schema version |
| POST | `/assets/datasets` | Register/upload a dataset |
| POST | `/assets/models` | Register/upload a PyTorch model artifact |
| GET | `/assets/{asset_id}` | Read asset metadata |
| POST | `/assessments` | Create assessment with selected assets/modules |
| GET | `/assessments/{assessment_id}` | Get assessment status/summary |
| POST | `/assessments/{assessment_id}/run` | Start assessment |
| GET | `/assessments/{assessment_id}/findings` | List findings |
| GET | `/findings/{finding_id}` | Get finding with evidence references |
| POST | `/findings/{finding_id}/disposition` | Record accept/review/quarantine decision |
| GET | `/assessments/{assessment_id}/coverage` | Detector capability/status summary |
| POST | `/inference/receipts` | Create/sign an inference receipt |
| POST | `/inference/verify` | Verify receipt/chain data |
| GET | `/reports/{assessment_id}` | Build/download local report |

## Draft request examples

### Create assessment

```json
{
  "name": "camera-dataset-check-001",
  "asset_ids": ["asset-id-1", "asset-id-2"],
  "modules": ["data_integrity", "model_integrity"],
  "config_ref": "config-id-or-version"
}
```

### Record disposition

```json
{
  "disposition": "review",
  "comment": "Investigate source batch before deciding."
}
```

## Draft response principles

- Every assessment exposes state and detector coverage.
- Every finding identifies the evidence and detector run that produced it.
- Scores are optional and accompanied by method/scale/threshold meaning.
- “No findings” must not imply complete assurance if checks were unavailable or failed.
- Error messages must not disclose secrets, local private paths, or raw sensitive data.

## Open API design questions

- Multipart upload limits and resumability.
- Background-job mechanism (start simple; add queue only if necessary).
- Local authentication and CSRF strategy if bound beyond loopback.
- Pagination and filtering.
- Report media type and retention.
- API versioning and backwards compatibility.
