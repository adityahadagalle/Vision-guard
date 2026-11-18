"""SIH 2026 Chaos Pipeline Adversarial Validation Suite.

Executes deterministic stress testing and adversarial threat simulations:
1. Attack A (Poisoned Model Attack): Submits a neural network artifact with hidden
   trigger representations. Asserts spectral signatures and activation clustering
   emit SUSPECTED_POISON findings, trigger QUARANTINE disposition, and the Epistemic
   Conflict Engine synthesizes BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER.
2. Attack B (Telemetry Replay Attack): Simulates edge telemetry forgery and replay
   by reusing monotonic sequence numbers and broken receipt hashes. Asserts HTTP 409
   Conflict rejection with REPLAY_DETECTED.
3. Attack C (Micro-Batch Drift Attack): Simulates small-sample statistical hallucination
   attempts with exactly N=29 samples (below the N=30 threshold). Asserts ADR-012
   epistemic gating enforces CapabilityStatus.INCONCLUSIVE without guessing scores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.repositories.assessment_repository import SqliteAssessmentRepository
from backend.app.db.repositories.inference_repository import SqliteInferenceRepository
from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.repositories.model_repository import SqliteModelRepository
from backend.app.domain.entities.enums import (
    AssessmentStatus,
    CapabilityStatus,
    FindingCategory,
    JobStatus,
    SuggestedDisposition,
)
from backend.app.modules.inference_integrity.audit import ChainAuditEngine
from backend.app.modules.inference_integrity.client import (
    EdgeClientTransmissionError,
    VisionGuardEdgeClient,
)
from backend.app.modules.model_integrity.architecture.registry import SimpleCNN
from backend.app.modules.model_integrity.conflict import ConflictType
from backend.app.services.orchestrator import AssessmentOrchestrator


class TestChaosValidationEngine:
    """SIH 2026 Chaos Benchmark Test Suite verifying resilience against adversarial vectors."""

    def test_chaos_attack_01_poisoned_model_backdoor(
        self,
        chaos_env: tuple[TestClient, sessionmaker[Session]],
        tmp_path: Path,
    ) -> None:
        """Attack Scenario 1: Poisoned Model Checkpoint & Latent Trojan Trigger.

        Simulates an adversary injecting a poisoned computer-vision model with a stealth
        backdoor trigger into the model repository.
        Expected Defense:
        - Spectral Signatures and Activation Clustering detect the bimodal distribution.
        - System emits FindingCategory.SUSPECTED_POISON.
        - Disposition recommends QUARANTINE.
        - Epistemic Conflict Engine detects the contradiction and emits FindingCategory.CONFLICT.
        """
        api_client, session_factory, test_config = chaos_env

        # 1. Provision model checkpoint and ingest via orchestrator
        model_id = "mod_chaos_poisoned"
        model = SimpleCNN()
        model_path = tmp_path / f"{model_id}.pt"
        torch.save(model.state_dict(), model_path)
        expected_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        test_config.quarantine_dir.mkdir(parents=True, exist_ok=True)
        quarantine_model = test_config.quarantine_dir / f"quarantine_{model_id}.pt"
        quarantine_model.write_bytes(model_path.read_bytes())

        with session_factory() as session:
            model_repo = SqliteModelRepository(session)
            job_queue = SqliteJobQueue(session)

            model_repo.create_model(
                model_id=model_id,
                name=f"{model_id}.pt",
                storage_path=str(quarantine_model),
                sha256=expected_sha256,
                size_bytes=quarantine_model.stat().st_size,
            )

            job_queue.submit_job(
                task_type="model_ingestion",
                payload={
                    "model_id": model_id,
                    "quarantine_path": str(quarantine_model),
                    "filename": f"{model_id}.pt",
                    "expected_sha256": expected_sha256,
                },
            )

            orchestrator = AssessmentOrchestrator(session=session, config=test_config)
            job_res = orchestrator.process_next_job(worker_id="chaos_ingest_worker")
            assert job_res is not None
            assert job_res["status"] == JobStatus.COMPLETED.value

        # 2. Synthesize poisoned latent representations (Class 4 has shifted trigger direction)
        np.random.seed(2026)
        n_samples = 100
        dim = 64
        poisoned_class_samples = np.random.randn(n_samples, dim)
        trigger = np.zeros(dim)
        trigger[:6] = 6.0  # Dense directional shift on 20% of samples
        poisoned_class_samples[:20] += trigger

        poisoned_latents: dict[str, Any] = {
            str(c): np.random.randn(n_samples, dim) for c in range(10)
        }
        poisoned_latents["4"] = poisoned_class_samples
        json_latents = {k: v.tolist() for k, v in poisoned_latents.items()}

        # 3. Schedule assessment via REST API
        req_payload = {
            "name": "SIH-2026 Chaos Backdoor Defense Assessment",
            "config": {
                "expected_architecture": "simple_cnn",
                "expected_sha256": expected_sha256,
                "latent_representations": json_latents,
                "outlier_ratio_threshold": 0.15,
                "silhouette_threshold": 0.35,
                "cluster_ratio_threshold": 0.10,
                "probes_count": 10,
            },
        }
        create_res = api_client.post(f"/api/v1/models/{model_id}/assessments", json=req_payload)
        assert create_res.status_code == status.HTTP_202_ACCEPTED
        assessment_id = create_res.json()["assessment_id"]

        # 4. Execute assessment job via orchestrator
        with session_factory() as session:
            orchestrator = AssessmentOrchestrator(session=session, config=test_config)
            assess_res = orchestrator.process_next_job(worker_id="chaos_orchestrator_worker")
            assert assess_res is not None
            assert assess_res["status"] == JobStatus.COMPLETED.value

        # 5. Verify assessment completion and detector findings via API
        get_res = api_client.get(f"/api/v1/assessments/{assessment_id}")
        assert get_res.status_code == status.HTTP_200_OK
        ass_data = get_res.json()
        assert ass_data["status"] == AssessmentStatus.COMPLETED.value

        findings_res = api_client.get(f"/api/v1/assessments/{assessment_id}/findings")
        assert findings_res.status_code == status.HTTP_200_OK
        items = findings_res.json()["items"]

        # Assert SUSPECTED_POISON findings generated
        poison_findings = [f for f in items if f["category"] == FindingCategory.SUSPECTED_POISON.value]
        assert len(poison_findings) >= 1, "Expected SUSPECTED_POISON finding from backdoor detectors."
        assert any(
            f["suggested_disposition"] == SuggestedDisposition.QUARANTINE.value
            for f in poison_findings
        ), "Expected QUARANTINE disposition recommendation."

        # Assert Conflict Engine synthesized BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER
        with session_factory() as session:
            ass_repo = SqliteAssessmentRepository(session)
            db_findings = ass_repo.get_findings(assessment_id)
            conflicts = [f for f in db_findings if f.category == FindingCategory.CONFLICT]
            assert len(conflicts) >= 1, "Expected automated Conflict Engine synthesis."
            assert (
                conflicts[0].details.get("conflict_type")
                == ConflictType.BEHAVIORAL_STABILITY_BACKDOOR_TRIGGER.value
            )

    def test_chaos_attack_02_telemetry_replay_attack(
        self,
        chaos_env: tuple[TestClient, sessionmaker[Session]],
        edge_keypair: tuple[bytes, bytes],
    ) -> None:
        """Attack Scenario 2: Telemetry Replay & Out-of-Order Sequence Forgery.

        Simulates an edge eavesdropper or malicious actor attempting to replay previous
        valid inference receipts to spoof device telemetry.
        Expected Defense:
        - Edge client records genuine genesis inference (seq=0).
        - Adversary re-transmits seq=0 payload.
        - Server detects monotonic collision and rejects with HTTP 409 Conflict (REPLAY_DETECTED).
        - Edge client sequence locking invariant guarantees sequence number does not advance.
        - Database audit verifies chain continuity is unbroken.
        """
        api_client, session_factory, _ = chaos_env
        priv_pem, pub_pem = edge_keypair
        device_id = "edge-camera-chaos-defense-01"

        client = VisionGuardEdgeClient(
            device_id=device_id,
            api_url="http://testserver/api/v1",
            private_key_pem=priv_pem,
            public_key_pem=pub_pem,
            http_client=api_client,
        )

        # 1. Ingest valid genesis receipt (seq=0)
        res_genesis = client.record_inference(
            model_sha256="a" * 64,
            input_hash="1" * 64,
            inference_output={"class": "pedestrian", "confidence": 0.98},
        )
        assert res_genesis["status"] == "ACCEPTED"
        assert res_genesis["sequence_number"] == 0
        assert client.sequence_number == 1

        # 2. Ingest second valid receipt (seq=1)
        res_seq1 = client.record_inference(
            model_sha256="a" * 64,
            input_hash="2" * 64,
            inference_output={"class": "cyclist", "confidence": 0.94},
        )
        assert res_seq1["status"] == "ACCEPTED"
        assert res_seq1["sequence_number"] == 1
        assert client.sequence_number == 2

        # 3. Adversarial Replay Attack: re-submit sequence 0
        replay_client = VisionGuardEdgeClient(
            device_id=device_id,
            api_url="http://testserver/api/v1",
            private_key_pem=priv_pem,
            public_key_pem=pub_pem,
            http_client=api_client,
        )
        assert replay_client.sequence_number == 0

        # Attempt to record sequence 0 again must be intercepted with HTTP 409
        with pytest.raises(EdgeClientTransmissionError) as exc_info:
            replay_client.record_inference(
                model_sha256="a" * 64,
                input_hash="1" * 64,
                inference_output={"class": "pedestrian", "confidence": 0.98},
            )

        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert exc_info.value.response_data["error_code"] == "REPLAY_DETECTED"
        assert "Replay attack detected" in exc_info.value.response_data["message"]

        # 4. Sequence lock verified: replay client sequence did NOT advance
        assert replay_client.sequence_number == 0

        # 5. Database audit: Verified chain has exactly 2 valid receipts, zero corruption
        with session_factory() as session:
            repo = SqliteInferenceRepository(session)
            audit = ChainAuditEngine.verify_device_chain(device_id, repo)
            assert audit.is_valid is True
            assert audit.verified_count == 2
            assert audit.break_sequence_number is None

    def test_chaos_attack_03_micro_batch_drift_attack(
        self,
        chaos_env: tuple[TestClient, sessionmaker[Session]],
    ) -> None:
        """Attack Scenario 3: Micro-Batch Drift Hallucination Attack (N=29).

        Simulates an adversary attempting to trigger false drift alarms or force
        statistical hallucination by submitting an underpowered micro-batch of exactly
        29 samples (violating the calibrated N >= 30 sample threshold).
        Expected Defense:
        - System enforces ADR-012 Epistemic Honesty.
        - Fails closed with CapabilityStatus.INCONCLUSIVE.
        - Does NOT fabricate a synthetic KS/Wasserstein drift score or hallucinate drift.
        - is_drifting remains strictly False.
        """
        api_client, session_factory, _ = chaos_env
        model_id = "model_chaos_microbatch_resilience"

        # Construct adversary micro-batch of exactly 29 samples
        adversarial_samples_n29 = [0.42 + 0.01 * i for i in range(29)]
        baseline_samples = [0.40 + 0.005 * i for i in range(100)]

        payload = {
            "model_id": model_id,
            "target_feature": "spatial_variance_score",
            "reference_data": baseline_samples,
            "production_data": adversarial_samples_n29,
            "ks_alpha": 0.05,
            "wasserstein_threshold": 0.10,
            "min_samples": 30,  # Enforce N >= 30 rule
        }

        response = api_client.post("/api/v1/distribution-shift/evaluate", json=payload)
        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert data["dataset_or_model_id"] == model_id

        # Epistemic assertion: is_drifting MUST NOT be True
        assert data["is_drifting"] is False, "Micro-batch must not trigger false drift."

        # Epistemic status: Overall status must be INCONCLUSIVE
        assert (
            data["overall_status"] == CapabilityStatus.INCONCLUSIVE.value
        ), "System must fail closed with INCONCLUSIVE on N < 30 samples."

        assert "Insufficient sample density" in data["status_reason"]
        assert "29 finite samples" in data["status_reason"]

        # Verify individual detector metrics are inconclusive without fabricated scores
        for metric in data["metrics"]:
            assert metric["capability_status"] == CapabilityStatus.INCONCLUSIVE.value
            assert metric["is_drifting"] is False
            assert metric["drift_score"] is None
            assert metric["p_value"] is None
