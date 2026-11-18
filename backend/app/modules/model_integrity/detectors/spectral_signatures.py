"""Spectral signatures backdoor analyzer for computer vision models.

Implements singular value decomposition (SVD) outlier analysis on latent
representations to identify poisoned sample clusters or Trojan activations (Tran et al., 2018).
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from scipy.linalg import svd

from backend.app.domain.entities.enums import (
    CapabilityStatus,
    EvidenceType,
    FindingCategory,
    SeverityLevel,
    SuggestedDisposition,
)
from backend.app.domain.entities.evidence import EvidenceRecord
from backend.app.domain.entities.finding import (
    AffectedAssetReference,
    ReproducibilityMetadata,
    UniversalFinding,
)
from backend.app.domain.ports.detector import (
    AssessmentContext,
    BaseDetector,
    DetectorRequirements,
    DetectorResult,
)


class SpectralSignaturesDetector(BaseDetector):
    """Evaluates class-conditional latent representations for spectral backdoor signatures."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.spectral_signatures"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requirements(self) -> DetectorRequirements:
        return DetectorRequirements(
            requires_model_checkpoint=True,
            requires_embeddings=True,
        )

    def evaluate(self, context: AssessmentContext) -> DetectorResult:
        """Executes SVD spectral signature evaluation across class-conditional latent representations.

        Args:
            context: Assessment context providing candidate snapshot, latent representations,
                     and configuration.

        Returns:
            DetectorResult reporting SUSPECTED_POISON findings or clean status with epistemic disclaimer.
        """
        # 1. Epistemic Capability Gating: Hardened host isolation check
        is_isolated = getattr(context, "isolation_status", None)
        if is_isolated is False:
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.UNAVAILABLE,
                status_reason=(
                    "Host isolation unhardened: Host platform lacks verified OS-level "
                    "isolation/sandboxing for model execution."
                ),
            )

        if (
            context.config.get("isolation_status") is False
            or context.config.get("isolation_hardened") is False
        ):
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.UNAVAILABLE,
                status_reason=(
                    "Host isolation unhardened: Execution environment lacks verified OS isolation controls."
                ),
            )

        isolation_level = context.config.get("isolation_level")
        if isolation_level and isolation_level != "HARDENED_ISOLATION_VERIFIED":
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.UNAVAILABLE,
                status_reason=(
                    f"Host isolation unhardened: Isolation level '{isolation_level}' "
                    "does not satisfy sandboxing requirements."
                ),
            )

        # 2. Extract latent representations
        class_latents: dict[Any, Any] | None = (
            context.config.get("class_latents")
            or context.config.get("latent_representations")
            or context.embeddings
        )

        if not class_latents or not isinstance(class_latents, dict):
            return DetectorResult(
                detector_id=self.detector_id,
                version=self.version,
                status=CapabilityStatus.INCONCLUSIVE,
                status_reason="No latent representations provided for spectral signatures evaluation.",
            )

        # 3. Epistemic Capability Gating: Sample density check (>= 10 samples per class)
        min_required_samples = int(context.config.get("min_samples_per_class", 10))
        for class_id, latents in class_latents.items():
            arr = np.asarray(latents, dtype=np.float64)
            if len(arr) < min_required_samples:
                return DetectorResult(
                    detector_id=self.detector_id,
                    version=self.version,
                    status=CapabilityStatus.INCONCLUSIVE,
                    status_reason=(
                        f"Insufficient sample density: Class '{class_id}' has {len(arr)} samples; "
                        f"minimum threshold for spectral SVD projection is {min_required_samples} samples per class."
                    ),
                )

        # 4. SVD Outlier Scoring (Tran et al., 2018)
        now = datetime.now(UTC)
        assessment_id = str(
            context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")
        )
        outlier_ratio_threshold = float(context.config.get("outlier_ratio_threshold", 0.15))
        mad_multiplier = float(context.config.get("outlier_mad_multiplier", 3.0))

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        flagged_classes: list[str] = []
        class_metrics: dict[str, Any] = {}

        for class_id_raw, latents in class_latents.items():
            class_id = str(class_id_raw)
            arr = np.asarray(latents, dtype=np.float64)
            n_samples, n_features = arr.shape

            # Step a: Compute class mean vector
            mean = np.mean(arr, axis=0)

            # Step b: Mean-center the representations
            centered = arr - mean

            # Step c: Compute SVD
            _, s, vt = svd(centered, full_matrices=False)

            # Step d: Top right singular vector (direction of maximum variance)
            v = vt[0]

            # Step e: Project centered representations onto top singular vector (squared dot product)
            projections = np.dot(centered, v) ** 2

            # Step f: Robust outlier detection via Median Absolute Deviation (MAD)
            med = float(np.median(projections))
            mad = float(np.median(np.abs(projections - med)))
            normalizer = 1.4826 * mad if mad > 1e-8 else float(np.std(projections))
            cutoff = med + mad_multiplier * normalizer

            outlier_mask = projections > cutoff
            outlier_count = int(np.sum(outlier_mask))
            outlier_ratio = float(outlier_count / n_samples)
            max_projection = float(np.max(projections))
            max_multiplier = float(max_projection / (med + 1e-9))

            class_metrics[class_id] = {
                "sample_count": n_samples,
                "feature_dim": n_features,
                "top_singular_value": float(s[0]),
                "median_projection": med,
                "mad": mad,
                "cutoff": cutoff,
                "outlier_count": outlier_count,
                "outlier_ratio": outlier_ratio,
                "max_outlier_multiplier": max_multiplier,
            }

            if outlier_ratio > outlier_ratio_threshold:
                flagged_classes.append(class_id)
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_spectral_{uid}"
                ev_id = f"ev_spectral_{uid}"

                severity = SeverityLevel.CRITICAL if outlier_ratio >= 0.25 else SeverityLevel.HIGH
                confidence = min(0.95, round(0.50 + outlier_ratio * 1.5, 2))

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.STATISTICAL_TEST,
                    finding_id=fnd_id,
                    signal_name="spectral_signature_svd_outlier",
                    rationale=(
                        f"Class '{class_id}' latent representations exhibit an anomalous spectral signature "
                        f"along the top singular vector: {outlier_count}/{n_samples} samples ({outlier_ratio:.1%}) "
                        f"exceed the threshold (max outlier multiplier: {max_multiplier:.2f}x median, "
                        f"threshold: {outlier_ratio_threshold:.1%})."
                    ),
                    created_at=now,
                    sample_reference=str(class_id),
                    observed_value=round(outlier_ratio, 4),
                    threshold_value=outlier_ratio_threshold,
                    metadata={
                        "class_id": class_id,
                        "outlier_count": outlier_count,
                        "total_samples": n_samples,
                        "max_outlier_multiplier": round(max_multiplier, 4),
                        "top_singular_value": round(float(s[0]), 4),
                    },
                )
                evidence_list.append(ev)

                fnd = UniversalFinding(
                    finding_id=fnd_id,
                    assessment_id=assessment_id,
                    detector_id=self.detector_id,
                    detector_version=self.version,
                    category=FindingCategory.SUSPECTED_POISON,
                    severity=severity,
                    title=f"Suspected Backdoor Poisoning: Spectral Outlier Cluster in Class '{class_id}'",
                    description=(
                        f"Latent feature representations for class '{class_id}' exhibit a significant "
                        f"bimodal/heavy-tailed distribution along the top singular vector under SVD decomposition. "
                        f"{outlier_count} of {n_samples} samples ({outlier_ratio:.1%}) exceed the outlier threshold "
                        f"(max outlier multiplier: {max_multiplier:.2f}x median). "
                        f"This concentration along a single eigenvector is characteristic of poisoned training samples "
                        f"or Trojan trigger activations (Tran et al., 2018). "
                        f"Limitations: Absence of detected spectral signatures does not prove absence of backdoors."
                    ),
                    confidence=confidence,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={
                            "class_id": class_id,
                            "outlier_ratio_threshold": outlier_ratio_threshold,
                            "mad_multiplier": mad_multiplier,
                            "min_samples_per_class": min_required_samples,
                        },
                    ),
                    created_at=now,
                    suggested_disposition=SuggestedDisposition.QUARANTINE,
                    anomaly_score=round(outlier_ratio, 4),
                    affected_assets=[
                        AffectedAssetReference(
                            asset_id=context.snapshot.asset_id,
                        )
                    ],
                    evidence_ids=[ev_id],
                    details={
                        "class_id": class_id,
                        "outlier_ratio": outlier_ratio,
                        "outlier_count": outlier_count,
                        "total_samples": n_samples,
                        "max_outlier_multiplier": round(max_multiplier, 4),
                        "top_singular_value": round(float(s[0]), 4),
                    },
                )
                findings.append(fnd)

        return DetectorResult(
            detector_id=self.detector_id,
            version=self.version,
            status=CapabilityStatus.COMPLETED,
            findings=findings,
            evidence=evidence_list,
            metadata={
                "evaluated_classes": [str(c) for c in class_latents],
                "flagged_classes": flagged_classes,
                "outlier_ratio_threshold": outlier_ratio_threshold,
                "class_metrics": class_metrics,
                "caveat": "Absence of detected backdoor indicators does not prove absence of backdoors.",
            },
        )
