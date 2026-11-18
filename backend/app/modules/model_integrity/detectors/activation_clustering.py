"""Activation clustering backdoor analyzer for computer vision models.

Implements dimensionality reduction and 2-cluster k-means evaluation with silhouette
scoring on class-conditional latent representations to detect Trojan sub-populations (Chen et al., 2018).
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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


class ActivationClusteringDetector(BaseDetector):
    """Evaluates class-conditional latent representations for bimodal activation clustering."""

    @property
    def detector_id(self) -> str:
        return "model_integrity.activation_clustering"

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
        """Executes activation clustering evaluation across class-conditional latent representations.

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
                status_reason="No latent representations provided for activation clustering evaluation.",
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
                        f"minimum threshold for activation clustering is {min_required_samples} samples per class."
                    ),
                )

        # 4. Activation Clustering (Chen et al., 2018)
        now = datetime.now(UTC)
        assessment_id = str(
            context.config.get("assessment_id", f"asm_{context.snapshot.snapshot_id}")
        )
        silhouette_threshold = float(context.config.get("silhouette_threshold", 0.35))
        cluster_ratio_threshold = float(context.config.get("cluster_ratio_threshold", 0.10))
        random_seed = int(context.config.get("cluster_seed", 42))
        max_pca_components = int(context.config.get("pca_components", 10))

        findings: list[UniversalFinding] = []
        evidence_list: list[EvidenceRecord] = []
        flagged_classes: list[str] = []
        class_metrics: dict[str, Any] = {}

        for class_id_raw, latents in class_latents.items():
            class_id = str(class_id_raw)
            arr = np.asarray(latents, dtype=np.float64)
            n_samples, n_features = arr.shape

            # Step a: PCA Dimensionality Reduction
            n_components = min(max_pca_components, n_features, n_samples - 1)
            pca = PCA(n_components=n_components, random_state=random_seed)
            reduced = pca.fit_transform(arr)

            # Step b: 2-Cluster k-Means Evaluation
            km = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
            cluster_labels = km.fit_predict(reduced)

            n0 = int(np.sum(cluster_labels == 0))
            n1 = int(np.sum(cluster_labels == 1))

            # Step c: Evaluate cluster sizes & silhouette score
            if n0 == 0 or n1 == 0:
                score = 0.0
                cluster_ratio = 0.0
            else:
                score = float(silhouette_score(reduced, cluster_labels))
                smaller = min(n0, n1)
                larger = max(n0, n1)
                cluster_ratio = float(smaller / larger) if larger > 0 else 0.0

            class_metrics[class_id] = {
                "sample_count": n_samples,
                "feature_dim": n_features,
                "pca_components": n_components,
                "cluster_0_count": n0,
                "cluster_1_count": n1,
                "cluster_ratio": cluster_ratio,
                "silhouette_score": score,
            }

            # Step d: Flag if both silhouette separation and cluster ratio exceed thresholds
            if score >= silhouette_threshold and cluster_ratio >= cluster_ratio_threshold:
                flagged_classes.append(class_id)
                uid = uuid.uuid4().hex[:12]
                fnd_id = f"fnd_activation_{uid}"
                ev_id = f"ev_activation_{uid}"

                severity = SeverityLevel.CRITICAL if score >= 0.50 else SeverityLevel.HIGH
                confidence = min(0.95, round(0.50 + score * 0.45, 2))

                smaller_count = min(n0, n1)
                larger_count = max(n0, n1)

                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    evidence_type=EvidenceType.ACTIVATION_CLUSTER,
                    finding_id=fnd_id,
                    signal_name="activation_cluster_bimodality",
                    rationale=(
                        f"Class '{class_id}' activations form 2 well-separated clusters in PCA projection space: "
                        f"silhouette score {score:.3f} >= threshold {silhouette_threshold:.2f} "
                        f"(cluster ratio: {cluster_ratio:.2f}, cluster sizes: {smaller_count}/{larger_count})."
                    ),
                    created_at=now,
                    sample_reference=str(class_id),
                    observed_value=round(score, 4),
                    threshold_value=silhouette_threshold,
                    metadata={
                        "class_id": class_id,
                        "silhouette_score": round(score, 4),
                        "cluster_ratio": round(cluster_ratio, 4),
                        "cluster_sizes": [n0, n1],
                        "total_samples": n_samples,
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
                    title=f"Suspected Backdoor Poisoning: Bimodal Activation Cluster in Class '{class_id}'",
                    description=(
                        f"Latent feature activations for class '{class_id}' separate into 2 distinct clusters "
                        f"with a high silhouette score of {score:.3f} "
                        f"(cluster size ratio: {cluster_ratio:.2f}, cluster sizes: {smaller_count}/{larger_count}). "
                        f"Bimodal activation clustering within a single class indicates that a subset of samples "
                        f"activates a divergent Trojan mechanism (Chen et al., 2018). "
                        f"Limitations: Absence of detected activation clusters does not prove absence of backdoors."
                    ),
                    confidence=confidence,
                    reproducibility=ReproducibilityMetadata(
                        execution_timestamp=now,
                        software_versions={"detector": self.version},
                        detector_config={
                            "class_id": class_id,
                            "silhouette_threshold": silhouette_threshold,
                            "cluster_ratio_threshold": cluster_ratio_threshold,
                            "pca_components": n_components,
                            "cluster_seed": random_seed,
                            "min_samples_per_class": min_required_samples,
                        },
                    ),
                    created_at=now,
                    suggested_disposition=SuggestedDisposition.QUARANTINE,
                    anomaly_score=round(score, 4),
                    affected_assets=[
                        AffectedAssetReference(
                            asset_id=context.snapshot.asset_id,
                        )
                    ],
                    evidence_ids=[ev_id],
                    details={
                        "class_id": class_id,
                        "silhouette_score": round(score, 4),
                        "cluster_ratio": round(cluster_ratio, 4),
                        "cluster_0_count": n0,
                        "cluster_1_count": n1,
                        "total_samples": n_samples,
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
                "silhouette_threshold": silhouette_threshold,
                "cluster_ratio_threshold": cluster_ratio_threshold,
                "class_metrics": class_metrics,
                "caveat": "Absence of detected backdoor indicators does not prove absence of backdoors.",
            },
        )
