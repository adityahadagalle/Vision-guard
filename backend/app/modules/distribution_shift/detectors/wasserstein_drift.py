"""Wasserstein Distance (Earth Mover's Distance) covariate shift detector.

Evaluates 1D continuous distribution drift (e.g., confidence scores, feature norms,
or bounding box aspect ratios) between an authoritative reference set and production telemetry
using the first Wasserstein metric (scipy.stats.wasserstein_distance).

Enforces ADR-012 Epistemic Honesty:
- Requires a minimum sample density (default N >= 30) in both windows.
- If sample density is insufficient, returns CapabilityStatus.INCONCLUSIVE
  without fabricating a distance or declaring false shifts.
- Distinctly separates distance metrics from statistical p-values (p_value=None).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from backend.app.domain.entities.drift import DriftMetric
from backend.app.domain.entities.enums import CapabilityStatus


class WassersteinDistanceDriftDetector:
    """Statistical detector evaluating 1D continuous distribution drift via Wasserstein distance.

    The first Wasserstein distance (Earth Mover's Distance) computes the minimum work
    required to transform the reference distribution into the production distribution.
    Unlike KS, it provides a geometric distance metric proportional to the magnitude of drift.
    """

    DETECTOR_ID = "distribution_shift.wasserstein_distance"
    VERSION = "1.0.0"

    @classmethod
    def evaluate(
        cls,
        reference_data: Sequence[float] | np.ndarray,
        production_data: Sequence[float] | np.ndarray,
        *,
        threshold: float = 0.10,
        min_samples: int = 30,
    ) -> DriftMetric:
        """Executes Wasserstein distance drift evaluation across reference and production samples.

        Args:
            reference_data: 1D array-like sequence of baseline/reference scalar values.
            production_data: 1D array-like sequence of runtime/production scalar values.
            threshold: Distance threshold above which distribution shift is flagged (default 0.10).
            min_samples: Minimum required finite sample count per window (default 30).

        Returns:
            DriftMetric containing the computed Wasserstein distance, threshold, and capability status.
        """
        # Step 1: Input conversion to 1D float arrays
        try:
            ref_arr = np.asarray(reference_data, dtype=np.float64).ravel()
            prod_arr = np.asarray(production_data, dtype=np.float64).ravel()
        except (ValueError, TypeError) as exc:
            return DriftMetric(
                metric_name=cls.DETECTOR_ID,
                drift_score=None,
                threshold=threshold,
                p_value=None,
                is_drifting=False,
                capability_status=CapabilityStatus.FAILED,
                status_reason=f"Failed to convert inputs to numerical arrays: {exc}",
                reference_sample_count=0,
                production_sample_count=0,
                details={"error": str(exc)},
            )

        # Step 2: Filter non-finite values (NaN / Inf)
        ref_clean = ref_arr[np.isfinite(ref_arr)]
        prod_clean = prod_arr[np.isfinite(prod_arr)]

        n_ref = int(len(ref_clean))
        n_prod = int(len(prod_clean))

        # Step 3: ADR-012 Epistemic Capability Gating (Sample Density)
        if n_ref < min_samples or n_prod < min_samples:
            return DriftMetric(
                metric_name=cls.DETECTOR_ID,
                drift_score=None,
                threshold=threshold,
                p_value=None,
                is_drifting=False,
                capability_status=CapabilityStatus.INCONCLUSIVE,
                status_reason=(
                    f"Insufficient sample density: reference has {n_ref} finite samples, "
                    f"production has {n_prod} finite samples; minimum required is {min_samples} per window."
                ),
                reference_sample_count=n_ref,
                production_sample_count=n_prod,
                details={
                    "min_samples_required": min_samples,
                    "ref_raw_count": int(len(ref_arr)),
                    "prod_raw_count": int(len(prod_arr)),
                    "ref_non_finite_count": int(len(ref_arr) - n_ref),
                    "prod_non_finite_count": int(len(prod_arr) - n_prod),
                },
            )

        # Step 4: Compute 1st Wasserstein distance (EMD)
        w_dist = float(stats.wasserstein_distance(ref_clean, prod_clean))

        # Step 5: Evaluate against configured decision threshold
        # Invariant: p_value is None because Wasserstein distance is a metric, not a hypothesis test
        is_drifting = bool(w_dist > threshold)

        details: dict[str, Any] = {
            "ref_mean": float(np.mean(ref_clean)),
            "prod_mean": float(np.mean(prod_clean)),
            "mean_delta": float(abs(np.mean(prod_clean) - np.mean(ref_clean))),
            "ref_std": float(np.std(ref_clean)),
            "prod_std": float(np.std(prod_clean)),
            "ref_median": float(np.median(ref_clean)),
            "prod_median": float(np.median(prod_clean)),
            "threshold": threshold,
        }

        return DriftMetric(
            metric_name=cls.DETECTOR_ID,
            drift_score=w_dist,
            threshold=threshold,
            p_value=None,
            is_drifting=is_drifting,
            capability_status=CapabilityStatus.COMPLETED,
            status_reason=None,
            reference_sample_count=n_ref,
            production_sample_count=n_prod,
            details=details,
        )


def detect_wasserstein_drift(
    reference_data: Sequence[float] | np.ndarray,
    production_data: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.10,
    min_samples: int = 30,
) -> DriftMetric:
    """Convenience functional interface for Wasserstein distance drift detection."""
    return WassersteinDistanceDriftDetector.evaluate(
        reference_data=reference_data,
        production_data=production_data,
        threshold=threshold,
        min_samples=min_samples,
    )
