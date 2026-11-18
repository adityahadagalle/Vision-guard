"""Kolmogorov-Smirnov (KS) two-sample covariate shift detector.

Evaluates continuous 1D distribution shift (e.g., confidence scores, feature norms,
or prediction logits) between an authoritative reference set and production telemetry
using the two-sample Kolmogorov-Smirnov statistical test (scipy.stats.ks_2samp).

Enforces ADR-012 Epistemic Honesty:
- Requires a minimum sample density (default N >= 30) in both windows.
- If sample density is insufficient, returns CapabilityStatus.INCONCLUSIVE
  without fabricating a drift score or guessing statistical significance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from backend.app.domain.entities.drift import DriftMetric
from backend.app.domain.entities.enums import CapabilityStatus


class KolmogorovSmirnovDriftDetector:
    """Statistical detector evaluating 1D continuous distribution drift via the KS test.

    The two-sample Kolmogorov-Smirnov test evaluates whether two independent samples
    were drawn from the same continuous distribution by measuring the supremum
    distance between their empirical cumulative distribution functions (ECDFs).
    """

    DETECTOR_ID = "distribution_shift.kolmogorov_smirnov"
    VERSION = "1.0.0"

    @classmethod
    def evaluate(
        cls,
        reference_data: Sequence[float] | np.ndarray,
        production_data: Sequence[float] | np.ndarray,
        *,
        alpha: float = 0.05,
        min_samples: int = 30,
        alternative: str = "two-sided",
    ) -> DriftMetric:
        """Executes the two-sample KS drift evaluation across reference and production samples.

        Args:
            reference_data: 1D array-like sequence of baseline/reference scalar values.
            production_data: 1D array-like sequence of runtime/production scalar values.
            alpha: Significance level threshold for rejecting the null hypothesis (default 0.05).
            min_samples: Minimum required finite sample count per window (default 30).
            alternative: Defines the alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            DriftMetric containing the KS statistic, p-value, drift verdict, and capability status.
        """
        # Step 1: Input conversion to 1D float arrays
        try:
            ref_arr = np.asarray(reference_data, dtype=np.float64).ravel()
            prod_arr = np.asarray(production_data, dtype=np.float64).ravel()
        except (ValueError, TypeError) as exc:
            return DriftMetric(
                metric_name=cls.DETECTOR_ID,
                drift_score=None,
                threshold=alpha,
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
                threshold=alpha,
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

        # Step 4: Execute two-sample Kolmogorov-Smirnov test
        res = stats.ks_2samp(ref_clean, prod_clean, alternative=alternative)
        ks_stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Step 5: Evaluate hypothesis testing decision
        # Null hypothesis H0: Both samples are drawn from the same continuous distribution.
        # Reject H0 when p_value < alpha => significant distribution shift.
        is_drifting = bool(p_val < alpha)

        details: dict[str, Any] = {
            "statistic_location": float(getattr(res, "statistic_location", 0.0)),
            "statistic_sign": int(getattr(res, "statistic_sign", 0)),
            "ref_mean": float(np.mean(ref_clean)),
            "prod_mean": float(np.mean(prod_clean)),
            "ref_std": float(np.std(ref_clean)),
            "prod_std": float(np.std(prod_clean)),
            "ref_median": float(np.median(ref_clean)),
            "prod_median": float(np.median(prod_clean)),
            "alpha": alpha,
            "alternative": alternative,
        }

        return DriftMetric(
            metric_name=cls.DETECTOR_ID,
            drift_score=ks_stat,
            threshold=alpha,
            p_value=p_val,
            is_drifting=is_drifting,
            capability_status=CapabilityStatus.COMPLETED,
            status_reason=None,
            reference_sample_count=n_ref,
            production_sample_count=n_prod,
            details=details,
        )


def detect_ks_drift(
    reference_data: Sequence[float] | np.ndarray,
    production_data: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    min_samples: int = 30,
) -> DriftMetric:
    """Convenience functional interface for Kolmogorov-Smirnov drift detection."""
    return KolmogorovSmirnovDriftDetector.evaluate(
        reference_data=reference_data,
        production_data=production_data,
        alpha=alpha,
        min_samples=min_samples,
    )
