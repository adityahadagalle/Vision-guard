import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../client';
import { DriftMetric, DriftReport } from '../../types/integrity';

export function useDriftReports(modelId: string = 'resnet50_tamper_resistant_v2') {
  const [reports, setReports] = useState<DriftReport[]>([]);
  const [metrics, setMetrics] = useState<DriftMetric[]>([]);
  const [isDrifting, setIsDrifting] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  const fetchReports = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const res = await apiClient.getDriftReports(modelId);

    if (res.data && res.data.length > 0) {
      setReports(res.data);
      setIsLive(!res.isOfflineFallback);

      // Latest report
      const latest = res.data[0];
      setMetrics(latest.metrics || []);
      setIsDrifting(latest.is_drifting);
    } else {
      setError(res.error);
    }

    setIsLoading(false);
  }, [modelId]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  // Epistemic check: Are there inconclusive metrics due to small sample size N < 30?
  const hasInconclusiveMetrics = metrics.some(
    (m) => m.capability_status === 'INCONCLUSIVE'
  );

  return {
    reports,
    metrics,
    isDrifting,
    hasInconclusiveMetrics,
    isLoading,
    error,
    isLive,
    refresh: fetchReports,
  };
}
