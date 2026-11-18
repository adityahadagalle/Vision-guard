import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../client';
import { AttestationReport } from '../../types/integrity';

const DEFAULT_REPORT: AttestationReport = {
  timestamp_utc: new Date().toISOString(),
  hardware_fingerprint: '3a8fe9b2c104d77e48b812f864e2239c011e4f9b8821adbc430e129845cd7a10',
  is_offline_enforced: true,
  is_non_root_user: true,
  is_hardware_bound: true,
  is_compliant: true,
};

export function useAttestation() {
  const [report, setReport] = useState<AttestationReport>(DEFAULT_REPORT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  const fetchAttestation = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const res = await apiClient.getAttestationReport();

    if (res.data) {
      setReport(res.data);
      setIsLive(!res.isOfflineFallback);
    } else {
      setError(res.error);
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchAttestation();
  }, [fetchAttestation]);

  return {
    report,
    isLoading,
    error,
    isLive,
    refresh: fetchAttestation,
  };
}
