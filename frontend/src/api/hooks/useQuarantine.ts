import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../client';
import { QuarantineItem } from '../../types/integrity';

export function useQuarantine() {
  const [items, setItems] = useState<QuarantineItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  const fetchQuarantine = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const res = await apiClient.getQuarantineItems();

    if (res.data) {
      setItems(res.data);
      setIsLive(!res.isOfflineFallback);
    } else {
      setError(res.error);
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchQuarantine();
  }, [fetchQuarantine]);

  return {
    items,
    isLoading,
    error,
    isLive,
    refresh: fetchQuarantine,
  };
}
