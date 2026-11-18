import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../client';
import { InferenceReceipt } from '../../types/integrity';

const INITIAL_CHAIN: InferenceReceipt[] = [
  {
    receipt_id: 'rcpt_7f9b8c2d-0001',
    edge_device_id: 'edge_camera_sector_04',
    sequence_number: 1042,
    timestamp_utc: '2026-11-18T11:42:01.120Z',
    model_id: 'resnet50_tamper_resistant_v2',
    input_sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    prediction_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
    prev_receipt_hash: '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',
    receipt_hash: '2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae',
    signature_b64: 'MEQCID...ed25519_canonical_sig_001...',
    public_key_b64: 'MCowBQYDK2VwAyEA...pubkey_ed25519_node_alpha...',
    verified: true,
    status: 'VALID',
  },
  {
    receipt_id: 'rcpt_7f9b8c2d-0002',
    edge_device_id: 'edge_camera_sector_04',
    sequence_number: 1043,
    timestamp_utc: '2026-11-18T11:42:04.450Z',
    model_id: 'resnet50_tamper_resistant_v2',
    input_sha256: 'a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0',
    prediction_sha256: 'b2c3d4e5f6a10718293a4b5c6d7e8f90123456789abcdef0123456789abcdef1',
    prev_receipt_hash: '2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae',
    receipt_hash: '8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4',
    signature_b64: 'MEQCIQ...ed25519_canonical_sig_002...',
    public_key_b64: 'MCowBQYDK2VwAyEA...pubkey_ed25519_node_alpha...',
    verified: true,
    status: 'VALID',
  },
  {
    receipt_id: 'rcpt_7f9b8c2d-0003',
    edge_device_id: 'edge_camera_sector_04',
    sequence_number: 1044,
    timestamp_utc: '2026-11-18T11:42:08.890Z',
    model_id: 'resnet50_tamper_resistant_v2',
    input_sha256: 'c3d4e5f6a1b20718293a4b5c6d7e8f90123456789abcdef0123456789abcdef2',
    prediction_sha256: 'd4e5f6a1b2c30718293a4b5c6d7e8f90123456789abcdef0123456789abcdef3',
    prev_receipt_hash: '8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4',
    receipt_hash: '3f79bb7b435b05321471daefd374cd681f22346743db812f5130f9d28190c37e',
    signature_b64: 'INVALID_SIGNATURE_TAMPER_DETECTED...',
    public_key_b64: 'MCowBQYDK2VwAyEA...pubkey_ed25519_node_alpha...',
    verified: false,
    status: 'TAMPERED',
  },
];

export function useInferenceChain(edgeDeviceId: string = 'edge_camera_sector_04') {
  const [receipts, setReceipts] = useState<InferenceReceipt[]>(INITIAL_CHAIN);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  const fetchChain = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const res = await apiClient.getLatestInferenceReceipt(edgeDeviceId);

    if (res.data) {
      // If we got a live receipt from the backend, prepend/append to chain
      if (!res.isOfflineFallback) {
        setIsLive(true);
        setReceipts((prev) => {
          const exists = prev.some((r) => r.receipt_id === res.data?.receipt_id);
          if (exists) return prev;
          return [res.data as InferenceReceipt, ...prev];
        });
      } else {
        setIsLive(false);
      }
    } else {
      setError(res.error);
    }

    setIsLoading(false);
  }, [edgeDeviceId]);

  useEffect(() => {
    fetchChain();
  }, [fetchChain]);

  const verifyChainIntegrity = useCallback(() => {
    // Audit hash-chain continuity: each previous_receipt_hash must equal preceding receipt_hash
    for (let i = 1; i < receipts.length; i++) {
      const prev = receipts[i - 1];
      const current = receipts[i];
      if (current.prev_receipt_hash !== prev.receipt_hash) {
        return false;
      }
    }
    return true;
  }, [receipts]);

  return {
    receipts,
    setReceipts,
    isLoading,
    error,
    isLive,
    refresh: fetchChain,
    isChainContinuous: verifyChainIntegrity(),
  };
}
