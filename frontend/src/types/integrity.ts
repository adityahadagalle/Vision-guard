/**
 * VisionGuard Core Integrity Types
 * Aligned with backend cryptographic, distribution shift, and attestation models.
 */

export type CapabilityStatus =
  | 'COMPLETED'
  | 'NOT_APPLICABLE'
  | 'UNAVAILABLE'
  | 'SKIPPED'
  | 'FAILED'
  | 'TIMED_OUT'
  | 'INCONCLUSIVE'
  | 'CONFLICTED';

export interface InferenceReceipt {
  receipt_id: string;
  edge_device_id?: string;
  sequence_number: number;
  timestamp_utc: string;
  model_id: string;
  input_sha256: string;
  prediction_sha256: string;
  prev_receipt_hash: string | null;
  receipt_hash: string;
  signature_b64: string;
  public_key_b64: string;
  verified: boolean;
  status: 'VALID' | 'TAMPERED' | 'UNVERIFIED' | 'BROKEN_CHAIN' | 'ACCEPTED';
}

export interface DriftMetric {
  metric_name: string;
  drift_score: number | null;
  threshold: number;
  p_value: number | null;
  is_drifting: boolean;
  capability_status: CapabilityStatus;
  status_reason: string | null;
  reference_sample_count: number;
  production_sample_count: number;
  details?: Record<string, unknown>;
}

export interface DriftReport {
  report_id: string;
  timestamp_utc: string;
  dataset_or_model_id: string | null;
  metrics: DriftMetric[];
  is_drifting: boolean;
  overall_status: CapabilityStatus;
  reference_sample_count: number;
  production_sample_count: number;
  details?: Record<string, unknown>;
}

export interface QuarantineItem {
  id: string;
  name: string;
  asset_type: 'dataset' | 'model';
  quarantine_reason: string;
  quarantined_at: string;
  sha256_digest: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'QUARANTINED' | 'UNDER_REVIEW' | 'RELEASED';
  details: {
    rule_violated?: string;
    detected_layer?: string;
    hash_expected?: string;
    hash_actual?: string;
    description?: string;
  };
}

export interface AttestationReport {
  timestamp_utc: string;
  hardware_fingerprint: string;
  is_offline_enforced: boolean;
  is_non_root_user: boolean;
  is_hardware_bound: boolean;
  is_compliant: boolean;
}

export interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  isOfflineFallback: boolean;
}

export type ActiveTab = 'hash-chain' | 'drift-monitor' | 'quarantine' | 'attestation';
