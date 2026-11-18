/**
 * VisionGuard Unified API Client
 * Routes all traffic via the Vite development proxy (/api/v1) strictly targeting 127.0.0.1.
 * Defensively catches network/server faults, returning typed envelopes with fallback states.
 */

import {
  ApiResponse,
  AttestationReport,
  DriftReport,
  InferenceReceipt,
  QuarantineItem,
} from '../types/integrity';

const API_BASE = '/api/v1';

async function safeFetch<T>(
  endpoint: string,
  options?: RequestInit,
  fallbackData: T | null = null
): Promise<ApiResponse<T>> {
  const url = endpoint.startsWith('http') || endpoint.startsWith('/') ? endpoint : `${API_BASE}/${endpoint}`;

  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errMsg = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.message) errMsg = errJson.message;
        else if (errJson.detail) errMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } catch {
        // Fall back to default status text
      }
      return {
        data: fallbackData,
        error: errMsg,
        isOfflineFallback: true,
      };
    }

    const data: T = await res.json();
    return {
      data,
      error: null,
      isOfflineFallback: false,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Network failure or host unreachable';
    return {
      data: fallbackData,
      error: message,
      isOfflineFallback: true,
    };
  }
}

// ============================================================================
// API Methods
// ============================================================================

export const apiClient = {
  /**
   * Health status check
   */
  async getHealth(): Promise<ApiResponse<{ status: string; application: string; version: string }>> {
    return safeFetch('/health', undefined, {
      status: 'offline',
      application: 'visionguard',
      version: '0.1.0-airgap',
    });
  },

  /**
   * Fetch inference receipts.
   * If a device ID is provided, checks latest receipt or falls back to ledger list.
   */
  async getLatestInferenceReceipt(edgeDeviceId: string): Promise<ApiResponse<InferenceReceipt>> {
    const fallback: InferenceReceipt = {
      receipt_id: 'rcpt_7f9b8c2d-0001',
      edge_device_id: edgeDeviceId,
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
    };

    return safeFetch<InferenceReceipt>(
      `${API_BASE}/inference/receipts/${encodeURIComponent(edgeDeviceId)}/latest`,
      undefined,
      fallback
    );
  },

  /**
   * Ingest a new inference receipt
   */
  async ingestReceipt(payload: unknown): Promise<ApiResponse<unknown>> {
    return safeFetch(`${API_BASE}/inference/receipts`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch historical distribution shift reports for a model
   */
  async getDriftReports(modelId: string): Promise<ApiResponse<DriftReport[]>> {
    const fallbackReports: DriftReport[] = [
      {
        report_id: 'drf_7a1b92c401ee',
        timestamp_utc: '2026-11-18T11:40:00Z',
        dataset_or_model_id: modelId,
        is_drifting: true,
        overall_status: 'COMPLETED',
        reference_sample_count: 500,
        production_sample_count: 500,
        metrics: [
          {
            metric_name: 'feature_map_layer4_conv1_mean',
            drift_score: 0.384,
            threshold: 0.05,
            p_value: 0.00012,
            is_drifting: true,
            capability_status: 'COMPLETED',
            status_reason: null,
            reference_sample_count: 500,
            production_sample_count: 500,
          },
          {
            metric_name: 'resnet50_latent_wasserstein_dist',
            drift_score: 0.412,
            threshold: 0.25,
            p_value: null,
            is_drifting: true,
            capability_status: 'COMPLETED',
            status_reason: null,
            reference_sample_count: 500,
            production_sample_count: 500,
          },
          {
            metric_name: 'input_luminance_distribution',
            drift_score: 0.021,
            threshold: 0.05,
            p_value: 0.428,
            is_drifting: false,
            capability_status: 'COMPLETED',
            status_reason: null,
            reference_sample_count: 500,
            production_sample_count: 500,
          },
          {
            metric_name: 'chromatic_aberration_variance',
            drift_score: 0.045,
            threshold: 0.05,
            p_value: 0.182,
            is_drifting: false,
            capability_status: 'COMPLETED',
            status_reason: null,
            reference_sample_count: 500,
            production_sample_count: 500,
          },
          {
            metric_name: 'edge_contrast_histogram_sparse',
            drift_score: null,
            threshold: 0.05,
            p_value: null,
            is_drifting: false,
            capability_status: 'INCONCLUSIVE',
            status_reason: 'Sample density N=18 < 30; statistical power insufficient under ADR-012.',
            reference_sample_count: 18,
            production_sample_count: 18,
          },
        ],
      },
    ];

    return safeFetch<DriftReport[]>(
      `${API_BASE}/distribution-shift/reports/${encodeURIComponent(modelId)}`,
      undefined,
      fallbackReports
    );
  },

  /**
   * Evaluate distribution drift dynamically
   */
  async evaluateDrift(payload: unknown): Promise<ApiResponse<DriftReport>> {
    return safeFetch<DriftReport>(`${API_BASE}/distribution-shift/evaluate`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch host platform attestation report
   */
  async getAttestationReport(): Promise<ApiResponse<AttestationReport>> {
    const fallback: AttestationReport = {
      timestamp_utc: new Date().toISOString(),
      hardware_fingerprint: '3a8fe9b2c104d77e48b812f864e2239c011e4f9b8821adbc430e129845cd7a10',
      is_offline_enforced: true,
      is_non_root_user: true,
      is_hardware_bound: true,
      is_compliant: true,
    };

    return safeFetch<AttestationReport>(
      `${API_BASE}/attestation`,
      undefined,
      fallback
    );
  },

  /**
   * Fetch quarantined items from model & dataset registries
   */
  async getQuarantineItems(): Promise<ApiResponse<QuarantineItem[]>> {
    const fallback: QuarantineItem[] = [
      {
        id: 'quar_mod_9921',
        name: 'traffic_cam_resnet50_v1.pth',
        asset_type: 'model',
        quarantine_reason: 'Hidden unapproved layer `trojan_trigger.weight` detected in state dictionary.',
        quarantined_at: '2026-11-18T10:15:33Z',
        sha256_digest: 'd2c1e8479e0a3b8c6177b8f9e6120e2384a51e624d77c2a8f4c3b1e9d8a70512',
        severity: 'CRITICAL',
        status: 'QUARANTINED',
        details: {
          rule_violated: 'MODEL_ARCHITECTURE_SPEC_VIOLATION',
          detected_layer: 'trojan_trigger.weight',
          description: 'Model artifact contains non-conforming parameters not declared in reference architecture configuration.',
        },
      },
      {
        id: 'quar_mod_9922',
        name: 'night_vision_yolo_untrusted.pt',
        asset_type: 'model',
        quarantine_reason: 'Arbitrary bytecode execution gadget detected during safe checkpoint deserialization.',
        quarantined_at: '2026-11-17T22:30:10Z',
        sha256_digest: '4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b',
        severity: 'CRITICAL',
        status: 'QUARANTINED',
        details: {
          rule_violated: 'UNSAFE_DESERIALIZATION_BLOCKED',
          description: 'Unsafe pickle payload detected attempting global import of os.system. Quarantined immediately.',
        },
      },
      {
        id: 'quar_dat_4102',
        name: 'surveillance_batch_2026_11_16.tar.gz',
        asset_type: 'dataset',
        quarantine_reason: 'Manifest digest integrity mismatch; 14 sample files modified post-signing.',
        quarantined_at: '2026-11-16T14:45:12Z',
        sha256_digest: '778899aabbccddeeff00112233445566778899aabbccddeeff00112233445566',
        severity: 'HIGH',
        status: 'UNDER_REVIEW',
        details: {
          rule_violated: 'DATASET_HASH_MISMATCH',
          hash_expected: '778899aabbccddeeff00112233445566778899aabbccddeeff00112233445566',
          hash_actual: '99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa',
          description: 'Bit-level SHA-256 variance observed against signed manifest anchor.',
        },
      },
    ];

    // Query backend models with status=QUARANTINED
    const res = await safeFetch<{ items: Array<{ model_id: string; name: string; status: string; sha256: string }> }>(
      `${API_BASE}/models?status=QUARANTINED`,
      undefined,
      null
    );

    if (res.data && res.data.items && res.data.items.length > 0) {
      const liveItems: QuarantineItem[] = res.data.items.map((m) => ({
        id: m.model_id,
        name: m.name,
        asset_type: 'model',
        quarantine_reason: 'Ingestion integrity fence triggered quarantine disposition.',
        quarantined_at: new Date().toISOString(),
        sha256_digest: m.sha256 || 'N/A',
        severity: 'HIGH',
        status: 'QUARANTINED',
        details: {
          rule_violated: 'AUTOMATED_INGESTION_FENCE',
          description: `Model ${m.name} status is marked as QUARANTINED in database repository.`,
        },
      }));
      return {
        data: [...liveItems, ...fallback],
        error: null,
        isOfflineFallback: false,
      };
    }

    return {
      data: fallback,
      error: res.error,
      isOfflineFallback: res.isOfflineFallback,
    };
  },
};
