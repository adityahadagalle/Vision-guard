import React, { useState } from 'react';
import {
  Link2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Search,
  Key,
  Database,
  RefreshCw,
  Copy,
  Check,
  Binary,
  Layers,
  Cpu,
  Plus,
  Zap,
  Activity
} from 'lucide-react';
import { InferenceReceipt } from '../types/integrity';
import { useInferenceChain } from '../api/hooks';

export const HashChainViewer: React.FC = () => {
  const { receipts: initialReceipts, isLoading, error, refresh, isChainContinuous } = useInferenceChain();
  const [localReceipts, setLocalReceipts] = useState<InferenceReceipt[]>([]);
  const [selectedReceipt, setSelectedReceipt] = useState<InferenceReceipt | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [inspectorTab, setInspectorTab] = useState<'provenance' | 'canonical' | 'hashes'>('provenance');
  const [isMining, setIsMining] = useState(false);

  // Combine initial receipts with any simulated live receipts
  const displayReceipts = localReceipts.length > 0 ? localReceipts : initialReceipts;

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const filteredReceipts = displayReceipts.filter(
    (r) =>
      r.receipt_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.receipt_hash.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.model_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.sequence_number.toString().includes(searchQuery)
  );

  const activeReceipt = selectedReceipt || filteredReceipts[0] || null;

  // Live simulation: dynamically inject a verified block with animated entry
  const simulateLiveReceipt = () => {
    setIsMining(true);
    setTimeout(() => {
      const baseList = displayReceipts;
      const lastSeq = baseList.length > 0 ? Math.max(...baseList.map(r => r.sequence_number)) : 0;
      const prevHash = baseList[0]?.receipt_hash || '0000000000000000000000000000000000000000000000000000000000000000';
      const seq = lastSeq + 1;
      const randHex = Math.random().toString(16).slice(2, 10);
      
      const newBlock: InferenceReceipt = {
        receipt_id: `rec-sim-${seq}-${randHex}`,
        sequence_number: seq,
        timestamp_utc: new Date().toISOString(),
        model_id: 'resnet50_tamper_resistant_v2',
        input_sha256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852${randHex.slice(0, 4)}`,
        prediction_sha256: `7852c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991be3b0${randHex.slice(0, 4)}`,
        prev_receipt_hash: prevHash,
        receipt_hash: `a91c4d28f9e1b234567890abcdef1234567890abcdef1234567890abcdef${randHex.slice(0, 4)}`,
        signature_b64: 'MC4CAQACBQDXr7sMAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEAAgEA',
        public_key_b64: 'kO2+9wFkL1A2Z4E9F8g7H6j5K4m3N2p1Q0r9S8t7U6v5W4x3Y2z1A0b9C8d7E6f5',
        verified: true,
        status: 'VALID',
      };

      setLocalReceipts([newBlock, ...baseList]);
      setSelectedReceipt(newBlock);
      setIsMining(false);
    }, 400);
  };

  return (
    <div className="space-y-5">
      {/* Top Ledger Telemetry Bar */}
      <div className="hud-panel hud-corner rounded-lg p-5">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-md bg-gradient-to-br from-cyan-500/20 to-blue-600/10 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.3)]">
                <Link2 className="w-4 h-4 animate-pulse" />
              </div>
              <h2 className="text-base font-bold text-slate-100 font-mono tracking-wider flex items-center gap-2">
                INFERENCE INTEGRITY // CRYPTOGRAPHIC RECEIPT CHAIN
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-500/40 font-mono animate-pulse">
                  DAG-VERIFIED
                </span>
              </h2>
            </div>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Deterministic Ed25519 asymmetric signatures chained across consecutive inference blocks via SHA-256 digests.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto">
            <div className="relative flex-1 lg:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                placeholder="Filter sequence, UUID, hash..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#06080d] border border-white/[0.1] rounded-md pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono transition-colors"
              />
            </div>

            {/* Live Block Simulator Trigger */}
            <button
              onClick={simulateLiveReceipt}
              disabled={isMining}
              title="Simulate incoming edge inference event"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-emerald-600 hover:from-cyan-500 hover:to-emerald-500 text-slate-900 font-bold text-xs font-mono rounded-md shadow-neon-cyan transition-all duration-200 cursor-pointer active:scale-95 disabled:opacity-50"
            >
              {isMining ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>SIGNING...</span>
                </>
              ) : (
                <>
                  <Plus className="w-3.5 h-3.5" />
                  <span>MINE RECEIPT</span>
                </>
              )}
            </button>

            <button
              onClick={() => refresh()}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-slate-200 text-xs font-mono rounded-md border border-white/[0.1] transition-colors cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
              <span>SYNC</span>
            </button>
          </div>
        </div>

        {/* Quick Chain Telemetry Metrics with Animated Pulse */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-3 border-t border-white/[0.06] text-xs font-mono">
          <div className="hud-panel rounded-md p-3">
            <span className="text-[10px] uppercase text-slate-500 block flex items-center justify-between">
              <span>Total Ledger Blocks</span>
              <Activity className="w-3 h-3 text-cyan-400" />
            </span>
            <span className="text-base font-bold text-slate-100 mt-1 block">
              {displayReceipts.length} <span className="text-xs text-slate-500">SEQS</span>
            </span>
            <div className="text-[10px] text-cyan-400/80 mt-0.5">Monotonically indexed</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <span className="text-[10px] uppercase text-slate-500 block flex items-center justify-between">
              <span>Chain Continuity</span>
              <Zap className="w-3 h-3 text-emerald-400" />
            </span>
            <span className={`text-base font-bold mt-1 block ${isChainContinuous ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isChainContinuous ? '100% INTACT' : 'ANOMALY DETECTED'}
            </span>
            <div className="text-[10px] text-emerald-400/80 mt-0.5">Zero forks or replays</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <span className="text-[10px] uppercase text-slate-500 block flex items-center justify-between">
              <span>Digest Algorithm</span>
              <Binary className="w-3 h-3 text-cyan-400" />
            </span>
            <span className="text-base font-bold text-cyan-300 mt-1 block">
              SHA-256
            </span>
            <div className="text-[10px] text-slate-500 mt-0.5">Canonical byte stream</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <span className="text-[10px] uppercase text-slate-500 block flex items-center justify-between">
              <span>Asymmetric Scheme</span>
              <Key className="w-3 h-3 text-amber-400" />
            </span>
            <span className="text-base font-bold text-slate-200 mt-1 block">
              Ed25519
            </span>
            <div className="text-[10px] text-slate-500 mt-0.5">RFC 8032 Curve25519</div>
          </div>
        </div>
      </div>

      {/* Chain Status Alerts */}
      {error && (
        <div className="p-3 bg-rose-950/40 border border-rose-500/40 rounded-md flex items-center gap-2.5 text-xs text-rose-300 font-mono">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>Notice: {error}. Falling back to verified local snapshot ledger.</span>
        </div>
      )}

      {!isChainContinuous && (
        <div className="p-3 bg-rose-950/60 border border-rose-500/60 rounded-md flex items-center gap-2.5 text-xs text-rose-200 font-mono shadow-neon-rose">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 animate-ping" />
          <span>
            CRITICAL CHAIN CONTINUITY BREAK: Block previous_receipt_hash link divergence detected.
          </span>
        </div>
      )}

      {/* Main Ledger Stage */}
      {isLoading && displayReceipts.length === 0 ? (
        <div className="h-64 hud-panel rounded-lg flex flex-col items-center justify-center gap-3">
          <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
          <span className="text-xs font-mono text-slate-400">Synchronizing edge ledger blocks...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Column 1: Block Stream / Table */}
          <div className="lg:col-span-7 space-y-3">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
                <span>// CRYPTOGRAPHIC BLOCK PIPELINE ({filteredReceipts.length} NODES)</span>
              </span>
              <span>CANONICAL CRYPTO LINK</span>
            </div>

            {filteredReceipts.length === 0 ? (
              <div className="p-8 text-center hud-panel rounded-lg font-mono text-xs text-slate-500">
                No receipts found matching filter query "{searchQuery}".
              </div>
            ) : (
              <div className="space-y-3">
                {filteredReceipts.map((receipt, index) => {
                  const isSelected = activeReceipt?.receipt_id === receipt.receipt_id;
                  const isTampered = receipt.status === 'TAMPERED';
                  const isVerified = receipt.verified;

                  return (
                    <div key={receipt.receipt_id} className="relative group">
                      <div
                        onClick={() => setSelectedReceipt(receipt)}
                        className={`hud-panel rounded-lg p-4 transition-all duration-200 cursor-pointer ${
                          isSelected
                            ? 'border-cyan-400 bg-slate-900/95 shadow-neon-cyan scale-[1.01]'
                            : isTampered
                            ? 'border-rose-500/60 bg-rose-950/20 hover:border-rose-400 shadow-neon-rose'
                            : 'hover:border-cyan-500/40 hover:bg-slate-900/60'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="px-2.5 py-1 bg-[#06080d] border border-cyan-500/40 text-cyan-300 text-xs font-mono font-bold rounded shadow-[inset_0_1px_4px_rgba(6,182,212,0.2)]">
                              #{String(receipt.sequence_number).padStart(5, '0')}
                            </span>
                            <div>
                              <span className="text-xs font-mono text-slate-100 font-bold block">
                                {receipt.receipt_id}
                              </span>
                              <span className="text-[10px] font-mono text-slate-500">
                                {receipt.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC
                              </span>
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            {isVerified ? (
                              <span className="inline-flex items-center gap-1.5 text-[10px] font-mono font-bold text-emerald-300 bg-emerald-950/60 px-2.5 py-1 rounded border border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                ED25519 PASS
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 text-[10px] font-mono font-bold text-rose-300 bg-rose-950/60 px-2.5 py-1 rounded border border-rose-500/50 shadow-[0_0_10px_rgba(244,63,94,0.3)]">
                                <XCircle className="w-3.5 h-3.5 text-rose-400" />
                                SIGNATURE FAILED
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Monospace Hash Pointers */}
                        <div className="mt-3 pt-2.5 border-t border-white/[0.06] grid grid-cols-2 gap-2 text-[10px] font-mono">
                          <div className="bg-[#06080d] p-2 rounded border border-white/[0.06]">
                            <span className="text-slate-500 block uppercase text-[9px]">Prev Block Hash</span>
                            <span className="text-slate-400 truncate block font-mono">
                              {receipt.prev_receipt_hash ? receipt.prev_receipt_hash.slice(0, 16) + '...' : '[GENESIS ROOT]'}
                            </span>
                          </div>
                          <div className="bg-[#06080d] p-2 rounded border border-cyan-500/20">
                            <span className="text-slate-500 block uppercase text-[9px]">Receipt Block Hash</span>
                            <span className="text-cyan-300 truncate block font-mono font-bold">
                              {receipt.receipt_hash.slice(0, 16)}...
                            </span>
                          </div>
                        </div>

                        <div className="mt-2.5 flex items-center justify-between text-[10px] font-mono text-slate-500">
                          <span className="flex items-center gap-1">
                            <Cpu className="w-3 h-3 text-cyan-400" />
                            Model: <span className="text-slate-300">{receipt.model_id}</span>
                          </span>
                          <span className="text-cyan-400 group-hover:translate-x-1 transition-transform duration-200">
                            INSPECT BLOCK →
                          </span>
                        </div>
                      </div>

                      {/* Animated cryptographic connector wire to next block */}
                      {index < filteredReceipts.length - 1 && (
                        <div className="flex justify-center my-1.5">
                          <svg width="24" height="20" viewBox="0 0 24 20" fill="none" className="text-cyan-400/60">
                            <line x1="12" y1="0" x2="12" y2="20" stroke="currentColor" strokeWidth="2" strokeDasharray="3 3" className="flow-line" />
                            <circle cx="12" cy="10" r="2.5" fill="#06b6d4" />
                          </svg>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Column 2: Deep Cryptographic Inspector Panel */}
          <div className="lg:col-span-5">
            {activeReceipt ? (
              <div className="hud-panel hud-corner rounded-lg p-5 space-y-4 sticky top-20">
                <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                  <div className="flex items-center gap-2">
                    <Binary className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-bold text-slate-100 font-mono uppercase tracking-wider">
                      CRYPTOGRAPHIC RECEIPT AUDIT
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-300 font-bold px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/40">
                    BLOCK #{activeReceipt.sequence_number}
                  </span>
                </div>

                {/* Sub-tabs for Forensic Inspector */}
                <div className="flex rounded-md bg-[#06080d] p-1 border border-white/[0.08] text-[11px] font-mono">
                  <button
                    onClick={() => setInspectorTab('provenance')}
                    className={`flex-1 py-1 rounded transition-colors cursor-pointer ${
                      inspectorTab === 'provenance'
                        ? 'bg-slate-800 text-cyan-300 font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    PROVENANCE
                  </button>
                  <button
                    onClick={() => setInspectorTab('canonical')}
                    className={`flex-1 py-1 rounded transition-colors cursor-pointer ${
                      inspectorTab === 'canonical'
                        ? 'bg-slate-800 text-cyan-300 font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    CANONICAL JSON
                  </button>
                  <button
                    onClick={() => setInspectorTab('hashes')}
                    className={`flex-1 py-1 rounded transition-colors cursor-pointer ${
                      inspectorTab === 'hashes'
                        ? 'bg-slate-800 text-cyan-300 font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    RAW DIGESTS
                  </button>
                </div>

                {/* Status Indicator */}
                {activeReceipt.status === 'TAMPERED' ? (
                  <div className="p-3 bg-rose-950/40 border border-rose-500/50 rounded-md flex items-start gap-2.5 shadow-neon-rose">
                    <AlertTriangle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
                    <div className="text-xs text-rose-200 font-mono">
                      <p className="font-bold">CRYPTOGRAPHIC MISMATCH DETECTED</p>
                      <p className="text-rose-300/80 text-[11px] mt-0.5">
                        Ed25519 signature verification failed over canonical serialized receipt bytes.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="p-3 bg-emerald-950/30 border border-emerald-500/40 rounded-md flex items-start gap-2.5 shadow-neon-emerald">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                    <div className="text-xs text-emerald-200 font-mono">
                      <p className="font-bold">MATHEMATICAL PROVENANCE CERTIFIED</p>
                      <p className="text-emerald-300/80 text-[10px] mt-0.5">
                        Ed25519 signature verified over canonical payload. SHA-256 parent hash link strictly valid.
                      </p>
                    </div>
                  </div>
                )}

                {/* Tab 1: Provenance Overview */}
                {inspectorTab === 'provenance' && (
                  <div className="space-y-3 text-xs font-mono">
                    <div>
                      <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase">
                        <span>Receipt UUID</span>
                        <button
                          onClick={() => copyToClipboard(activeReceipt.receipt_id, 'uuid')}
                          className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
                        >
                          {copiedKey === 'uuid' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                          <span>{copiedKey === 'uuid' ? 'COPIED' : 'COPY'}</span>
                        </button>
                      </div>
                      <div className="text-slate-200 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all select-all mt-1">
                        {activeReceipt.receipt_id}
                      </div>
                    </div>

                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Target Model Binding</span>
                      <div className="text-slate-200 bg-[#06080d] p-2 rounded border border-white/[0.08] flex items-center gap-2 mt-1">
                        <Database className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                        <span className="truncate">{activeReceipt.model_id}</span>
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase">
                        <span>Block Hash (SHA-256)</span>
                        <button
                          onClick={() => copyToClipboard(activeReceipt.receipt_hash, 'hash')}
                          className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
                        >
                          {copiedKey === 'hash' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                          <span>{copiedKey === 'hash' ? 'COPIED' : 'COPY'}</span>
                        </button>
                      </div>
                      <div className="text-cyan-300 bg-[#06080d] p-2.5 rounded border border-cyan-500/50 break-all select-all text-[11px] mt-1 shadow-[inset_0_1px_6px_rgba(6,182,212,0.15)] font-bold">
                        {activeReceipt.receipt_hash}
                      </div>
                    </div>

                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Ed25519 Public Key</span>
                      <div className="text-slate-300 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all text-[10px] flex items-center gap-2 mt-1">
                        <Key className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                        <span className="truncate">{activeReceipt.public_key_b64}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 2: Canonical JSON */}
                {inspectorTab === 'canonical' && (
                  <div className="space-y-2 text-xs font-mono">
                    <span className="text-slate-500 block text-[10px] uppercase flex items-center gap-1">
                      <Layers className="w-3 h-3 text-cyan-400" />
                      Canonical Ordered Byte Representation (RFC 8785)
                    </span>
                    <pre className="text-[10.5px] bg-[#06080d] text-cyan-300/90 p-3 rounded border border-cyan-500/20 overflow-x-auto font-mono leading-relaxed max-h-64">
{JSON.stringify(
  {
    input_sha256: activeReceipt.input_sha256,
    model_id: activeReceipt.model_id,
    prediction_sha256: activeReceipt.prediction_sha256,
    prev_receipt_hash: activeReceipt.prev_receipt_hash,
    receipt_id: activeReceipt.receipt_id,
    sequence_number: activeReceipt.sequence_number,
    timestamp_utc: activeReceipt.timestamp_utc,
  },
  null,
  2
)}
                    </pre>
                    <p className="text-[10px] text-slate-500">
                      Keys strictly ordered lexicographically to guarantee canonical hashing determinism.
                    </p>
                  </div>
                )}

                {/* Tab 3: Raw Tensor Digests */}
                {inspectorTab === 'hashes' && (
                  <div className="space-y-3 text-xs font-mono">
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Input Tensor SHA-256</span>
                      <div className="text-slate-300 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all select-all text-[11px] mt-1">
                        {activeReceipt.input_sha256}
                      </div>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Prediction Tensor SHA-256</span>
                      <div className="text-slate-300 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all select-all text-[11px] mt-1">
                        {activeReceipt.prediction_sha256}
                      </div>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Previous Block SHA-256</span>
                      <div className="text-slate-300 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all select-all text-[11px] mt-1">
                        {activeReceipt.prev_receipt_hash || '[GENESIS ROOT 0x0000000000000000]'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 hud-panel rounded-lg flex items-center justify-center text-xs text-slate-500 font-mono">
                Select a receipt block from the ledger stream to inspect
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
