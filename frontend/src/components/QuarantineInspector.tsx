import React, { useState } from 'react';
import {
  ShieldAlert,
  AlertOctagon,
  Box,
  Layers,
  Search,
  RefreshCw,
  AlertTriangle,
  Copy,
  Check,
  Lock,
  FileCode,
  ShieldX,
  Radiation,
  Zap,
  Plus
} from 'lucide-react';
import { QuarantineItem } from '../types/integrity';
import { useQuarantine } from '../api/hooks';

export const QuarantineInspector: React.FC = () => {
  const { items: initialItems, isLoading, error, refresh } = useQuarantine();
  const [localItems, setLocalItems] = useState<QuarantineItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<QuarantineItem | null>(null);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'model' | 'dataset'>('all');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [isInjectingThreat, setIsInjectingThreat] = useState(false);

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const displayItems = localItems.length > 0 ? localItems : initialItems;

  const filtered = displayItems.filter((item) => {
    const matchesSearch =
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.quarantine_reason.toLowerCase().includes(search.toLowerCase()) ||
      item.id.toLowerCase().includes(search.toLowerCase());
    const matchesType = filterType === 'all' || item.asset_type === filterType;
    return matchesSearch && matchesType;
  });

  const activeItem = selectedItem || filtered[0] || null;
  const criticalCount = displayItems.filter((i) => i.severity === 'CRITICAL').length;
  const modelsCount = displayItems.filter((i) => i.asset_type === 'model').length;
  const datasetsCount = displayItems.filter((i) => i.asset_type === 'dataset').length;

  // Live simulation: simulate intercepting and quarantining a backdoored model
  const simulateThreatInterception = () => {
    setIsInjectingThreat(true);
    setTimeout(() => {
      const randId = Math.random().toString(16).slice(2, 8);
      const newThreat: QuarantineItem = {
        id: `threat-quar-${randId}`,
        name: `trojan_vision_transformer_b16_${randId}.pt`,
        asset_type: 'model',
        quarantine_reason: 'SPECTRAL_SIGNATURE_ANOMALY: Top eigenvector outlier score 4.82 > 2.5 threshold in attention projection layer.',
        quarantined_at: new Date().toISOString(),
        sha256_digest: `8a7b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f${randId}`,
        severity: 'CRITICAL',
        status: 'QUARANTINED',
        details: {
          rule_violated: 'RULE_SPECTRAL_SIGNATURE_BACKDOOR',
          detected_layer: 'blocks.11.attn.proj.weight (Spectral Eigenvalue Outlier)',
          description: 'Latent representation PCA clusters separated into two distinct Gaussian mixtures under trigger patch presence.',
        },
      };

      setLocalItems([newThreat, ...displayItems]);
      setSelectedItem(newThreat);
      setIsInjectingThreat(false);
    }, 450);
  };

  return (
    <div className="space-y-5">
      {/* Top Threat Containment Header */}
      <div className="hud-panel hud-corner rounded-lg p-5 space-y-4">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-md bg-gradient-to-br from-rose-500/20 to-red-600/10 border border-rose-500/40 flex items-center justify-center text-rose-400 shadow-neon-rose">
                <ShieldAlert className="w-4 h-4 animate-pulse" />
              </div>
              <h2 className="text-base font-bold text-slate-100 font-mono tracking-wider flex items-center gap-2">
                THREAT QUARANTINE // INODE-FENCED ISOLATION CHAMBER
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-950/80 text-rose-300 border border-rose-500/50 font-mono animate-pulse">
                  READ-ONLY FENCE
                </span>
              </h2>
            </div>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Physically isolated containment zone for backdoor-injected weights, poisoned datasets, and unsafe pickle artifacts.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto">
            <div className="relative flex-1 lg:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search artifact name, ID, reason..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-[#06080d] border border-white/[0.1] rounded-md pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-rose-500 font-mono"
              />
            </div>

            {/* Simulate Threat Interception Button */}
            <button
              onClick={simulateThreatInterception}
              disabled={isInjectingThreat}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-rose-700 to-amber-700 hover:from-rose-600 hover:to-amber-600 text-slate-100 font-bold text-xs font-mono rounded-md shadow-neon-rose transition-all duration-200 cursor-pointer active:scale-95 disabled:opacity-50"
            >
              {isInjectingThreat ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>INTERCEPTING...</span>
                </>
              ) : (
                <>
                  <Plus className="w-3.5 h-3.5" />
                  <span>SIMULATE INTERCEPTION</span>
                </>
              )}
            </button>

            <button
              onClick={() => refresh()}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-slate-200 text-xs font-mono rounded-md border border-white/[0.1] transition-colors cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-rose-400' : ''}`} />
              <span>SYNC</span>
            </button>
          </div>
        </div>

        {/* Quick Threat Matrix Readout */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-2 border-t border-white/[0.06] text-xs font-mono">
          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase text-slate-400 flex items-center justify-between">
              <span>Total Fenced Threats</span>
              <Lock className="w-3.5 h-3.5 text-rose-400" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-xl font-bold text-slate-100">{displayItems.length}</span>
              <span className="text-xs text-slate-500">artifacts locked</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Zero runtime execution permitted</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase text-slate-400 flex items-center justify-between">
              <span>Critical Severity</span>
              <AlertOctagon className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className={`text-xl font-bold ${criticalCount > 0 ? 'text-rose-400 drop-shadow-[0_0_8px_rgba(244,63,94,0.6)]' : 'text-slate-100'}`}>
                {criticalCount}
              </span>
              <span className="text-xs text-slate-500">high priority</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Backdoor / Poisoning attacks</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase text-slate-400 flex items-center justify-between">
              <span>Quarantined Models</span>
              <Box className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-xl font-bold text-slate-100">{modelsCount}</span>
              <span className="text-xs text-slate-500">PyTorch .pt/.pth</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Spectral / Activation flagged</div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase text-slate-400 flex items-center justify-between">
              <span>Corrupted Datasets</span>
              <Layers className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-xl font-bold text-slate-100">{datasetsCount}</span>
              <span className="text-xs text-slate-500">manifests fenced</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">SHA-256 hash or path violation</div>
          </div>
        </div>

        {/* Visual Threat Containment Vault Perimeter Radar */}
        <div className="p-3.5 rounded-lg bg-[#06080d] border border-rose-500/20 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-full bg-rose-950/40 border border-rose-500/40 shrink-0">
              <Radiation className="w-5 h-5 text-rose-400 animate-spin" style={{ animationDuration: '8s' }} />
              <span className="absolute inset-0 rounded-full border border-rose-400/40 radar-wave"></span>
            </div>
            <div>
              <div className="text-xs font-mono font-bold text-rose-300 tracking-wider">
                INODE PARTITION ISOLATION PERIMETER: ARMORED
              </div>
              <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                Physical directory path <code className="text-cyan-300">storage/quarantine/</code> permissions set to 0400 (Read-Only). No process can load weights.
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 self-end sm:self-auto text-xs font-mono">
            <span className="px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-500/30 flex items-center gap-1 font-bold">
              <Lock className="w-3 h-3 text-emerald-400" />
              INODE-FENCE: SECURED
            </span>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-950/30 border border-rose-500/30 rounded-md flex items-center gap-2 text-xs text-rose-300 font-mono">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>Notice: {error}. Presenting cached quarantine snapshot records.</span>
          </div>
        )}
      </div>

      {/* Filter and Mode Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono px-1">
        <span className="text-slate-400 flex items-center gap-1.5">
          <ShieldX className="w-3.5 h-3.5 text-rose-400" />
          <span>FENCED ASSET REGISTRY ({filtered.length} / {displayItems.length})</span>
        </span>

        <div className="inline-flex rounded-md bg-[#06080d] p-1 border border-white/[0.08] text-xs font-mono">
          <button
            onClick={() => setFilterType('all')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterType === 'all'
                ? 'bg-slate-800 text-slate-100 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({displayItems.length})
          </button>
          <button
            onClick={() => setFilterType('model')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterType === 'model'
                ? 'bg-rose-950 text-rose-300 font-bold border border-rose-500/40 shadow-neon-rose'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Models ({modelsCount})
          </button>
          <button
            onClick={() => setFilterType('dataset')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterType === 'dataset'
                ? 'bg-amber-950 text-amber-300 font-bold border border-amber-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Datasets ({datasetsCount})
          </button>
        </div>
      </div>

      {/* Main Grid: Containment List + Inspector */}
      {isLoading && displayItems.length === 0 ? (
        <div className="h-64 hud-panel rounded-lg flex flex-col items-center justify-center gap-3">
          <RefreshCw className="w-6 h-6 text-rose-400 animate-spin" />
          <span className="text-xs font-mono text-slate-400">Loading isolated containment records...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* List Column */}
          <div className="lg:col-span-7 space-y-3">
            {filtered.length === 0 ? (
              <div className="p-8 text-center hud-panel rounded-lg font-mono text-xs text-slate-500">
                No quarantined artifacts match query "{search}".
              </div>
            ) : (
              filtered.map((item) => {
                const isSelected = activeItem?.id === item.id;
                const isCritical = item.severity === 'CRITICAL';

                return (
                  <div
                    key={item.id}
                    onClick={() => setSelectedItem(item)}
                    className={`hud-panel rounded-lg p-4 transition-all duration-200 cursor-pointer ${
                      isSelected
                        ? 'border-rose-500 bg-slate-900/95 shadow-neon-rose scale-[1.01]'
                        : isCritical
                        ? 'border-rose-500/40 bg-rose-950/20 hover:border-rose-400'
                        : 'hover:border-cyan-500/40 hover:bg-slate-900/60'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-3">
                        <div
                          className={`w-9 h-9 rounded flex items-center justify-center shrink-0 border ${
                            item.asset_type === 'model'
                              ? 'bg-rose-500/10 text-rose-400 border-rose-500/40'
                              : 'bg-amber-500/10 text-amber-400 border-amber-500/40'
                          }`}
                        >
                          {item.asset_type === 'model' ? (
                            <Box className="w-4 h-4 animate-pulse" />
                          ) : (
                            <Layers className="w-4 h-4" />
                          )}
                        </div>

                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono font-bold text-slate-100">
                              {item.name}
                            </span>
                            <span className="px-1.5 py-0.2 text-[9px] font-mono rounded bg-[#06080d] border border-white/[0.1] text-slate-400">
                              {item.asset_type.toUpperCase()}
                            </span>
                          </div>
                          <p className="text-xs text-rose-300 mt-1 font-mono leading-relaxed">
                            {item.quarantine_reason}
                          </p>
                        </div>
                      </div>

                      <span
                        className={`px-2.5 py-1 text-[10px] font-mono font-bold rounded border shrink-0 ${
                          isCritical
                            ? 'bg-rose-950 text-rose-300 border-rose-500/50 shadow-[0_0_10px_rgba(244,63,94,0.4)] animate-pulse'
                            : 'bg-amber-950 text-amber-300 border-amber-500/40'
                        }`}
                      >
                        [{item.severity}]
                      </span>
                    </div>

                    <div className="mt-3 pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[10px] font-mono text-slate-500">
                      <span>UUID: {item.id}</span>
                      <span>{item.quarantined_at.replace('T', ' ').slice(0, 19)} UTC</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Forensic Inspector Column */}
          <div className="lg:col-span-5">
            {activeItem ? (
              <div className="hud-panel hud-corner rounded-lg p-5 space-y-4 sticky top-20">
                <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                  <div className="flex items-center gap-2">
                    <AlertOctagon className="w-4 h-4 text-rose-400" />
                    <h3 className="text-xs font-bold text-slate-100 font-mono uppercase tracking-wider">
                      FORENSIC THREAT DIAGNOSTICS
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-500/40 font-bold">
                    [{activeItem.status}]
                  </span>
                </div>

                <div className="space-y-3 text-xs font-mono">
                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Quarantined Artifact</span>
                    <div className="text-slate-100 font-bold mt-1 text-sm">{activeItem.name}</div>
                  </div>

                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Security Invariant Violated</span>
                    <div className="text-rose-300 bg-[#06080d] p-2 rounded border border-rose-500/40 mt-1 font-bold flex items-center gap-2">
                      <Zap className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      <span>{activeItem.details.rule_violated || 'INTEGRITY_SECURITY_VIOLATION'}</span>
                    </div>
                  </div>

                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Forensic Evidence & Vector</span>
                    <p className="text-slate-300 mt-1 leading-relaxed bg-[#06080d] p-3 rounded border border-white/[0.08] text-[11px]">
                      {activeItem.details.description || activeItem.quarantine_reason}
                    </p>
                  </div>

                  {activeItem.details.detected_layer && (
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase">Flagged Tensor / Weight</span>
                      <div className="text-amber-300 bg-[#06080d] p-2 rounded border border-amber-500/40 text-[11px] mt-1 flex items-center gap-1.5">
                        <FileCode className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                        <span className="truncate">{activeItem.details.detected_layer}</span>
                      </div>
                    </div>
                  )}

                  <div>
                    <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase">
                      <span>Artifact SHA-256 Digest</span>
                      <button
                        onClick={() => copyToClipboard(activeItem.sha256_digest, 'sha')}
                        className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
                      >
                        {copiedKey === 'sha' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedKey === 'sha' ? 'COPIED' : 'COPY'}</span>
                      </button>
                    </div>
                    <div className="text-slate-300 bg-[#06080d] p-2 rounded border border-white/[0.08] break-all select-all text-[11px] mt-1">
                      {activeItem.sha256_digest}
                    </div>
                  </div>

                  <div>
                    <span className="text-slate-500 block text-[10px] uppercase">Inode Containment Path</span>
                    <div className="text-slate-400 bg-[#06080d] p-2 rounded border border-white/[0.08] text-[11px] mt-1 truncate">
                      /storage/quarantine/{activeItem.asset_type}s/{activeItem.id}.fence
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-white/[0.08] flex items-center justify-between text-xs">
                  <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1 font-semibold">
                    <Lock className="w-3 h-3 text-emerald-400" />
                    Inode access permanently fenced
                  </span>
                  <div className="text-[10px] font-mono px-2 py-1 bg-slate-900 border border-white/[0.1] rounded text-slate-300">
                    ADR-012 DEFENSE
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-64 hud-panel rounded-lg flex items-center justify-center text-xs text-slate-500 font-mono">
                Select a quarantined asset to inspect forensic diagnostics
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
