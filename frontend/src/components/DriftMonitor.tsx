import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Filter,
  BarChart2,
  TrendingUp,
  SlidersHorizontal,
  HelpCircle,
  RefreshCw,
  Cpu,
  Zap,
  Flame,
  RotateCcw
} from 'lucide-react';
import { DriftMetric } from '../types/integrity';
import { useDriftReports } from '../api/hooks';

export const DriftMonitor: React.FC = () => {
  const [selectedModelId, setSelectedModelId] = useState<string>('resnet50_tamper_resistant_v2');
  const { metrics: initialMetrics, isDrifting: initialDrifting, hasInconclusiveMetrics, isLoading, refresh } =
    useDriftReports(selectedModelId);

  const [filterMode, setFilterMode] = useState<'all' | 'drift_only' | 'normal_only' | 'inconclusive'>('all');
  const [syntheticDriftShift, setSyntheticDriftShift] = useState<number>(0); // 0 = nominal, >0 = simulated drift

  // If synthetic drift is injected, dynamically modify the first metric for real-time visual demonstration
  const metrics = initialMetrics.map((m, idx) => {
    if (idx === 0 && syntheticDriftShift > 0) {
      return {
        ...m,
        drift_score: Number((0.034 + syntheticDriftShift * 0.25).toFixed(4)),
        p_value: syntheticDriftShift > 0.4 ? 0.00012 : 0.142,
        is_drifting: syntheticDriftShift > 0.4,
        status_reason: syntheticDriftShift > 0.4 ? 'Simulated adversarial distribution divergence' : m.status_reason,
      };
    }
    return m;
  });

  const isDrifting = initialDrifting || metrics.some((m) => m.is_drifting);

  const filteredMetrics = metrics.filter((m) => {
    if (filterMode === 'drift_only') return m.is_drifting;
    if (filterMode === 'normal_only') return !m.is_drifting && m.capability_status === 'COMPLETED';
    if (filterMode === 'inconclusive') return m.capability_status === 'INCONCLUSIVE';
    return true;
  });

  const totalEvaluated = metrics.length;
  const driftDetectedCount = metrics.filter((m) => m.is_drifting).length;
  const inconclusiveCount = metrics.filter((m) => m.capability_status === 'INCONCLUSIVE').length;
  const stableCount = totalEvaluated - driftDetectedCount - inconclusiveCount;

  return (
    <div className="space-y-5">
      {/* Header and Telemetry Stats Bar */}
      <div className="hud-panel hud-corner rounded-lg p-5 space-y-4">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-md bg-gradient-to-br from-indigo-500/20 to-purple-600/10 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-[0_0_12px_rgba(99,102,241,0.3)]">
                <Activity className="w-4 h-4 animate-pulse" />
              </div>
              <h2 className="text-base font-bold text-slate-100 font-mono tracking-wider flex items-center gap-2">
                DISTRIBUTION SHIFT // STATISTICAL DRIFT & OOD MONITOR
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-950/80 text-indigo-300 border border-indigo-500/40 font-mono">
                  BONFERRONI α=0.05
                </span>
              </h2>
            </div>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Real-time two-sample Kolmogorov-Smirnov and 1D Wasserstein divergence tests against frozen reference datasets.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto">
            <div className="flex items-center gap-1.5 bg-[#06080d] border border-white/[0.1] rounded-md px-3 py-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <input
                type="text"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                placeholder="Model Identifier..."
                className="bg-transparent text-xs text-slate-200 font-mono focus:outline-none w-48 sm:w-56"
                title="Monitored Architecture Target"
              />
            </div>

            <button
              onClick={() => refresh()}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-slate-200 text-xs font-mono rounded-md border border-white/[0.1] transition-colors cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
              <span>EVALUATE</span>
            </button>
          </div>
        </div>

        {/* Global Drift Telemetry Matrix Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-2 border-t border-white/[0.06]">
          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase font-mono text-slate-400 flex items-center justify-between">
              <span>Shift Equilibrium</span>
              <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span
                className={`text-base font-bold font-mono tracking-wider ${
                  isDrifting ? 'text-rose-400 drop-shadow-[0_0_8px_rgba(244,63,94,0.6)]' : 'text-emerald-400'
                }`}
              >
                {isDrifting ? 'SHIFT DETECTED' : 'IN EQUILIBRIUM'}
              </span>
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-0.5">
              {isDrifting ? 'Statistical alarm triggered' : 'Nominal manifold bounds'}
            </div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase font-mono text-slate-400 flex items-center justify-between">
              <span>Drifting Channels</span>
              <TrendingUp className="w-3.5 h-3.5 text-rose-400" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className={`text-xl font-bold font-mono ${driftDetectedCount > 0 ? 'text-rose-400' : 'text-slate-100'}`}>
                {driftDetectedCount}
              </span>
              <span className="text-xs text-slate-500 font-mono">
                / {totalEvaluated} channels
              </span>
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-0.5">
              Statistically significant (p &lt; α)
            </div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase font-mono text-slate-400 flex items-center justify-between">
              <span>ADR-012 Epistemic Gate</span>
              <HelpCircle className="w-3.5 h-3.5 text-amber-500" />
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className={`text-xl font-bold font-mono ${inconclusiveCount > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
                {inconclusiveCount}
              </span>
              <span className="text-xs text-slate-500 font-mono">
                low density (N &lt; 30)
              </span>
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-0.5">
              Strictly non-hallucinatory
            </div>
          </div>

          <div className="hud-panel rounded-md p-3">
            <div className="text-[10px] uppercase font-mono text-slate-400 flex items-center justify-between">
              <span>Hypothesis Threshold</span>
              <BarChart2 className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="mt-1.5 text-base font-bold font-mono text-cyan-300">
              Bonferroni α = 0.05
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-0.5">
              FWER Family-wise correction
            </div>
          </div>
        </div>

        {/* Interactive Holographic SVG Distribution Curve Visualizer */}
        <div className="mt-4 p-4 rounded-lg bg-[#06080d] border border-cyan-500/20 space-y-3">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
                FEATURE MANIFOLD PROBABILITY DENSITY FUNCTION (PDF) DIVERGENCE
              </span>
            </div>

            {/* Interactive Simulation Controls */}
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="text-slate-400 text-[10px]">SIMULATE OOD ATTACK:</span>
              <button
                onClick={() => setSyntheticDriftShift(syntheticDriftShift === 0 ? 0.75 : 0)}
                className={`px-2.5 py-1 rounded text-[11px] font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
                  syntheticDriftShift > 0
                    ? 'bg-rose-950 text-rose-300 border border-rose-500/60 shadow-neon-rose'
                    : 'bg-slate-900 text-slate-300 border border-white/[0.1] hover:border-cyan-500/40'
                }`}
              >
                {syntheticDriftShift > 0 ? (
                  <>
                    <RotateCcw className="w-3 h-3 text-rose-400" />
                    <span>RESET EQUILIBRIUM</span>
                  </>
                ) : (
                  <>
                    <Flame className="w-3 h-3 text-amber-400" />
                    <span>INJECT DRIFT ANOMALY</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* SVG Canvas comparing Baseline vs Production Bell Curves */}
          <div className="relative h-32 w-full overflow-hidden rounded bg-[#04060a] border border-white/[0.04]">
            <svg className="w-full h-full" viewBox="0 0 600 120" preserveAspectRatio="none">
              <defs>
                <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="productionGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={isDrifting ? '#f43f5e' : '#10b981'} stopOpacity="0.35" />
                  <stop offset="100%" stopColor={isDrifting ? '#f43f5e' : '#10b981'} stopOpacity="0" />
                </linearGradient>
              </defs>

              {/* Grid lines */}
              <line x1="0" y1="30" x2="600" y2="30" stroke="#1e293b" strokeDasharray="4 4" strokeWidth="0.5" />
              <line x1="0" y1="70" x2="600" y2="70" stroke="#1e293b" strokeDasharray="4 4" strokeWidth="0.5" />
              <line x1="0" y1="110" x2="600" y2="110" stroke="#334155" strokeWidth="1" />

              {/* Reference Baseline Curve (Cyan Bell) */}
              <path
                d="M 50 110 Q 180 110, 240 25 T 300 25 T 450 110 L 450 110 Z"
                fill="url(#baselineGrad)"
              />
              <path
                d="M 50 110 Q 180 110, 240 25 T 300 25 T 450 110"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.5"
              />

              {/* Dynamic Production Curve (Emerald or Shifted Rose) */}
              {syntheticDriftShift > 0 ? (
                <>
                  <path
                    d="M 140 110 Q 270 110, 340 30 T 400 30 T 550 110 L 550 110 Z"
                    fill="url(#productionGrad)"
                    className="transition-all duration-700"
                  />
                  <path
                    d="M 140 110 Q 270 110, 340 30 T 400 30 T 550 110"
                    fill="none"
                    stroke="#f43f5e"
                    strokeWidth="2.5"
                    strokeDasharray="4 2"
                    className="transition-all duration-700"
                  />
                  {/* Maximum KS Divergence Distance Marker D */}
                  <line x1="280" y1="30" x2="280" y2="95" stroke="#f43f5e" strokeWidth="2" strokeDasharray="2 2" />
                  <circle cx="280" cy="30" r="3" fill="#f43f5e" />
                  <circle cx="280" cy="95" r="3" fill="#f43f5e" />
                  <text x="290" y="65" fill="#f43f5e" fontSize="10" fontFamily="monospace" fontWeight="bold">
                    D_max = 0.284 (p &lt; 0.001)
                  </text>
                </>
              ) : (
                <>
                  <path
                    d="M 55 110 Q 185 110, 245 28 T 305 28 T 455 110 L 455 110 Z"
                    fill="url(#productionGrad)"
                    className="transition-all duration-700"
                  />
                  <path
                    d="M 55 110 Q 185 110, 245 28 T 305 28 T 455 110"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    className="transition-all duration-700"
                  />
                </>
              )}
            </svg>

            {/* Legend Overlay */}
            <div className="absolute top-2 left-3 flex items-center gap-4 text-[10px] font-mono">
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="w-2.5 h-0.5 bg-cyan-400 inline-block"></span>
                Baseline Reference (N=1000)
              </span>
              <span className={`flex items-center gap-1.5 ${isDrifting ? 'text-rose-400' : 'text-emerald-400'}`}>
                <span className={`w-2.5 h-0.5 inline-block ${isDrifting ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
                Observed Production (N=250)
              </span>
            </div>
          </div>
        </div>

        {/* Epistemic Gating Warning Notice */}
        {hasInconclusiveMetrics && (
          <div className="p-3 bg-amber-950/40 border border-amber-500/50 rounded-md flex items-center gap-2.5 text-xs text-amber-200 font-mono shadow-neon-amber">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              <strong>ADR-012 EPISTEMIC GATE ACTIVE:</strong> Production sample volume is under statistical power minimum ($N &lt; 30$).
              Channels are strictly classified as <strong>INCONCLUSIVE</strong> without hallucinating synthetic p-values.
            </span>
          </div>
        )}
      </div>

      {/* Filter and Mode Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono px-1">
        <span className="text-slate-400 flex items-center gap-1.5">
          <Filter className="w-3.5 h-3.5 text-cyan-400" />
          <span>STATISTICAL MANIFOLD VECTORS ({filteredMetrics.length} / {totalEvaluated})</span>
        </span>

        <div className="inline-flex rounded-md bg-[#06080d] p-1 border border-white/[0.08] text-xs font-mono">
          <button
            onClick={() => setFilterMode('all')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterMode === 'all'
                ? 'bg-slate-800 text-cyan-300 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({totalEvaluated})
          </button>
          <button
            onClick={() => setFilterMode('drift_only')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterMode === 'drift_only'
                ? 'bg-rose-950 text-rose-300 font-bold border border-rose-500/40 shadow-neon-rose'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Drift ({driftDetectedCount})
          </button>
          <button
            onClick={() => setFilterMode('normal_only')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterMode === 'normal_only'
                ? 'bg-emerald-950 text-emerald-300 font-bold border border-emerald-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Stable ({stableCount})
          </button>
          <button
            onClick={() => setFilterMode('inconclusive')}
            className={`px-3 py-1 rounded transition-colors cursor-pointer ${
              filterMode === 'inconclusive'
                ? 'bg-amber-950 text-amber-300 font-bold border border-amber-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Inconclusive ({inconclusiveCount})
          </button>
        </div>
      </div>

      {/* Metrics Feed Table */}
      {isLoading && metrics.length === 0 ? (
        <div className="h-48 hud-panel rounded-lg flex flex-col items-center justify-center gap-3">
          <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
          <span className="text-xs font-mono text-slate-400">Evaluating empirical hypothesis vectors...</span>
        </div>
      ) : filteredMetrics.length === 0 ? (
        <div className="p-8 text-center hud-panel rounded-lg font-mono text-xs text-slate-500">
          No statistical vectors match filter "{filterMode}".
        </div>
      ) : (
        <div className="hud-panel rounded-lg overflow-hidden divide-y divide-white/[0.06]">
          {filteredMetrics.map((metric: DriftMetric) => {
            const isDrift = metric.is_drifting;
            const isInconclusive = metric.capability_status === 'INCONCLUSIVE';
            const isKS = metric.metric_name.toLowerCase().includes('ks') || metric.metric_name.toLowerCase().includes('kolmogorov') || metric.p_value !== null;
            
            const scoreVal = metric.drift_score ?? 0;
            const threshVal = metric.threshold || 0.1;
            const percentage = Math.min(Math.round((scoreVal / (threshVal * 1.5)) * 100), 100);

            return (
              <div
                key={metric.metric_name}
                className={`p-4 transition-all duration-200 ${
                  isInconclusive
                    ? 'bg-amber-950/10 hover:bg-amber-950/20'
                    : isDrift
                    ? 'bg-rose-950/20 hover:bg-rose-950/30'
                    : 'hover:bg-slate-900/50'
                }`}
              >
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                  {/* Channel descriptor */}
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center gap-2.5">
                      {/* Animated soundwave equalizer bars */}
                      <div className="flex items-end gap-0.5 h-3.5 w-4 shrink-0">
                        <div className={`w-0.5 rounded-full ${isDrift ? 'bg-rose-400' : 'bg-cyan-400'} animate-eq-1`} />
                        <div className={`w-0.5 rounded-full ${isDrift ? 'bg-rose-400' : 'bg-cyan-400'} animate-eq-2`} />
                        <div className={`w-0.5 rounded-full ${isDrift ? 'bg-rose-400' : 'bg-cyan-400'} animate-eq-3`} />
                        <div className={`w-0.5 rounded-full ${isDrift ? 'bg-rose-400' : 'bg-cyan-400'} animate-eq-4`} />
                      </div>

                      <span className="text-xs font-mono font-bold text-slate-100">
                        {metric.metric_name}
                      </span>
                      <span className="px-2 py-0.5 bg-[#06080d] border border-white/[0.08] text-slate-400 text-[9.5px] font-mono rounded">
                        {isKS ? '2-SAMPLE KS' : 'WASSERSTEIN DIVERGENCE'}
                      </span>
                    </div>

                    <div className="text-[11px] font-mono text-slate-400 flex flex-wrap items-center gap-2">
                      <span>N_prod: <strong className="text-slate-200">{metric.production_sample_count}</strong></span>
                      <span className="text-slate-600">//</span>
                      <span>N_base: <strong className="text-slate-200">{metric.reference_sample_count}</strong></span>
                      {metric.status_reason && (
                        <>
                          <span className="text-slate-600">//</span>
                          <span className="text-amber-400 font-semibold">{metric.status_reason}</span>
                        </>
                      )}
                    </div>

                    {/* Proportional visual bar */}
                    {!isInconclusive && (
                      <div className="w-full max-w-md bg-[#06080d] h-2 rounded-full overflow-hidden mt-1.5 border border-white/[0.06] p-0.5">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isDrift ? 'bg-gradient-to-r from-rose-500 to-amber-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]' : 'bg-gradient-to-r from-cyan-500 to-emerald-400'
                          }`}
                          style={{ width: `${Math.max(percentage, 8)}%` }}
                        />
                      </div>
                    )}
                  </div>

                  {/* Quantitative telemetry readouts */}
                  <div className="flex items-center gap-5 self-start lg:self-center">
                    <div className="grid grid-cols-3 gap-2.5 text-xs font-mono">
                      <div className="bg-[#06080d] px-2.5 py-1.5 rounded border border-white/[0.06]">
                        <span className="text-[9px] text-slate-500 uppercase block">Distance</span>
                        <span
                          className={`font-bold ${
                            isInconclusive
                              ? 'text-amber-400'
                              : isDrift
                              ? 'text-rose-400'
                              : 'text-slate-100'
                          }`}
                        >
                          {isInconclusive ? 'GATED' : metric.drift_score !== null ? metric.drift_score.toFixed(4) : '—'}
                        </span>
                      </div>

                      <div className="bg-[#06080d] px-2.5 py-1.5 rounded border border-white/[0.06]">
                        <span className="text-[9px] text-slate-500 uppercase block">P-Value</span>
                        <span
                          className={`font-bold ${
                            isInconclusive
                              ? 'text-amber-400'
                              : metric.p_value !== null && metric.p_value < 0.05
                              ? 'text-rose-400 drop-shadow-[0_0_6px_rgba(244,63,94,0.6)]'
                              : 'text-slate-300'
                          }`}
                        >
                          {isInconclusive ? 'INCONCLUSIVE' : metric.p_value !== null ? metric.p_value.toFixed(5) : '—'}
                        </span>
                      </div>

                      <div className="bg-[#06080d] px-2.5 py-1.5 rounded border border-white/[0.06]">
                        <span className="text-[9px] text-slate-500 uppercase block">Threshold</span>
                        <span className="text-slate-400">
                          {isKS ? `α=${metric.threshold}` : `≤${metric.threshold.toFixed(2)}`}
                        </span>
                      </div>
                    </div>

                    {/* Status badge */}
                    <div className="shrink-0 w-36 flex justify-end">
                      {isInconclusive ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-bold bg-amber-950/60 text-amber-300 border border-amber-500/40">
                          <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
                          INCONCLUSIVE
                        </span>
                      ) : isDrift ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-bold bg-rose-950/70 text-rose-300 border border-rose-500/50 shadow-neon-rose animate-pulse">
                          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                          DRIFT ALERT
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-bold bg-emerald-950/50 text-emerald-300 border border-emerald-500/40">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                          STABLE
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
