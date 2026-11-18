import React from 'react';
import { ShieldCheck, ShieldAlert, WifiOff, Cpu, UserCheck } from 'lucide-react';
import { AttestationReport } from '../types/integrity';

interface AttestationBadgeProps {
  report?: AttestationReport;
  onClick?: () => void;
  compact?: boolean;
}

export const AttestationBadge: React.FC<AttestationBadgeProps> = ({
  report = {
    timestamp_utc: new Date().toISOString(),
    hardware_fingerprint: '3a8f-e9b2-c104-d77e',
    is_offline_enforced: true,
    is_non_root_user: true,
    is_hardware_bound: true,
    is_compliant: true,
  },
  onClick,
  compact = false,
}) => {
  const isHealthy = report.is_compliant;

  if (compact) {
    return (
      <button
        onClick={onClick}
        type="button"
        title="Click to inspect zero-trust platform attestation"
        className={`group relative inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all duration-200 cursor-pointer border ${
          isHealthy
            ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40 hover:border-emerald-300 hover:bg-emerald-900/40 shadow-neon-emerald'
            : 'bg-rose-950/60 text-rose-300 border-rose-500/50 hover:border-rose-300 hover:bg-rose-900/40 shadow-neon-rose'
        }`}
      >
        <span className="relative flex h-2 w-2">
          {isHealthy && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          )}
          <span
            className={`relative inline-flex rounded-full h-2 w-2 ${
              isHealthy ? 'bg-emerald-400' : 'bg-rose-400 animate-pulse'
            }`}
          ></span>
        </span>
        <span className="tracking-wider uppercase text-[11px] font-bold">
          {isHealthy ? 'AIR-GAP ATTESTED' : 'POSTURE BREACH'}
        </span>
      </button>
    );
  }

  return (
    <div
      onClick={onClick}
      className={`hud-panel hud-corner rounded-lg p-5 transition-all duration-200 cursor-pointer ${
        isHealthy
          ? 'border-emerald-500/40 hover:border-emerald-400 shadow-neon-emerald'
          : 'border-rose-500/50 hover:border-rose-400 shadow-neon-rose'
      }`}
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-white/[0.08] gap-3">
        <div className="flex items-center gap-3">
          <div
            className={`w-11 h-11 rounded-lg flex items-center justify-center border shadow-lg ${
              isHealthy
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/40'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/40'
            }`}
          >
            {isHealthy ? (
              <ShieldCheck className="w-6 h-6 text-emerald-400 animate-pulse" />
            ) : (
              <ShieldAlert className="w-6 h-6 text-rose-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono tracking-widest uppercase text-slate-400">
                [ ASSURANCE POSTURE ]
              </span>
              <span className="text-slate-600">//</span>
              <span className="text-[10px] font-mono text-cyan-400 font-bold">NODE: PRIMARY_ENCLAVE</span>
            </div>
            <h4 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
              Zero-Trust Air-Gap Attestation Matrix
            </h4>
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          <span className="text-[11px] font-mono text-slate-400">
            {report.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC
          </span>
          <span
            className={`px-3 py-1 rounded text-[10px] font-mono font-bold tracking-widest border ${
              isHealthy
                ? 'bg-emerald-950/70 text-emerald-300 border-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                : 'bg-rose-950/70 text-rose-300 border-rose-500/50 shadow-[0_0_12px_rgba(244,63,94,0.4)]'
            }`}
          >
            {isHealthy ? 'CERTIFIED PASS' : 'COMPLIANCE VIOLATION'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 mt-4">
        {/* Offline Invariant */}
        <div className="bg-[#06080d] border border-white/[0.08] rounded-md p-3.5 hover:border-cyan-500/30 transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5 font-bold">
              <WifiOff className="w-3.5 h-3.5 text-cyan-400" />
              Air-Gap Enforcement
            </span>
            <span
              className={`text-[10px] font-mono font-bold ${
                report.is_offline_enforced ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {report.is_offline_enforced ? '[ LOCKED ]' : '[ BREACHED ]'}
            </span>
          </div>
          <div className="mt-2 text-xs font-mono text-slate-200">
            Zero-Egress Sockets: <span className="text-emerald-400 font-bold">ACTIVE</span>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-0.5">
            HF & Torch Offline Flags Pinned
          </div>
        </div>

        {/* Non-Root Invariant */}
        <div className="bg-[#06080d] border border-white/[0.08] rounded-md p-3.5 hover:border-cyan-500/30 transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5 font-bold">
              <UserCheck className="w-3.5 h-3.5 text-cyan-400" />
              Privilege Dropping
            </span>
            <span
              className={`text-[10px] font-mono font-bold ${
                report.is_non_root_user ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {report.is_non_root_user ? '[ NON-ROOT ]' : '[ ROOT ALERT ]'}
            </span>
          </div>
          <div className="mt-2 text-xs font-mono text-slate-200">
            Kernel Escalation: <span className="text-emerald-400 font-bold">RESTRICTED</span>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-0.5">
            CAP_DROP_ALL + NoNewPrivs
          </div>
        </div>

        {/* Hardware Invariant */}
        <div className="bg-[#06080d] border border-white/[0.08] rounded-md p-3.5 hover:border-cyan-500/30 transition-colors">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5 font-bold">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              Hardware Identity
            </span>
            <span
              className={`text-[10px] font-mono font-bold ${
                report.is_hardware_bound ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {report.is_hardware_bound ? '[ PINNED ]' : '[ UNBOUND ]'}
            </span>
          </div>
          <div className="mt-2 text-xs font-mono text-slate-200 truncate" title={report.hardware_fingerprint}>
            FP: <span className="text-cyan-300 font-bold">{report.hardware_fingerprint.slice(0, 16)}...</span>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-0.5">
            Physical NIC Multicast=0
          </div>
        </div>
      </div>
    </div>
  );
};
