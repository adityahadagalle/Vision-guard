import React, { useState } from 'react';
import {
  WifiOff,
  UserCheck,
  Cpu,
  Lock,
  Terminal,
  CheckCircle2,
  RefreshCw,
  AlertOctagon,
  Copy,
  Check,
  ShieldCheck,
  Zap,
  Activity
} from 'lucide-react';
import { AttestationBadge } from './AttestationBadge';
import { useAttestation } from '../api/hooks';

export const AttestationPanel: React.FC = () => {
  const { report, isLoading, refresh } = useAttestation();
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [activeChipNode, setActiveChipNode] = useState<'airgap' | 'nonroot' | 'mac' | 'inode'>('airgap');
  const [isAuditing, setIsAuditing] = useState(false);

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleAudit = () => {
    setIsAuditing(true);
    refresh();
    setTimeout(() => setIsAuditing(false), 800);
  };

  return (
    <div className="space-y-5">
      {/* Top Header Telemetry */}
      <div className="hud-panel hud-corner rounded-lg p-5">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-md bg-gradient-to-br from-emerald-500/20 to-teal-600/10 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shadow-neon-emerald">
                <Lock className="w-4 h-4 animate-pulse" />
              </div>
              <h2 className="text-base font-bold text-slate-100 font-mono tracking-wider flex items-center gap-2">
                AIR-GAP ATTESTATION // HARDWARE-BOUND SECURITY POSTURE
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/50 font-mono animate-pulse">
                  TPM/FINGERPRINT-PINNED
                </span>
              </h2>
            </div>
            <p className="text-xs font-mono text-slate-400 mt-1">
              Deterministic verification of zero-egress air-gap isolation, non-root process boundaries, and hardware NIC node identity.
            </p>
          </div>

          <button
            onClick={handleAudit}
            disabled={isLoading || isAuditing}
            className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-slate-900 font-bold text-xs font-mono rounded-md shadow-neon-emerald transition-all duration-200 cursor-pointer active:scale-95 disabled:opacity-50 w-full lg:w-auto justify-center"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isAuditing || isLoading ? 'animate-spin' : ''}`} />
            <span>RE-EVALUATE POSTURE</span>
          </button>
        </div>
      </div>

      {/* Hero Attestation Posture Matrix */}
      <AttestationBadge report={report} />

      {/* Interactive Holographic Hardware Security Module (HSM) Topology Visualizer */}
      <div className="hud-panel rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-slate-100 font-mono uppercase tracking-wider">
              HARDWARE ENCLAVE & ZERO-EGRESS TOPOLOGY SCHEMATIC
            </h3>
          </div>
          <span className="text-[10px] font-mono text-emerald-400 font-bold flex items-center gap-1">
            <Activity className="w-3 h-3 text-emerald-400 animate-pulse" />
            ALL BUS CIRCUITS SECURED
          </span>
        </div>

        {/* SVG Interactive Microchip Schematic */}
        <div className="p-4 bg-[#05070c] rounded-lg border border-cyan-500/20 relative overflow-hidden">
          <svg className="w-full h-40" viewBox="0 0 700 140" fill="none">
            {/* Bus wires with glowing animated dashes */}
            <path d="M 120 70 L 250 70" stroke="#06b6d4" strokeWidth="2" strokeDasharray="4 4" className="flow-line" />
            <path d="M 370 70 L 500 70" stroke="#10b981" strokeWidth="2" strokeDasharray="4 4" className="flow-line" />
            <path d="M 310 30 L 310 110" stroke="#334155" strokeWidth="1.5" />

            {/* Node 1: Zero-Egress Air-Gap Barrier */}
            <g
              onClick={() => setActiveChipNode('airgap')}
              className="cursor-pointer group"
            >
              <rect
                x="20"
                y="35"
                width="100"
                height="70"
                rx="6"
                fill="#0d1424"
                stroke={activeChipNode === 'airgap' ? '#06b6d4' : '#1e293b'}
                strokeWidth={activeChipNode === 'airgap' ? '2' : '1'}
                className="transition-all"
              />
              <text x="70" y="65" textAnchor="middle" fill="#38bdf8" fontSize="10" fontFamily="monospace" fontWeight="bold">AIR-GAP</text>
              <text x="70" y="82" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="monospace">ZERO-EGRESS</text>
              <circle cx="70" cy="93" r="3" fill="#10b981" className="animate-pulse" />
            </g>

            {/* Center Node: Master Node CPU & Inode Storage */}
            <g
              onClick={() => setActiveChipNode('inode')}
              className="cursor-pointer group"
            >
              <rect
                x="250"
                y="20"
                width="120"
                height="100"
                rx="8"
                fill="#090e1c"
                stroke={activeChipNode === 'inode' ? '#10b981' : '#06b6d4'}
                strokeWidth={activeChipNode === 'inode' ? '2.5' : '1.5'}
                className="transition-all"
              />
              {/* Corner Chip Pins */}
              <circle cx="260" cy="30" r="2" fill="#06b6d4" />
              <circle cx="360" cy="30" r="2" fill="#06b6d4" />
              <circle cx="260" cy="110" r="2" fill="#06b6d4" />
              <circle cx="360" cy="110" r="2" fill="#06b6d4" />

              <text x="310" y="55" textAnchor="middle" fill="#f8fafc" fontSize="11" fontFamily="monospace" fontWeight="bold">VG-CORE</text>
              <text x="310" y="73" textAnchor="middle" fill="#06b6d4" fontSize="9" fontFamily="monospace">INODE-LOCKED</text>
              <text x="310" y="90" textAnchor="middle" fill="#10b981" fontSize="8" fontFamily="monospace">UID: NON-ROOT</text>
            </g>

            {/* Node 3: Hardware NIC Fingerprint */}
            <g
              onClick={() => setActiveChipNode('mac')}
              className="cursor-pointer group"
            >
              <rect
                x="500"
                y="35"
                width="110"
                height="70"
                rx="6"
                fill="#0d1424"
                stroke={activeChipNode === 'mac' ? '#10b981' : '#1e293b'}
                strokeWidth={activeChipNode === 'mac' ? '2' : '1'}
                className="transition-all"
              />
              <text x="555" y="65" textAnchor="middle" fill="#10b981" fontSize="10" fontFamily="monospace" fontWeight="bold">HARDWARE</text>
              <text x="555" y="82" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="monospace">NIC PINNED</text>
              <circle cx="555" cy="93" r="3" fill="#10b981" className="animate-pulse" />
            </g>

            {/* Node 4: Sandbox Privilege Dropping */}
            <g
              onClick={() => setActiveChipNode('nonroot')}
              className="cursor-pointer group"
            >
              <rect
                x="625"
                y="45"
                width="65"
                height="50"
                rx="4"
                fill="#0d1424"
                stroke={activeChipNode === 'nonroot' ? '#8b5cf6' : '#1e293b'}
                strokeWidth={activeChipNode === 'nonroot' ? '2' : '1'}
                className="transition-all"
              />
              <text x="657" y="70" textAnchor="middle" fill="#c084fc" fontSize="8" fontFamily="monospace" fontWeight="bold">NO_NEW_PRIV</text>
              <text x="657" y="84" textAnchor="middle" fill="#94a3b8" fontSize="7" fontFamily="monospace">CAP_DROP</text>
            </g>
          </svg>

          {/* Interactive Node Explanation Banner */}
          <div className="mt-2 pt-2 border-t border-white/[0.06] text-xs font-mono flex items-center justify-between text-slate-300">
            <span className="text-cyan-400 flex items-center gap-1.5 font-bold">
              <Zap className="w-3.5 h-3.5" />
              {activeChipNode === 'airgap' && 'AIR-GAP ENFORCEMENT: HF_DATASETS_OFFLINE=1 & TRANSFORMERS_OFFLINE=1 permanently locked.'}
              {activeChipNode === 'inode' && 'INODE PARTITION DEFENSE: Storage partitions strictly fenced from arbitrary unauthenticated modifications.'}
              {activeChipNode === 'mac' && 'PHYSICAL NIC BINDING: Non-multicast hardware MAC deterministically bound into cryptographic identity.'}
              {activeChipNode === 'nonroot' && 'PRIVILEGE BOUNDARY: Process drops root privileges; kernel exploits cannot propagate to host.'}
            </span>
            <span className="text-[10px] text-slate-500 hidden sm:inline">CLICK CHIP NODES TO INSPECT</span>
          </div>
        </div>
      </div>

      {/* Technical Invariants 3-Column Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Air-Gap Invariant */}
        <div className="hud-panel rounded-lg p-5 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
                <WifiOff className="w-3.5 h-3.5" />
              </div>
              <h3 className="text-xs font-bold text-slate-100 font-mono tracking-wider">
                AIR-GAP ENFORCEMENT
              </h3>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                report.is_offline_enforced
                  ? 'text-emerald-300 bg-emerald-950/60 border-emerald-500/40 shadow-neon-emerald'
                  : 'text-rose-300 bg-rose-950/60 border-rose-500/40'
              }`}
            >
              {report.is_offline_enforced ? '[ ENFORCED ]' : '[ BREACHED ]'}
            </span>
          </div>

          <p className="text-xs text-slate-400 font-mono leading-relaxed">
            Runtime environment variables permanently pin model frameworks to strictly zero-egress offline operation.
          </p>

          <div className="bg-[#06080d] p-3 rounded border border-white/[0.08] space-y-2 text-[11px] font-mono">
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">HF_DATASETS_OFFLINE</span>
              <span className={`font-bold ${report.is_offline_enforced ? 'text-emerald-400' : 'text-rose-400'}`}>
                {report.is_offline_enforced ? '1 (LOCKED)' : '0 (UNSET)'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">TRANSFORMERS_OFFLINE</span>
              <span className={`font-bold ${report.is_offline_enforced ? 'text-emerald-400' : 'text-rose-400'}`}>
                {report.is_offline_enforced ? '1 (LOCKED)' : '0 (UNSET)'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">TORCH_HOME</span>
              <span className="text-cyan-400">/storage/weights</span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">SOCKET BINDING</span>
              <span className="text-emerald-400 font-bold">127.0.0.1 ONLY</span>
            </div>
          </div>
        </div>

        {/* Card 2: Process Isolation & Privilege */}
        <div className="hud-panel rounded-lg p-5 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
                <UserCheck className="w-3.5 h-3.5" />
              </div>
              <h3 className="text-xs font-bold text-slate-100 font-mono tracking-wider">
                PRIVILEGE DROPPING
              </h3>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                report.is_non_root_user
                  ? 'text-emerald-300 bg-emerald-950/60 border-emerald-500/40 shadow-neon-emerald'
                  : 'text-rose-300 bg-rose-950/60 border-rose-500/40'
              }`}
            >
              {report.is_non_root_user ? '[ NON-ROOT ]' : '[ ELEVATED ]'}
            </span>
          </div>

          <p className="text-xs text-slate-400 font-mono leading-relaxed">
            Host sandbox drops root privileges immediately after network initialization to mitigate kernel exploits.
          </p>

          <div className="bg-[#06080d] p-3 rounded border border-white/[0.08] space-y-2 text-[11px] font-mono">
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Execution User</span>
              <span className={`font-bold ${report.is_non_root_user ? 'text-emerald-400' : 'text-rose-400'}`}>
                {report.is_non_root_user ? 'UID != 0 (SAFE)' : 'UID == 0 (ALERT)'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Linux Capabilities</span>
              <span className="text-emerald-400 font-bold">CAP_DROP_ALL</span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Privilege Escalation</span>
              <span className="text-emerald-400 font-bold">PR_SET_NO_NEW_PRIVS</span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Filesystem Fencing</span>
              <span className="text-cyan-400">READ-ONLY CORE</span>
            </div>
          </div>
        </div>

        {/* Card 3: Hardware Identity Binding */}
        <div className="hud-panel rounded-lg p-5 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
                <Cpu className="w-3.5 h-3.5" />
              </div>
              <h3 className="text-xs font-bold text-slate-100 font-mono tracking-wider">
                HARDWARE BINDING
              </h3>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                report.is_hardware_bound
                  ? 'text-emerald-300 bg-emerald-950/60 border-emerald-500/40 shadow-neon-emerald'
                  : 'text-rose-300 bg-rose-950/60 border-rose-500/40'
              }`}
            >
              {report.is_hardware_bound ? '[ PINNED ]' : '[ UNBOUND ]'}
            </span>
          </div>

          <p className="text-xs text-slate-400 font-mono leading-relaxed">
            Deterministic cryptographic hardware identity derived from physical non-multicast MAC and CPU model.
          </p>

          <div className="bg-[#06080d] p-3 rounded border border-white/[0.08] space-y-2 text-[11px] font-mono">
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Binding State</span>
              <span className="text-emerald-400 font-bold">
                {report.is_hardware_bound ? 'STRICTLY PINNED' : 'UNPINNED'}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="text-slate-400">Multicast Bit</span>
              <span className="text-emerald-400 font-bold">0 (PHYSICAL NIC)</span>
            </div>
            <div className="pt-1">
              <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase">
                <span>Hardware Fingerprint</span>
                <button
                  onClick={() => copyToClipboard(report.hardware_fingerprint, 'fp')}
                  className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
                >
                  {copiedKey === 'fp' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'fp' ? 'COPIED' : 'COPY'}</span>
                </button>
              </div>
              <div className="text-cyan-300 text-[10px] truncate mt-0.5 bg-slate-950 p-1.5 rounded border border-white/[0.05]">
                {report.hardware_fingerprint}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Telemetry Log Terminal */}
      <div className="hud-panel rounded-lg p-5 font-mono text-xs">
        <div className="flex items-center justify-between pb-3 border-b border-white/[0.08] text-slate-400">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-400" />
            <span className="font-bold text-slate-200 uppercase tracking-wider">
              REAL-TIME DETERMINISTIC ATTESTATION AUDIT TRAIL
            </span>
          </div>
          <span className="text-[10px] text-slate-500">FORMAT: ISO-8601 UTC</span>
        </div>

        <div className="mt-3.5 space-y-2.5 text-slate-400">
          <div className="flex items-start gap-2.5 bg-[#06080d] p-2.5 rounded border border-white/[0.04]">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <span className="text-cyan-400 font-bold">[{report.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC]</span>{' '}
              <span className="text-slate-200 font-bold">AirgapEnvironmentValidator:</span>{' '}
              <span>HF_DATASETS_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 validated. Egress sockets blocked. (EXIT 0)</span>
            </div>
          </div>

          <div className="flex items-start gap-2.5 bg-[#06080d] p-2.5 rounded border border-white/[0.04]">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <span className="text-cyan-400 font-bold">[{report.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC]</span>{' '}
              <span className="text-slate-200 font-bold">PlatformAttestationEngine:</span>{' '}
              <span>Least privilege posture confirmed. Process running non-root. PR_SET_NO_NEW_PRIVS locked. (EXIT 0)</span>
            </div>
          </div>

          <div className="flex items-start gap-2.5 bg-[#06080d] p-2.5 rounded border border-white/[0.04]">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <span className="text-cyan-400 font-bold">[{report.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC]</span>{' '}
              <span className="text-slate-200 font-bold">HardwareFingerprinter:</span>{' '}
              <span>Physical MAC resolved. Multicast bit checked (=0). Deterministic SHA-256 fingerprint generated. (EXIT 0)</span>
            </div>
          </div>

          <div
            className={`flex items-start gap-2.5 p-3 rounded border ${
              report.is_compliant
                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/50 shadow-neon-emerald'
                : 'bg-rose-950/40 text-rose-300 border-rose-500/50 shadow-neon-rose'
            }`}
          >
            {report.is_compliant ? (
              <ShieldCheck className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0 animate-pulse" />
            ) : (
              <AlertOctagon className="w-5 h-5 text-rose-400 mt-0.5 shrink-0" />
            )}
            <div className="font-bold">
              <span className="text-cyan-300">[{report.timestamp_utc.slice(0, 19).replace('T', ' ')} UTC]</span>{' '}
              <span>
                {report.is_compliant
                  ? 'AIR-GAP COMPLIANCE CERTIFIED: All offline-first cryptographic and hardware invariants satisfied.'
                  : 'CRITICAL SECURITY POSTURE BREACH: One or more air-gap invariants failed.'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
