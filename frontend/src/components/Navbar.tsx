import React from 'react';
import {
  Shield,
  Link2,
  Activity,
  ShieldAlert,
  Lock,
  Radio,
  Terminal,
  Cpu,
  Zap
} from 'lucide-react';
import { ActiveTab, AttestationReport } from '../types/integrity';
import { AttestationBadge } from './AttestationBadge';

interface NavbarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  attestationReport?: AttestationReport;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  attestationReport,
}) => {
  const tabs = [
    {
      id: 'hash-chain' as ActiveTab,
      idx: '01',
      label: 'INFERENCE INTEGRITY',
      subtext: 'Ed25519 Receipt Chain',
      icon: Link2,
    },
    {
      id: 'drift-monitor' as ActiveTab,
      idx: '02',
      label: 'DISTRIBUTION SHIFT',
      subtext: 'KS / Wasserstein Gating',
      icon: Activity,
    },
    {
      id: 'quarantine' as ActiveTab,
      idx: '03',
      label: 'THREAT QUARANTINE',
      subtext: 'Isolated Storage Fencing',
      icon: ShieldAlert,
    },
    {
      id: 'attestation' as ActiveTab,
      idx: '04',
      label: 'AIR-GAP ATTESTATION',
      subtext: 'Hardware Fingerprint Posture',
      icon: Lock,
    },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#06080d]/90 backdrop-blur-xl border-b border-white/[0.08] shadow-[0_8px_32px_rgba(0,0,0,0.8)]">
      {/* Top telemetry status strip */}
      <div className="border-b border-white/[0.04] bg-[#080c16]/90 px-4 sm:px-6 lg:px-8 py-1.5 text-[11px] font-mono text-slate-400 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-cyan-400 font-bold tracking-wider">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            VISIONGUARD // OPERATOR CONSOLE
          </span>
          <span className="text-slate-700">|</span>
          <span className="hidden md:flex items-center gap-1.5 text-slate-400">
            <Cpu className="w-3 h-3 text-slate-500" />
            NODE: <span className="text-slate-200">VG-EDGE-PRIMARY</span>
          </span>
          <span className="hidden md:inline text-slate-700">|</span>
          <span className="hidden lg:flex items-center gap-1.5 text-slate-400">
            <Zap className="w-3 h-3 text-emerald-400" />
            HEURISTIC: <span className="text-slate-200">NON-PARAMETRIC GATING</span>
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-500/30">
            <Radio className="w-3 h-3 animate-pulse" />
            <span className="text-[10px] tracking-widest font-bold uppercase">
              AIR-GAP ENFORCED [127.0.0.1]
            </span>
          </div>
          <span className="text-slate-700">|</span>
          <AttestationBadge
            compact
            report={attestationReport}
            onClick={() => onSelectTab('attestation')}
          />
        </div>
      </div>

      {/* Main command bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Brand identifier */}
          <div className="flex items-center gap-3">
            <div className="relative group">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500/20 to-emerald-500/10 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.35)] transition-transform duration-300 group-hover:scale-105">
                <Shield className="w-5 h-5 text-cyan-300" />
              </div>
              <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400"></span>
              </span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono font-black text-sm tracking-widest text-slate-100 bg-gradient-to-r from-slate-100 via-cyan-100 to-slate-300 bg-clip-text text-transparent">
                  VISIONGUARD
                </span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-500/40 font-bold tracking-wider shadow-[0_0_8px_rgba(6,182,212,0.2)]">
                  OFFLINE-FIRST
                </span>
              </div>
              <p className="text-[10px] font-mono text-slate-400">
                AI Lifecycle Integrity Assurance
              </p>
            </div>
          </div>

          {/* Nav Tabs Desktop */}
          <nav className="hidden md:flex items-center gap-1.5">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;

              return (
                <button
                  key={tab.id}
                  onClick={() => onSelectTab(tab.id)}
                  className={`group relative flex items-center gap-2 px-3.5 py-2 rounded-md font-mono transition-all duration-200 cursor-pointer ${
                    isActive
                      ? 'bg-slate-900/90 text-cyan-300 border border-cyan-500/50 shadow-neon-cyan'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/50 border border-transparent'
                  }`}
                >
                  <span
                    className={`text-[10px] font-bold ${
                      isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-400'
                    }`}
                  >
                    [{tab.idx}]
                  </span>
                  <Icon
                    className={`w-4 h-4 transition-transform duration-200 group-hover:scale-110 ${
                      isActive ? 'text-cyan-300 drop-shadow-[0_0_6px_rgba(6,182,212,0.8)]' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />
                  <div className="text-left">
                    <span className="text-xs font-semibold tracking-wider block">
                      {tab.label}
                    </span>
                  </div>
                  {isActive && (
                    <span className="absolute -bottom-1 left-3 right-3 h-[2px] bg-gradient-to-r from-cyan-500 via-emerald-400 to-cyan-500 shadow-[0_0_10px_rgba(6,182,212,1)]" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Mobile Navigation strip */}
        <div className="flex md:hidden overflow-x-auto py-2 border-t border-white/[0.06] gap-1.5 scrollbar-none">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-mono whitespace-nowrap shrink-0 border transition-all ${
                  isActive
                    ? 'bg-slate-900 text-cyan-300 border-cyan-500/50 shadow-neon-cyan'
                    : 'text-slate-400 border-transparent hover:bg-slate-900/40'
                }`}
              >
                <span className="text-[10px] text-slate-500">[{tab.idx}]</span>
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
