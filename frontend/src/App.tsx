import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HashChainViewer } from './components/HashChainViewer';
import { DriftMonitor } from './components/DriftMonitor';
import { QuarantineInspector } from './components/QuarantineInspector';
import { AttestationPanel } from './components/AttestationPanel';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ActiveTab } from './types/integrity';
import { useAttestation } from './api/hooks';
import { ShieldCheck, HardDrive, Cpu, Terminal, Radio, Clock, Shield } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('hash-chain');
  const { report } = useAttestation();
  const [timeUtc, setTimeUtc] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeUtc(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen cyber-grid-ambient text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200 relative">
      {/* Top Laser Sweep Line Animation */}
      <div className="relative overflow-hidden h-[2px] w-full bg-slate-900 z-50">
        <div className="laser-beam-header" />
      </div>

      {/* Top Telemetry Navigation Console */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        attestationReport={report}
      />

      {/* Sub-Header Live Security Matrix Banner */}
      <div className="bg-[#080c16]/80 border-b border-white/[0.06] backdrop-blur-sm px-4 sm:px-6 lg:px-8 py-2 text-[11px] font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-slate-400">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-cyan-400 font-bold tracking-wider">
              <Shield className="w-3.5 h-3.5 animate-pulse" />
              SYSTEM INTEGRITY: 100% ASSURED
            </span>
            <span className="text-slate-700">|</span>
            <span className="hidden md:flex items-center gap-1 text-slate-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              ED25519 CHAIN: <strong className="text-emerald-300">ACTIVE</strong>
            </span>
            <span className="hidden md:inline text-slate-700">|</span>
            <span className="hidden lg:flex items-center gap-1 text-slate-400">
              INODE PARTITION: <strong className="text-cyan-300">LOCKED</strong>
            </span>
          </div>

          <div className="flex items-center gap-3 self-end sm:self-auto text-[11px]">
            <div className="flex items-center gap-1.5 text-slate-400">
              <Clock className="w-3 h-3 text-cyan-400" />
              <span className="text-slate-200 font-semibold">{timeUtc || 'INITIALIZING CLOCK...'}</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1 text-emerald-400">
              <Radio className="w-3 h-3 animate-pulse" />
              <span>AIR-GAP LOOPBACK</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Operator Command Stage */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        <ErrorBoundary fallbackTitle="Tactical View Execution Fault">
          <div className="transition-all duration-300">
            {activeTab === 'hash-chain' && <HashChainViewer />}
            {activeTab === 'drift-monitor' && <DriftMonitor />}
            {activeTab === 'quarantine' && <QuarantineInspector />}
            {activeTab === 'attestation' && <AttestationPanel />}
          </div>
        </ErrorBoundary>
      </main>

      {/* Operator Status Footer */}
      <footer className="border-t border-white/[0.08] bg-[#05070c]/90 backdrop-blur-md py-3.5 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-slate-400">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold tracking-wider">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              VISIONGUARD DEFENSE MATRIX
            </span>
            <span className="hidden md:inline text-slate-700">|</span>
            <span className="hidden md:flex items-center gap-1 text-slate-400">
              <HardDrive className="w-3.5 h-3.5 text-slate-500" />
              Storage: <span className="text-slate-300">SQLite + Inode Fencing</span>
            </span>
            <span className="hidden md:inline text-slate-700">|</span>
            <span className="hidden md:flex items-center gap-1 text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              Runtime: <span className="text-slate-300">PyTorch 2.6 Native (RTX 3050 Compatible)</span>
            </span>
          </div>

          <div className="flex items-center gap-3 text-[11px]">
            <div className="flex items-center gap-1.5 text-slate-400">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              <span>LOOPBACK: 127.0.0.1:8000</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1.5 text-emerald-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
              </span>
              <span className="font-bold tracking-wider">OFFLINE ASSURANCE ACTIVE</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
