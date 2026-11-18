import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Terminal } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('VisionGuard UI ErrorBoundary caught an exception:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="card-bezel rounded-lg p-6 space-y-4 my-4 font-mono border-rose-500/50 bg-rose-950/20 shadow-bezel-rose">
          <div className="flex items-center gap-2.5 text-rose-400">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <h3 className="text-sm font-bold tracking-wider uppercase">
              {this.props.fallbackTitle || 'DEFENSIVE UI BOUNDARY FAULT DETECTED'}
            </h3>
          </div>
          <p className="text-xs text-rose-300/80 leading-relaxed">
            An unhandled runtime exception was intercepted by the VisionGuard UI defense barrier. The application state remains intact.
          </p>
          {this.state.error && (
            <div className="p-3 bg-[#06080d] rounded border border-rose-500/30 text-[11px] text-slate-300 overflow-x-auto flex items-start gap-2">
              <Terminal className="w-3.5 h-3.5 text-rose-400 mt-0.5 shrink-0" />
              <code>{this.state.error.message}</code>
            </div>
          )}
          <button
            onClick={this.handleReset}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs rounded border border-white/[0.1] transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
            <span>RESET COMPONENT BARRIER</span>
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
