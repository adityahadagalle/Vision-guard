/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#05070c',
          darker: '#080c16',
          panel: '#0b101d',
          card: '#0f1628',
          cardInner: '#141d33',
          border: '#1e293b',
          borderSubtle: 'rgba(255, 255, 255, 0.08)',
          borderHover: 'rgba(6, 182, 212, 0.4)',
          textMuted: '#64748b',
          textSecondary: '#94a3b8',
          textPrimary: '#f8fafc',
          neonCyan: '#06b6d4',
          neonGreen: '#10b981',
          neonAmber: '#f59e0b',
          neonRose: '#f43f5e',
          neonPurple: '#8b5cf6',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      animation: {
        'laser-sweep': 'laserSweep 4s linear infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-fast': 'pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-spin': 'radarSpin 5s linear infinite',
        'border-beam': 'borderBeam 6s linear infinite',
        'shimmer': 'shimmer 2.5s infinite linear',
        'data-stream': 'dataStream 15s linear infinite',
        'ping-slow': 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
      keyframes: {
        laserSweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(200%)' },
        },
        radarSpin: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        borderBeam: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        dataStream: {
          '0%': { strokeDashoffset: '1000' },
          '100%': { strokeDashoffset: '0' },
        },
      },
      boxShadow: {
        'neon-cyan': '0 0 20px -2px rgba(6, 182, 212, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.1)',
        'neon-emerald': '0 0 20px -2px rgba(16, 185, 129, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.1)',
        'neon-rose': '0 0 20px -2px rgba(244, 63, 94, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.1)',
        'neon-amber': '0 0 20px -2px rgba(245, 158, 11, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.1)',
        'hud-card': '0 8px 32px 0 rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.08)',
      }
    },
  },
  plugins: [],
}
