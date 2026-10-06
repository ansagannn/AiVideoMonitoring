/** @type {import('tailwindcss').Config} */
import animate from 'tailwindcss-animate'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Aegis AI palette (Stitch AI styling)
        background: '#0B0F19',
        'surface-base': '#0F131D',
        'surface-container-lowest': '#0a0e18',
        'surface-container-low': '#171b26',
        'surface-container': '#1c1f2a',
        'surface-container-high': '#262a35',
        'surface-container-highest': '#313540',
        'surface-bright': '#353944',
        'surface-variant': '#313540',
        primary: '#c3f5ff',
        'primary-container': '#00e5ff',
        'on-primary-container': '#00626e',
        'on-primary': '#00363d',
        'on-surface': '#dfe2f1',
        'on-surface-variant': '#bac9cc',
        'outline-variant': '#3b494c',
        'border-subtle': '#30363D',
        'text-primary': '#FFFFFF',
        'text-secondary': '#94A3B8',
        'status-success': '#10B981',
        'status-warning': '#F59E0B',
        'status-danger': '#EF4444',
      },
      borderRadius: {
        DEFAULT: '0.5rem',
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
        xl: '0.75rem',
        full: '9999px',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular'],
      },
      fontSize: {
        'label-caps': ['12px', { lineHeight: '1', letterSpacing: '0.05em', fontWeight: '700' }],
        'data-mono': ['14px', { lineHeight: '1', letterSpacing: '-0.01em', fontWeight: '500' }],
        'headline-md': ['24px', { lineHeight: '1.4', fontWeight: '600' }],
        'headline-lg': ['32px', { lineHeight: '1.2', fontWeight: '600' }],
      },
      boxShadow: {
        cyan: '0 0 12px rgba(0, 229, 255, 0.3)',
        red: '0 0 12px rgba(239, 68, 68, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [animate],
}
