/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#ffffff",
        surface: "#0e0f15",
        "surface-raised": "#151722",
        "surface-border": "rgba(255, 255, 255, 0.08)",
        primary: {
          50: "#f0fdfa",
          100: "#ccfbf1",
          400: "#2dd4bf",
          500: "#06b6d4",
          600: "#0891b2",
        },
        spatial: {
          cyan: "#00f0ff",
          purple: "#9d4edd",
          glow: "rgba(0, 240, 255, 0.15)",
        }
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "'SF Pro Display'",
          "'Inter'",
          "system-ui",
          "sans-serif"
        ],
        mono: [
          "'SF Mono'",
          "'JetBrains Mono'",
          "'Fira Code'",
          "monospace"
        ]
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 12s linear infinite',
      }
    },
  },
  plugins: [],
}
