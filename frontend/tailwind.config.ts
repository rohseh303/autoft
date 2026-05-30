import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./node_modules/@tremor/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        tremor: {
          brand: {
            faint: "#0B1F3A",
            muted: "#172554",
            subtle: "#3b82f6",
            DEFAULT: "#3b82f6",
            emphasis: "#60a5fa",
            inverted: "#ffffff",
          },
        },
      },
    },
  },
  plugins: [],
  safelist: [
    { pattern: /^(bg|text|border|ring)-(blue|emerald|amber|rose|slate)-(50|100|200|300|400|500|600|700|800|900)$/ },
  ],
};

export default config;
