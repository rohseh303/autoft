import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// Client lives in ./client, builds to ./dist. The Elysia BFF (server/index.ts)
// proxies all /api/* in dev and serves ./dist + /api in prod.
export default defineConfig({
  root: "client",
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./client", import.meta.url)),
      "@shared": fileURLToPath(new URL("./shared", import.meta.url)),
    },
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.BFF_URL ?? "http://localhost:8787",
        changeOrigin: true,
        // SSE needs the proxy to not buffer.
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => proxyReq.setHeader("connection", "keep-alive"));
        },
      },
    },
  },
});
