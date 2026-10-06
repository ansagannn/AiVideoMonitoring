import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Backend FastAPI dev URL (overridable via VITE_BACKEND_URL).
// We proxy /api so the frontend can call relative URLs in dev without CORS or
// .env tweaks.
const BACKEND_URL = process.env.VITE_BACKEND_URL || "http://localhost:8000"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
})

