import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/chat": "http://127.0.0.1:8787",
      "/conversations": "http://127.0.0.1:8787",
      "/daily-summaries": "http://127.0.0.1:8787",
      "/healthz": "http://127.0.0.1:8787"
    }
  }
});
