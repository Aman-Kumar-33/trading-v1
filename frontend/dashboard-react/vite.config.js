import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Shipped artifact is this build's dist/ served by the Python stdlib server (scripts/dashboard.py /
// Dashboard.exe) - the proxy below is only for `npm run dev` fast iteration against that same backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
