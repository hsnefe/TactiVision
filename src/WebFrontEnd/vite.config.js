import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api istekleri backend'e (uvicorn :8000) yönlendirilir.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
