import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// The FastAPI server runs on :8000. During development Vite proxies API calls
// to it, so no CORS configuration is needed in the browser.
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": "http://127.0.0.1:8000",
        },
    },
    build: {
        outDir: "../issue_planner/web/dist",
        emptyOutDir: true,
    },
});
