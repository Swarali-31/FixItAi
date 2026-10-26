# Vite React Proxy and CORS Configuration Errors
Category: Frontend Development & Build Tooling
Technologies: React, Vite, JavaScript, TypeScript, HTTP, CORS

When executing API calls from a React Vite development server, requests to `/api/...` may return `504 Gateway Timeout`, `404 Not Found`, or `CORS policy: No 'Access-Control-Allow-Origin' header is present`.

## Root Cause
The Vite dev server runs on port 5173 while the backend API runs on port 8000. Without a reverse proxy configured in `vite.config.js`, the browser blocks cross-origin fetch requests, or requests route to the Vite HTML fallback instead of the API backend.

## Resolution Steps
1. Open `vite.config.js` and configure the development proxy:
   ```javascript
   export default defineConfig({
     server: {
       proxy: {
         '/api': {
           target: 'http://127.0.0.1:8000',
           changeOrigin: true,
         }
       }
     }
   });
   ```
2. Verify that the backend service is running and listening on `127.0.0.1:8000`.
3. In frontend code, use relative fetch paths e.g. `fetch('/api/search')` instead of hardcoding absolute ports.
