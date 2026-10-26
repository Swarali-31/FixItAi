# FastAPI and Uvicorn HTTP 405 Method Not Allowed & Reload Errors
Category: Web Frameworks & API Routing
Technologies: Python, FastAPI, Uvicorn, REST, HTTP

When executing API requests against a FastAPI server running via Uvicorn, developers may encounter:
`HTTP 405 Method Not Allowed` or changes in code not reflecting dynamically.

## Root Cause
1. An endpoint was created using `@app.post(...)` or `@app.get(...)`, but the HTTP client requested a mismatched method or requested a route handled by a fallback wildcard path like `@app.get("/{path:path}")`.
2. Uvicorn was launched without the `--reload` flag, causing in-memory route tables to remain stale after code modifications.

## Resolution Steps
1. Stop the running Uvicorn process and start it with hot-reload enabled:
   `uvicorn api:app --reload --host 127.0.0.1 --port 8000`
2. Check request method headers (`Content-Type: application/json` and `POST` vs `GET`).
3. Ensure custom API routes are declared above any catch-all fallback or SPA static file mounts in the router script.
