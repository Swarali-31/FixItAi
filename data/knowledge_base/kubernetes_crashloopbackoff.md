# Kubernetes Pod CrashLoopBackOff Resolution
Category: Orchestration & Cloud Native
Technologies: Kubernetes, Docker, Linux, Go, Java

When inspecting Kubernetes pods via `kubectl get pods`:
A pod is reported with status `CrashLoopBackOff` and increasing restart counts.

## Root Cause
The container inside the pod started, encountered a fatal exception or missed a required environment variable/configuration, and terminated immediately (exit code 1 or 137). Kubernetes automatically restarted it, entering an exponential backoff loop.

## Resolution Steps
1. Inspect the exact application logs before crash:
   `kubectl logs <pod-name> --previous`
2. Describe pod events to check for OOM or failed probes:
   `kubectl describe pod <pod-name>`
3. Check the exit code:
   - `Exit Code 1`: General application error (missing dependency or runtime syntax crash).
   - `Exit Code 137`: Out-Of-Memory (OOMKilled) - increase memory limits.
   - `Exit Code 139`: Segmentation fault (native C/C++ library failure).
4. Verify environment variables and Secret/ConfigMap mounts:
   Ensure all database connection URLs and API keys exist.
