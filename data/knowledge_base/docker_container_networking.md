# Docker Container Connection Refused & Localhost Network Resolution
Category: Containerization & Virtualization
Technologies: Docker, Linux, Networking, HTTP

When attempting to reach host services or another container from within a Docker container, connection requests often fail with:
`Connection refused` or `Host is unreachable (ECONNREFUSED)`.

## Root Cause
Inside a Docker container, `localhost` and `127.0.0.1` refer strictly to the container's own internal loopback interface, not the host machine or other containers.

## Resolution Steps
1. On macOS or Windows Docker Desktop, replace `localhost` with the special DNS hostname:
   `host.docker.internal`
2. In Linux environments, add the host network mapping flag to `docker run`:
   `--add-host=host.docker.internal:host-gateway` or run with `--network host`.
3. If connecting between multiple Docker containers, place both on a user-defined bridge network:
   `docker network create my-app-net`
   `docker run --network my-app-net --name backend my-backend`
   `docker run --network my-app-net --name frontend my-frontend`
   Then reach the backend using its container name: `http://backend:8000`.
