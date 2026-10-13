# Docker port is already allocated
Category: Containers and networking
Technologies: Docker, networking, Linux, Windows

The Docker error "port is already allocated" means another process or container is bound to the requested host address and port. Inspect active containers with docker ps and check the process listening on that host port (for example, Get-NetTCPConnection on Windows or ss -ltnp on Linux). Stop the conflicting service only if it is safe to do so, or map the container to a different host port such as 8081:80. The left side of a publish mapping is the host port; the right side is the container port.

For Docker Compose, inspect all services for duplicate published ports and check stale containers. A container can listen internally on port 80 while the host maps 8081 to it. Confirm the app binds to 0.0.0.0 inside the container when external access is expected.
