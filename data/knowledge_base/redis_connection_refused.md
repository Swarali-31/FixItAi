# Redis Connection Refused on Port 6379 (ECONNREFUSED)
Category: In-Memory Caching & Key-Value Stores
Technologies: Redis, Linux, Python, Node.js, Docker

When a client application attempts to establish a connection to Redis:
`Error: connect ECONNREFUSED 127.0.0.1:6379` or `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.`

## Root Cause
1. The `redis-server` daemon process is not running.
2. Redis is bound strictly to a private IP or Unix domain socket instead of loopback/all interfaces.
3. Protected-mode is enabled without a configured password, preventing remote access.

## Resolution Steps
1. Verify if Redis is running:
   `redis-cli ping` (should output `PONG`)
2. Start the service:
   `sudo systemctl start redis-server` or `sudo service redis start`
3. Edit `/etc/redis/redis.conf` to check binding:
   `bind 127.0.0.1 ::1`
4. If testing in Docker:
   `docker run -d --name my-redis -p 6379:6379 redis:alpine`
