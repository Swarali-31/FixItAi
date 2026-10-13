# Troubleshoot HTTP 500 Internal Server Error
Category: Web application troubleshooting
Technologies: HTTP, REST, Python, Java

HTTP 500 indicates the server encountered an unexpected condition. The response alone does not identify the cause. Correlate the request timestamp and request ID with application and reverse-proxy logs, then inspect the server-side exception and stack trace. Check recent deployments, dependency changes, database connectivity, environment configuration, permissions, and resource exhaustion.

Reproduce with a minimal request in a safe environment. Return a stable error response to clients and keep detailed traces in protected server logs; never return secrets or stack traces publicly. If the issue began after a release, compare configuration and roll back only through the normal deployment procedure.
