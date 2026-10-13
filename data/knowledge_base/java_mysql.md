# Java JDBC communications link failure
Category: Database connectivity
Technologies: Java, MySQL, SQL, networking

When a Java application reports Communications link failure while opening a MySQL connection, first check whether the database hostname resolves from the application host and whether the configured TCP port is reachable. A local database may use 127.0.0.1; inside a container, localhost refers to that container, so use the database service name or reachable host address.

Confirm that MySQL is running and listening on the expected port (commonly 3306). Check firewall rules, cloud security groups, VPN routes, and container network membership. Verify JDBC URL host, port, database name, TLS options, and driver compatibility. A connection timeout usually indicates reachability or listener problems; authentication failures normally return a distinct access-denied error. Do not expose credentials in logs. Retry with bounded exponential backoff only for transient connection failures.

Troubleshooting: test DNS resolution from the same runtime; test TCP connectivity; inspect MySQL server logs; verify bind-address and port; validate JDBC URL and Connector/J version; then retry with a short connection timeout.
