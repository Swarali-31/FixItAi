# MySQL connection timeout
Category: Database connectivity
Technologies: MySQL, SQL, networking

A MySQL connection timeout usually means the client could not establish a TCP connection before its deadline. Confirm the service is available, the host and port are correct, and the client can reach the server across firewalls, security groups, VPNs, and container networks. Check whether MySQL is listening on the expected interface and whether connection limits are exhausted.

Distinguish a connection timeout from a query timeout: the former happens before a session is established, while the latter occurs during a query. Inspect server availability and logs, measure network latency, then set a bounded connection timeout appropriate to the environment. Use connection pooling and bounded retries for transient failures; avoid infinite retries that amplify an outage.
