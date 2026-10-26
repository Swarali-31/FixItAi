# PostgreSQL Connection Refused on Port 5432
Category: Database Connectivity & Configuration
Technologies: PostgreSQL, SQL, Linux, Docker, Python

When connecting to a PostgreSQL instance, clients may encounter:
`psycopg2.OperationalError: could not connect to server: Connection refused`
`Is the server running on host "localhost" (127.0.0.1) and accepting TCP/IP connections on port 5432?`

## Root Cause
The PostgreSQL server daemon (`postgres` / `postgresql`) is either stopped, bound only to Unix domain sockets, or configured with `listen_addresses = 'localhost'` when the client is trying to reach it across a network interface or container bridge.

## Resolution Steps
1. Verify if the PostgreSQL daemon is active:
   `sudo systemctl status postgresql` or `pg_isready -h localhost -p 5432`
2. Start or restart the service if stopped:
   `sudo systemctl restart postgresql`
3. Edit `postgresql.conf` to allow TCP connections:
   `listen_addresses = '*'` or `listen_addresses = 'localhost,127.0.0.1'`
4. Edit `pg_hba.conf` to allow authentication from your client subnet:
   `host all all 127.0.0.1/32 scram-sha-256`
5. If running inside a container, ensure port mapping is explicitly defined:
   `-p 5432:5432`
