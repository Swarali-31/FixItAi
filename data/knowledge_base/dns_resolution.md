# Application cannot resolve a DNS hostname
Category: Networking
Technologies: DNS, networking, Linux, Windows, Docker

Name resolution failures mean the runtime cannot translate a hostname to an address. Test resolution from the same machine, container, or pod that runs the application using nslookup, dig, or the platform's resolver tools. Check spelling, search domains, DNS server settings, VPN connectivity, private-zone visibility, and container or cluster DNS configuration.

Compare the result with a known-good host and inspect resolver logs or DNS records. A temporary address may help isolate DNS from routing, but do not hardcode it as a lasting fix because service addresses can change. Restarting an application may clear a stale local cache, but verify the underlying record and network path first.
