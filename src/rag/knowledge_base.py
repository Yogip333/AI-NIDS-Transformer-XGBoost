"""
Cybersecurity Threat Knowledge Base.

Curated entries for each attack type in CICIDS2017 with:
  - Description & characteristics
  - MITRE ATT&CK techniques
  - Zeek log indicators
  - Network IOCs
  - Recommended response actions
"""
from dataclasses import dataclass, field


@dataclass
class ThreatEntry:
    attack_id: int
    attack_name: str
    category: str
    description: str
    techniques: list[str]
    zeek_indicators: list[str]
    network_iocs: list[str]
    severity: str
    response_actions: list[str]
    mitre_tactics: list[str]
    tags: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Return rich-text representation for embedding."""
        return (
            f"ATTACK: {self.attack_name}\n"
            f"CATEGORY: {self.category}\n"
            f"SEVERITY: {self.severity}\n"
            f"DESCRIPTION: {self.description}\n"
            f"MITRE TACTICS: {', '.join(self.mitre_tactics)}\n"
            f"MITRE TECHNIQUES: {', '.join(self.techniques)}\n"
            f"ZEEK LOG INDICATORS: {'; '.join(self.zeek_indicators)}\n"
            f"NETWORK IOCS: {'; '.join(self.network_iocs)}\n"
            f"RESPONSE ACTIONS: {'; '.join(self.response_actions)}\n"
            f"TAGS: {', '.join(self.tags)}"
        )


THREAT_KNOWLEDGE_BASE: list[ThreatEntry] = [

    ThreatEntry(
        attack_id=0,
        attack_name="BENIGN",
        category="Normal Traffic",
        description=(
            "Normal, benign network traffic represents legitimate communications between "
            "hosts. Characteristics include consistent flow patterns, normal protocol "
            "distributions, and expected bandwidth usage for business operations."
        ),
        techniques=[],
        zeek_indicators=["Normal conn.log patterns", "Regular HTTP/DNS activity", "Expected port usage"],
        network_iocs=[],
        severity="NONE",
        response_actions=["No action required", "Continue baseline monitoring"],
        mitre_tactics=[],
        tags=["benign", "normal", "baseline"],
    ),

    ThreatEntry(
        attack_id=1,
        attack_name="Bot",
        category="Malware / Botnet",
        description=(
            "Botnet traffic indicates a compromised host under command-and-control (C2) "
            "communication. Bots typically exhibit periodic beacon patterns, encrypted C2 "
            "channels over common ports (80, 443), and may participate in DDoS campaigns. "
            "Zeek SSL and HTTP logs often reveal abnormal certificate patterns or domain "
            "generation algorithm (DGA) activity in dns.log."
        ),
        techniques=["T1071 - Application Layer Protocol", "T1095 - Non-Standard Port",
                    "T1571 - Non-Standard Port", "T1008 - Fallback Channels",
                    "T1102 - Web Service C2"],
        zeek_indicators=[
            "ssl.log: unusual certificate subjects or self-signed certs",
            "dns.log: high-entropy domain queries (DGA patterns)",
            "conn.log: periodic beaconing at fixed intervals (± 5 sec jitter)",
            "conn.log: long-duration low-volume connections to external IPs",
            "http.log: unusual User-Agent strings or base64 encoded URIs",
        ],
        network_iocs=[
            "Periodic outbound connections at fixed intervals",
            "Low-byte-count encrypted flows to non-standard destinations",
            "High volume of DNS queries to newly registered domains",
            "Connections to known botnet C2 IP ranges",
            "HTTP POST requests with base64 or gzip payloads to suspicious domains",
        ],
        severity="HIGH",
        response_actions=[
            "Immediately isolate the infected host from the network",
            "Block outbound traffic to identified C2 IPs/domains at the firewall",
            "Capture full packet data for forensic analysis",
            "Run antivirus and EDR scan on the compromised host",
            "Review DNS logs for DGA domain patterns",
            "Check for persistence mechanisms (startup registry, scheduled tasks)",
            "Report IOCs to threat intelligence platform",
        ],
        mitre_tactics=["Command and Control", "Exfiltration", "Impact"],
        tags=["botnet", "c2", "malware", "dga", "beacon", "persistence"],
    ),

    ThreatEntry(
        attack_id=2,
        attack_name="DDoS",
        category="Denial of Service",
        description=(
            "Distributed Denial of Service (DDoS) attacks flood target services with "
            "massive traffic volumes from multiple sources, exhausting resources. "
            "Common techniques include UDP floods, SYN floods, HTTP floods, and "
            "amplification attacks (DNS/NTP reflection). Zeek conn.log shows extreme "
            "packet rates and abnormally short flow durations with high byte counts."
        ),
        techniques=["T1498 - Network Denial of Service", "T1498.001 - Direct Network Flood",
                    "T1498.002 - Reflection Amplification"],
        zeek_indicators=[
            "conn.log: extreme flow volumes (>100K flows/min from diverse sources)",
            "conn.log: very short duration flows (<10ms) with SYN-only patterns",
            "conn.log: many RSTO or REJ states indicating failed handshakes",
            "dns.log: UDP DNS amplification (small queries, huge responses)",
            "conn.log: bandwidth spike >1 Gbps from multiple source IPs",
        ],
        network_iocs=[
            "Thousands of connections per second to a single target IP",
            "Traffic from multiple geographically dispersed sources",
            "Abnormally high SYN packet rates without corresponding ACKs",
            "UDP floods with spoofed source addresses",
            "DNS response sizes 50-70x larger than queries (amplification factor)",
        ],
        severity="CRITICAL",
        response_actions=[
            "Activate DDoS mitigation service (upstream scrubbing centre)",
            "Implement rate limiting and ingress filtering (RFC 2827)",
            "Enable BGP blackholing for attacking source prefixes",
            "Contact ISP for upstream traffic scrubbing",
            "Deploy CAPTCHA challenges for HTTP-based attacks",
            "Analyse traffic patterns to distinguish legitimate from attack traffic",
            "Document attack signature and update firewall ACLs",
        ],
        mitre_tactics=["Impact"],
        tags=["ddos", "flood", "dos", "availability", "amplification", "syn-flood"],
    ),

    ThreatEntry(
        attack_id=3,
        attack_name="DoS GoldenEye",
        category="Denial of Service",
        description=(
            "GoldenEye is an application-layer HTTP DoS tool that keeps HTTP connections "
            "open by continuously sending partial HTTP requests. It targets web servers "
            "by exhausting connection pool limits. Zeek http.log shows many incomplete "
            "requests and long-lived connections with minimal data transfer."
        ),
        techniques=["T1499 - Endpoint Denial of Service", "T1499.002 - Service Exhaustion Flood"],
        zeek_indicators=[
            "http.log: many slow HTTP connections with missing headers",
            "conn.log: long-duration TCP connections to port 80/443 with low byte counts",
            "http.log: high connection count to single web server IP",
            "conn.log: Source sending hundreds of connections to same destination port",
        ],
        network_iocs=[
            "Hundreds of slow HTTP connections from single source",
            "HTTP requests with unusual or randomized User-Agent headers",
            "Keep-Alive connections without completing request headers",
            "CPU/memory exhaustion on web server processes",
        ],
        severity="HIGH",
        response_actions=[
            "Implement HTTP request timeout limits",
            "Deploy WAF rules to detect and block GoldenEye patterns",
            "Enable connection rate limiting per source IP",
            "Configure web server to close idle connections faster",
            "Block source IPs at perimeter firewall",
        ],
        mitre_tactics=["Impact"],
        tags=["dos", "http", "goldeneye", "slow-http", "application-layer"],
    ),

    ThreatEntry(
        attack_id=4,
        attack_name="DoS Hulk",
        category="Denial of Service",
        description=(
            "HULK (HTTP Unbearable Load King) is a web server stress tool that generates "
            "unique HTTP GET requests with randomised headers and URLs to bypass caching. "
            "It creates high HTTP traffic volume. Zeek http.log shows massive volumes of "
            "GET requests with randomised URI parameters and headers."
        ),
        techniques=["T1499 - Endpoint Denial of Service", "T1499.002 - Service Exhaustion Flood"],
        zeek_indicators=[
            "http.log: massive GET request rate with randomised URIs",
            "http.log: unique User-Agent per request (evasion technique)",
            "conn.log: high packets/second to web server ports",
            "http.log: requests with randomised query parameters",
        ],
        network_iocs=[
            "Thousands of unique GET requests per minute",
            "Randomised URL query strings to defeat caching",
            "Random User-Agent strings per connection",
            "High bandwidth utilisation from single source",
        ],
        severity="HIGH",
        response_actions=[
            "Deploy WAF with rate limiting and behavioural analysis",
            "Enable bot detection / challenge pages",
            "Implement connection rate limiting at the load balancer",
            "Block source IPs with automated firewall rules",
            "Consider CDN or DDoS protection service",
        ],
        mitre_tactics=["Impact"],
        tags=["dos", "http", "hulk", "high-rate", "web-server"],
    ),

    ThreatEntry(
        attack_id=5,
        attack_name="DoS Slowhttptest",
        category="Denial of Service",
        description=(
            "Slowhttptest implements multiple slow HTTP DoS attack vectors: slow headers "
            "(Slowloris), slow body (Slow POST), and slow read attacks. These attacks "
            "hold connections open by sending data extremely slowly, exhausting the server "
            "connection pool. Zeek conn.log shows abnormally long-lived connections with "
            "very low throughput."
        ),
        techniques=["T1499 - Endpoint Denial of Service", "T1499.002 - Service Exhaustion Flood"],
        zeek_indicators=[
            "conn.log: long-duration connections (>60s) to port 80/443 with <1 KB data",
            "http.log: incomplete HTTP headers or body lingering for minutes",
            "conn.log: SF state connections with minimal bytes transferred",
            "http.log: very low bytes/sec on established connections",
        ],
        network_iocs=[
            "Connections open for minutes with minimal data",
            "HTTP headers sent at 1 byte per 10 seconds",
            "POST requests with very slow body delivery",
            "Server connection pool exhausted (503 errors in response)",
        ],
        severity="HIGH",
        response_actions=[
            "Set aggressive HTTP header timeout (10-20 seconds)",
            "Configure minimum data rate requirements for connections",
            "Deploy WAF with slow-HTTP detection rules",
            "Rate limit connections per source IP",
            "Implement maximum body size limits",
        ],
        mitre_tactics=["Impact"],
        tags=["dos", "slow-http", "slowloris", "connection-exhaustion"],
    ),

    ThreatEntry(
        attack_id=6,
        attack_name="DoS slowloris",
        category="Denial of Service",
        description=(
            "Slowloris holds HTTP connections open by sending partial HTTP requests, "
            "never completing them. By sending a header line every ~15 seconds, it keeps "
            "connections alive indefinitely while consuming server resources. A single "
            "attacker machine can take down most web servers. Zeek conn.log shows many "
            "long-lived low-bandwidth connections."
        ),
        techniques=["T1499 - Endpoint Denial of Service", "T1499.001 - OS Exhaustion Flood"],
        zeek_indicators=[
            "conn.log: hundreds of long-lived TCP connections to port 80/443",
            "http.log: connections with partial headers that never complete",
            "conn.log: very low bytes transferred over long durations (hours)",
            "conn.log: source IP holding >100 simultaneous connections",
        ],
        network_iocs=[
            "Single source holding hundreds of HTTP connections simultaneously",
            "HTTP headers sent periodically every 15 seconds",
            "Never-completing HTTP request sequences",
            "Server Apache/nginx process table exhaustion",
        ],
        severity="HIGH",
        response_actions=[
            "Install mod_reqtimeout (Apache) or equivalent timeout module",
            "Deploy Nginx with appropriate keepalive_timeout settings",
            "Rate limit connections per source IP",
            "Use a reverse proxy or CDN with connection limits",
            "Block source IP at firewall level",
        ],
        mitre_tactics=["Impact"],
        tags=["dos", "slowloris", "slow-http", "connection-exhaustion", "web"],
    ),

    ThreatEntry(
        attack_id=7,
        attack_name="FTP-Patator",
        category="Brute Force",
        description=(
            "FTP-Patator is an automated brute-force credential guessing attack against "
            "FTP servers (port 21). The attacker tries username/password combinations at "
            "high speed. Zeek conn.log and files.log show rapid successive short-duration "
            "connections to port 21, many ending in authentication failures."
        ),
        techniques=["T1110 - Brute Force", "T1110.001 - Password Guessing",
                    "T1110.003 - Password Spraying"],
        zeek_indicators=[
            "conn.log: many short-duration connections to port 21 from same source",
            "files.log: repeated authentication attempts visible in FTP control channel",
            "conn.log: high connection rate (>10/sec) from single source to port 21",
            "conn.log: mostly RSTO or S0 states indicating failed authentications",
        ],
        network_iocs=[
            "Hundreds of connection attempts per minute to port 21",
            "Sequential username/password combinations in authentication",
            "Multiple 530 Authentication Failed responses",
            "Same source IP with many short-lived FTP sessions",
        ],
        severity="HIGH",
        response_actions=[
            "Block source IP after 5 failed authentication attempts (fail2ban)",
            "Implement account lockout after failed attempts",
            "Enable FTP connection rate limiting",
            "Migrate from FTP to SFTP (port 22) or FTPS",
            "Enable MFA for FTP accounts",
            "Review FTP access logs for successful authentications from attacker IPs",
            "Consider disabling FTP entirely and use SFTP/SCP",
        ],
        mitre_tactics=["Credential Access", "Initial Access"],
        tags=["brute-force", "ftp", "credential", "patator", "port-21"],
    ),

    ThreatEntry(
        attack_id=8,
        attack_name="Heartbleed",
        category="Vulnerability Exploitation",
        description=(
            "Heartbleed (CVE-2014-0160) is a critical OpenSSL vulnerability that allows "
            "attackers to read server memory contents through malformed TLS heartbeat "
            "requests. Exploiting Heartbleed can leak private keys, passwords, and session "
            "tokens. Zeek ssl.log shows malformed TLS handshake patterns and abnormal "
            "certificate negotiation."
        ),
        techniques=["T1212 - Exploitation for Credential Access",
                    "T1190 - Exploit Public-Facing Application",
                    "T1587.003 - Develop Capabilities: Code Signing Certificates"],
        zeek_indicators=[
            "ssl.log: TLS heartbeat extension activity on OpenSSL servers",
            "conn.log: many short SSL connections to port 443",
            "ssl.log: connections to servers running vulnerable OpenSSL versions",
            "conn.log: resp_bytes significantly larger than orig_bytes (memory leak)",
        ],
        network_iocs=[
            "TLS heartbeat requests with malformed payload length",
            "Server responding with more data than requested",
            "Multiple rapid SSL connections testing heartbeat",
            "OpenSSL version 1.0.1 - 1.0.1f fingerprint",
        ],
        severity="CRITICAL",
        response_actions=[
            "IMMEDIATE: Patch OpenSSL to version 1.0.1g or later",
            "Revoke and reissue all SSL/TLS certificates on affected servers",
            "Force password resets for all users who may have had credentials exposed",
            "Invalidate all active session tokens",
            "Check for evidence of successful exploitation in server memory dumps",
            "Scan all servers for Heartbleed vulnerability",
            "Implement TLS 1.3 which removes heartbeat extension",
        ],
        mitre_tactics=["Credential Access", "Collection", "Initial Access"],
        tags=["heartbleed", "cve-2014-0160", "openssl", "tls", "memory-leak", "critical"],
    ),

    ThreatEntry(
        attack_id=9,
        attack_name="Infiltration",
        category="Advanced Persistent Threat",
        description=(
            "Infiltration attacks represent advanced threat actors gaining initial access "
            "and moving laterally within the network. These attacks use legitimate tools "
            "and protocols to blend with normal traffic (living off the land). Multi-stage "
            "attack chains may span DNS, HTTP, and SSL protocols. Zeek cross-protocol "
            "correlation is essential for detection."
        ),
        techniques=["T1566 - Phishing", "T1021 - Remote Services",
                    "T1041 - Exfiltration Over C2 Channel",
                    "T1055 - Process Injection", "T1027 - Obfuscated Files or Information",
                    "T1003 - OS Credential Dumping"],
        zeek_indicators=[
            "Multiple Zeek log types showing correlated activity within session window",
            "dns.log: queries for internal resources from unusual source",
            "ssl.log: encrypted channels to external IPs during off-hours",
            "conn.log: lateral movement patterns (scanning internal IP ranges)",
            "http.log: download of executable content (MIME type mismatches)",
            "files.log: suspicious file downloads or uploads",
        ],
        network_iocs=[
            "Internal host communicating with external C2 on unusual ports",
            "Lateral movement: sequential connections to internal hosts",
            "Data staging: large encrypted transfers to external destinations",
            "Living-off-the-land: WMI, PSExec, or RDP traffic from unexpected sources",
            "DNS exfiltration: encoded data in DNS query names",
        ],
        severity="CRITICAL",
        response_actions=[
            "Initiate incident response procedures immediately",
            "Isolate affected hosts and segment the network",
            "Preserve forensic evidence: memory dumps and network captures",
            "Engage threat hunting team for lateral movement analysis",
            "Review all authentication events for the past 30 days",
            "Analyse cross-protocol session data for attack chain reconstruction",
            "Notify management and security leadership",
            "Consider engaging external incident response firm",
        ],
        mitre_tactics=["Initial Access", "Execution", "Lateral Movement", "Exfiltration"],
        tags=["apt", "infiltration", "lateral-movement", "advanced", "multi-stage"],
    ),

    ThreatEntry(
        attack_id=10,
        attack_name="PortScan",
        category="Reconnaissance",
        description=(
            "Port scanning is a reconnaissance technique used to discover open services "
            "on target hosts. Attackers use tools like Nmap to systematically probe ports. "
            "Zeek conn.log shows massive numbers of short-duration connections to many "
            "different destination ports from a single source, most resulting in RST or "
            "S0 (no response) states."
        ),
        techniques=["T1046 - Network Service Discovery", "T1595 - Active Scanning",
                    "T1595.001 - Scanning IP Blocks"],
        zeek_indicators=[
            "conn.log: single source IP connecting to >100 unique destination ports",
            "conn.log: many S0 or REJ states (probed but no service)",
            "conn.log: very short durations (<1ms) for most flows",
            "conn.log: sequential port numbers indicating systematic scan",
            "conn.log: SYN-only packets without completing handshake (stealth scan)",
        ],
        network_iocs=[
            "Single source scanning sequential port ranges",
            "RST/ACK responses from closed ports in rapid succession",
            "Nmap OS fingerprinting probes",
            "UDP probes to common UDP service ports",
            "ICMP probes mixed with TCP SYN probes",
        ],
        severity="MEDIUM",
        response_actions=[
            "Block source IP at perimeter firewall immediately",
            "Investigate what triggered the scan (may be follow-on attack)",
            "Review which ports responded as open",
            "Close unnecessary open ports and services",
            "Enable IDS alerts for port scan patterns",
            "Log the event in threat intelligence platform",
            "Check if scan originated from inside network (compromised host)",
        ],
        mitre_tactics=["Discovery", "Reconnaissance"],
        tags=["portscan", "nmap", "reconnaissance", "discovery", "scanning"],
    ),

    ThreatEntry(
        attack_id=11,
        attack_name="SSH-Patator",
        category="Brute Force",
        description=(
            "SSH-Patator is an automated brute-force credential guessing attack against "
            "SSH servers (port 22). Rapid authentication attempts are made with different "
            "username/password or key combinations. Zeek conn.log shows many short-lived "
            "SSH connections from the same source, with most failing authentication."
        ),
        techniques=["T1110 - Brute Force", "T1110.001 - Password Guessing",
                    "T1110.004 - Credential Stuffing"],
        zeek_indicators=[
            "conn.log: high rate of short connections to port 22 from same source",
            "ssh.log: many authentication failure events",
            "conn.log: connection durations consistent with failed auth (~2-3 seconds)",
            "conn.log: dozens of connections per minute from single source to port 22",
        ],
        network_iocs=[
            "More than 10 SSH authentication attempts per minute from same source",
            "Sequential username lists in authentication headers",
            "SSH banner exchange followed immediately by connection close",
            "Multiple 'Authentication failed' syslog entries",
        ],
        severity="HIGH",
        response_actions=[
            "Block source IP immediately using fail2ban or firewall rule",
            "Implement SSH key-based authentication and disable password auth",
            "Change SSH to non-standard port (security through obscurity + firewall rules)",
            "Enable MFA for SSH access",
            "Review successful SSH logins from attacker IPs",
            "Check for privilege escalation after any successful authentication",
            "Consider using a bastion host / jump server for SSH access",
        ],
        mitre_tactics=["Credential Access", "Initial Access"],
        tags=["brute-force", "ssh", "credential", "patator", "port-22"],
    ),

    ThreatEntry(
        attack_id=12,
        attack_name="Web Attack Brute Force",
        category="Web Attack",
        description=(
            "Web-based brute force attacks target login forms or API endpoints, trying "
            "many username/password combinations. The attacker uses automated tools to "
            "submit authentication requests at high speed. Zeek http.log shows many POST "
            "requests to login endpoints with small request bodies and 401/403 responses."
        ),
        techniques=["T1110 - Brute Force", "T1110.003 - Password Spraying",
                    "T1190 - Exploit Public-Facing Application"],
        zeek_indicators=[
            "http.log: repeated POST requests to /login or authentication endpoints",
            "http.log: HTTP 401 or 403 response codes at high rate",
            "http.log: same source IP sending identical requests with varied credentials",
            "conn.log: many short connections to port 80/443 with small payload sizes",
        ],
        network_iocs=[
            "Hundreds of POST requests to login endpoint per minute",
            "High proportion of 401/403 HTTP responses",
            "Same User-Agent across all requests",
            "Rapid succession of authentication attempts without delay",
        ],
        severity="HIGH",
        response_actions=[
            "Implement CAPTCHA on login forms",
            "Enable account lockout after 5 failed attempts",
            "Rate limit authentication attempts per IP",
            "Implement MFA for all user accounts",
            "Deploy WAF with brute force detection rules",
            "Alert on accounts with >5 failed logins in 1 minute",
            "Review server access logs for successful authentications post-attack",
        ],
        mitre_tactics=["Credential Access", "Initial Access"],
        tags=["web-attack", "brute-force", "http", "login", "credential"],
    ),

    ThreatEntry(
        attack_id=13,
        attack_name="Web Attack Sql Injection",
        category="Web Attack",
        description=(
            "SQL Injection attacks manipulate database queries by injecting malicious SQL "
            "code into web application inputs. Attackers can extract data, bypass "
            "authentication, or execute OS commands. Zeek http.log shows requests with "
            "SQL keywords (SELECT, UNION, OR 1=1) in URL parameters or POST bodies."
        ),
        techniques=["T1190 - Exploit Public-Facing Application",
                    "T1555 - Credentials from Password Stores",
                    "T1005 - Data from Local System"],
        zeek_indicators=[
            "http.log: URIs containing SQL keywords (UNION, SELECT, DROP, INSERT)",
            "http.log: requests with special characters (', --, ;, /*)",
            "http.log: error responses (500) after malformed query submissions",
            "http.log: unusually large response bodies (data extraction)",
        ],
        network_iocs=[
            "URL parameters containing SQL metacharacters",
            "UNION SELECT statements in HTTP requests",
            "Boolean-based injection: OR 1=1, AND 1=2",
            "Time-based injection: SLEEP(), WAITFOR DELAY patterns",
            "Stacked queries using semicolons",
        ],
        severity="CRITICAL",
        response_actions=[
            "Immediately patch the vulnerable application input validation",
            "Implement parameterised queries / prepared statements",
            "Deploy WAF with SQL injection detection signatures",
            "Audit database access logs for data exfiltration",
            "Check for exfiltrated sensitive data (credentials, PII)",
            "Review and sanitise all user input fields",
            "Implement least-privilege database accounts",
            "Notify compliance officer if PII/financial data was exposed (GDPR/UK DPA)",
        ],
        mitre_tactics=["Initial Access", "Collection", "Credential Access"],
        tags=["sqli", "web-attack", "injection", "database", "owasp-a1"],
    ),

    ThreatEntry(
        attack_id=14,
        attack_name="Web Attack XSS",
        category="Web Attack",
        description=(
            "Cross-Site Scripting (XSS) attacks inject malicious scripts into web pages "
            "viewed by other users. Attackers can steal session cookies, redirect users, "
            "or perform actions on behalf of victims. Zeek http.log shows requests with "
            "HTML/JavaScript patterns in parameters, such as <script> tags or onerror "
            "handlers."
        ),
        techniques=["T1059.007 - Command and Scripting Interpreter: JavaScript",
                    "T1185 - Browser Session Hijacking",
                    "T1534 - Internal Spearphishing"],
        zeek_indicators=[
            "http.log: requests with <script>, javascript:, onerror= in parameters",
            "http.log: HTML entity encoding patterns (&#x3C;script&#x3E;)",
            "http.log: redirect parameters containing javascript: protocol",
            "http.log: event handlers like onclick=, onload= in URL parameters",
        ],
        network_iocs=[
            "HTTP requests containing <script> tags in URL or POST body",
            "JavaScript event handlers in URL parameters",
            "Base64 encoded payloads in web parameters",
            "Document.cookie access patterns in web traffic",
            "Unusual redirect patterns to external domains",
        ],
        severity="HIGH",
        response_actions=[
            "Implement Content Security Policy (CSP) headers",
            "Enable HTTPOnly and Secure flags on session cookies",
            "Deploy WAF with XSS detection rules (OWASP CRS)",
            "Sanitise and encode all user-supplied input",
            "Implement X-XSS-Protection and X-Frame-Options headers",
            "Audit all input fields for reflected XSS vulnerabilities",
            "Check server logs for evidence of successful XSS exploitation",
            "Consider penetration testing for comprehensive XSS coverage",
        ],
        mitre_tactics=["Collection", "Credential Access", "Execution"],
        tags=["xss", "web-attack", "javascript", "injection", "owasp-a7"],
    ),
]


def get_threat_by_id(attack_id: int) -> ThreatEntry | None:
    for entry in THREAT_KNOWLEDGE_BASE:
        if entry.attack_id == attack_id:
            return entry
    return None


def get_all_texts() -> list[tuple[str, dict]]:
    """Return list of (text, metadata) tuples for ChromaDB ingestion."""
    results = []
    for entry in THREAT_KNOWLEDGE_BASE:
        meta = {
            "attack_id":   entry.attack_id,
            "attack_name": entry.attack_name,
            "category":    entry.category,
            "severity":    entry.severity,
        }
        results.append((entry.to_text(), meta))
    return results
