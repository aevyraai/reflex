"""Generate 100 synthetic security incident reports with 3-sentence executive brief summaries."""

import json
import csv
import random

random.seed(42)

# ── building blocks ──────────────────────────────────────────────────────────

COMPANIES = [
    ("Meridian Financial Group", "financial services", "Chicago"),
    ("Apex Healthcare Systems", "healthcare", "Houston"),
    ("NovaTech Manufacturing", "manufacturing", "Detroit"),
    ("Crestline Retail Corp", "retail", "Atlanta"),
    ("Pinnacle Legal Partners", "legal services", "New York"),
    ("Starfield Logistics", "logistics", "Memphis"),
    ("Clearwater Energy", "energy", "Dallas"),
    ("Summit Education Group", "education", "Boston"),
    ("HarborView Hotels", "hospitality", "Miami"),
    ("Redwood Insurance", "insurance", "San Francisco"),
    ("Cascade Software Inc", "software development", "Seattle"),
    ("Ironbridge Construction", "construction", "Phoenix"),
    ("BlueSky Media", "media and publishing", "Los Angeles"),
    ("Northgate Pharma", "pharmaceuticals", "Raleigh"),
    ("Vantage Point Capital", "investment management", "New York"),
    ("Orion Government Solutions", "government contracting", "Washington DC"),
    ("Pacific Coast Telecom", "telecommunications", "Portland"),
    ("Riverside Medical Center", "healthcare", "Cleveland"),
    ("Granite State Credit Union", "banking", "Manchester"),
    ("SilverLine Transportation", "transportation", "Denver"),
]

INCIDENT_TYPES = [
    "ransomware",
    "phishing",
    "data_breach_misconfig",
    "data_breach_sql",
    "insider_threat",
    "ddos",
    "bec",  # business email compromise
    "supply_chain",
    "credential_stuffing",
    "cryptojacking",
    "zero_day",
    "rdp_brute_force",
    "vishing",
    "malware_dropper",
    "api_key_leak",
]

SEVERITIES = ["Critical", "High", "Medium", "Low"]
SEVERITY_WEIGHTS = [0.2, 0.35, 0.3, 0.15]

DURATIONS = {
    "Critical": ["18 hours", "22 hours", "31 hours", "3 days", "5 days", "8 days"],
    "High":     ["4 hours", "7 hours", "11 hours", "16 hours", "2 days"],
    "Medium":   ["45 minutes", "90 minutes", "3 hours", "6 hours", "10 hours"],
    "Low":      ["15 minutes", "30 minutes", "1 hour", "2 hours"],
}

AFFECTED_COUNTS = {
    "Critical": [500, 800, 1200, 2000, 3500, 5000],
    "High":     [50, 100, 200, 400],
    "Medium":   [5, 10, 20, 40],
    "Low":      [1, 2, 3, 5],
}

FINANCIAL_IMPACTS = {
    "Critical": ["$280K", "$450K", "$620K", "$1.1M", "$2.3M", "$4.8M"],
    "High":     ["$35K", "$72K", "$120K", "$190K"],
    "Medium":   ["$8K", "$15K", "$24K"],
    "Low":      ["$1K", "$3K", "$5K"],
}


def pick(lst): return random.choice(lst)
def pickw(lst, weights): return random.choices(lst, weights=weights, k=1)[0]


# ── per-incident-type templates ───────────────────────────────────────────────

def make_ransomware(co, sev, dur, affected, impact):
    vectors = [
        "a phishing email that delivered a malicious macro-enabled attachment",
        "an exposed RDP port that was brute-forced by the attacker",
        "a compromised third-party VPN credential purchased on a darknet forum",
        "a drive-by download on a compromised supplier website",
    ]
    payloads = ["LockBit 3.0", "BlackCat/ALPHV", "Cl0p", "Royal", "Akira", "Play"]
    systems = [
        "file servers and backup infrastructure",
        "ERP system and shared network drives",
        "domain controllers and all Windows endpoints",
        "finance department workstations and document management system",
    ]
    vector = pick(vectors)
    payload = pick(payloads)
    system = pick(systems)
    remediation = pick([
        "Affected systems were isolated, backups restored from air-gapped copies, and a forced credential reset was completed org-wide.",
        "The environment was rebuilt from clean backups, endpoint detection rules updated, and MFA enforced on all remote access.",
        "Incident response retainer was engaged; encrypted systems were restored from offline backups taken 24 hours prior.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Date Detected: {pick(['2024-01-','2024-02-','2024-03-','2024-04-','2024-05-','2024-06-'])}{random.randint(1,28):02d}
Classification: {sev} — Ransomware
Reporting Entity: {co[0]} IT Security Team

EXECUTIVE SUMMARY
On the morning of the incident, the {co[0]} security operations center detected unusual file-rename activity across multiple servers. Investigation confirmed deployment of {payload} ransomware following {vector}. Encryption spread across {system} before containment was achieved after approximately {dur}.

TIMELINE OF EVENTS
- T+0:00  Attacker gains initial foothold via {vector}
- T+0:45  Lateral movement detected; attacker enumerates internal shares
- T+2:10  Ransomware payload deployed; encryption begins on primary file server
- T+2:35  SOC alert triggers; IR process initiated
- T+3:15  Affected segments isolated from network
- T+{dur}  Systems restored; operations resumed

SCOPE AND IMPACT
Approximately {affected} employees and contractors were unable to access business-critical systems during the outage. Customer-facing services experienced degraded availability for the first {pick(['4','6','8','12'])} hours. Estimated business impact is {impact} in lost productivity and incident response costs. {'Sensitive customer records were not accessed based on forensic review.' if sev in ['Medium','Low'] else 'Forensic review is ongoing to determine whether data exfiltration occurred prior to encryption.'}

TECHNICAL DETAILS
Initial Access: {vector.capitalize()}
Malware Family: {payload}
Affected Systems: {system.capitalize()}
Persistence Mechanism: Scheduled task planted in SYSTEM context
Lateral Movement: Pass-the-hash using harvested NTLM hashes

REMEDIATION ACTIONS TAKEN
{remediation}
Network segmentation has been reviewed and additional east-west firewall rules applied. Email gateway rules updated to block {pick(['.xlsm','.docm','.js','.hta'])} attachments. Tabletop exercise scheduled for next quarter.

RECOMMENDATIONS
1. Implement privileged access workstations for administrative staff
2. Migrate backup infrastructure to immutable object storage
3. Enforce phishing-resistant MFA on all remote access methods
4. Conduct quarterly ransomware simulation exercises
"""
    sentence1 = f"A {payload} ransomware attack encrypted {system} at {co[0]} after the attacker gained initial access through {vector}."
    sentence2 = f"The {dur} outage affected approximately {affected} employees, with an estimated business impact of {impact} in lost productivity and recovery costs."
    sentence3 = f"{remediation}"
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_phishing(co, sev, dur, affected, impact):
    lures = [
        "a spoofed IT helpdesk notification requesting credential verification",
        "a fake DocuSign link embedded in a vendor invoice email",
        "a forged Microsoft 365 re-authentication prompt",
        "a convincing payroll portal phishing page",
    ]
    outcomes = [
        "captured credentials for 12 employee accounts, three of which had access to the HR system",
        "harvested credentials for the CFO and two finance team members",
        "obtained VPN credentials for four engineers with production access",
        "compromised the CEO's email account and used it to send internal spearphish",
    ]
    lure = pick(lures)
    outcome = pick(outcomes)
    remediation = pick([
        "Compromised accounts were suspended, passwords reset, and MFA retroactively enforced across the affected department.",
        "All harvested sessions were invalidated, email forwarding rules audited and removed, and security awareness training deployed.",
        "Affected accounts were locked, a forensic review of sent mail confirmed no data exfiltration, and conditional access policies tightened.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Credential Phishing
Reporting Entity: {co[0]} Information Security

SUMMARY
{co[0]}'s email security gateway flagged a wave of {pick(['47','83','112','204','31'])} inbound phishing emails on {pick(['Monday','Tuesday','Wednesday','Thursday','Friday'])} morning. The campaign used {lure}. Despite gateway filtering, {pick(['3','5','8','11'])} emails reached end-user inboxes and the attacker successfully {outcome}.

TIMELINE
- T+0:00  Phishing emails delivered to staff inboxes
- T+0:22  First credential entered on attacker-controlled page
- T+1:05  Attacker logs into corporate systems using harvested credentials
- T+{pick(['2:30','3:10','4:45'])}  Anomalous login detected by SIEM; IR initiated
- T+{dur}  All compromised sessions terminated; investigation complete

IMPACT
{affected} user accounts were compromised. {'No sensitive data was exfiltrated.' if sev in ['Low','Medium'] else 'The attacker accessed the HR system and downloaded a partial employee roster before being detected.'} Estimated cost of incident response and lost productivity is {impact}.

TECHNICAL INDICATORS
Phishing Domain: {pick(['secure-login','corp-auth','docusign-verify','helpdesk-reset'])}-{random.randint(100,999)}.{pick(['com','net','org'])}
Hosting: {pick(['DigitalOcean','AWS','Cloudflare Workers','Hostinger'])} (takedown requested)
TTPs: T1566.002 (Spearphishing Link), T1078 (Valid Accounts)

REMEDIATION
{remediation}
Phishing simulation campaign scheduled for next month. Email banner warnings added for external senders impersonating internal domains.
"""
    sentence1 = f"A targeted phishing campaign using {lure} at {co[0]} successfully {outcome}."
    sentence2 = f"The incident lasted {dur} and affected {affected} employees, resulting in an estimated {impact} in response costs and lost productivity."
    sentence3 = f"{remediation}"
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_data_breach_misconfig(co, sev, dur, affected, impact):
    stores = [
        ("an S3 bucket containing customer PII", "customer records", "AWS S3"),
        ("a publicly exposed Elasticsearch index with transaction logs", "financial transaction records", "Elasticsearch"),
        ("an Azure Blob storage container with HR documents", "employee records and payroll data", "Azure Blob"),
        ("an unsecured MongoDB instance with application data", "user account data", "MongoDB"),
    ]
    store_desc, data_type, tech = pick(stores)
    discoverer = pick([
        "a security researcher who disclosed responsibly via email",
        "an external threat intelligence vendor monitoring for exposed credentials",
        "the company's own cloud security posture management tool",
        "a bug bounty submission",
    ])
    root_cause_misconfig = pick([
        "A recent infrastructure migration changed the default ACL settings without a post-migration security review.",
        "A developer applied a permissive bucket policy during testing and did not revert it before promotion to production.",
        "A Terraform configuration change removed the private access flag during a routine update.",
        "The storage account was created by a contractor who was unaware of the organisation's access control standards.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Data Exposure (Misconfiguration)
Reporting Entity: {co[0]} Cloud Security Team

INCIDENT OVERVIEW
{co[0]} was notified of an exposed {tech} instance by {discoverer}. {store_desc.capitalize()} had been publicly accessible without authentication for an estimated {dur}. The exposed data included {data_type} for approximately {affected} individuals.

TIMELINE
- T-{dur}  Misconfiguration introduced during infrastructure migration
- T+0:00  Exposure identified and reported
- T+0:35  Cloud team notified; access policy corrected
- T+1:10  Full audit of exposed data completed
- T+4:00  Affected individuals notified per breach notification requirements

SCOPE OF EXPOSURE
Data type exposed: {data_type.capitalize()}
Records affected: approximately {affected}
Duration of exposure: {dur}
Evidence of unauthorized access: {'None identified in access logs' if sev in ['Low','Medium'] else 'Download activity from 3 non-company IP addresses detected in access logs'}

ROOT CAUSE
{root_cause_misconfig}

REMEDIATION
Access was restricted within {pick(['35 minutes','1 hour','2 hours'])} of notification. A full audit of cloud storage permissions is underway. {pick(['Regulatory notification filed with relevant authority.', 'Legal team assessing breach notification obligations.', 'Affected individuals notified via email.'])}
"""
    sentence1 = f"A misconfigured {tech} instance at {co[0]} exposed {data_type} for approximately {affected} individuals for {dur} before being discovered by {discoverer}."
    sentence2 = f"{'Evidence of unauthorised access was detected in access logs, raising the risk of data misuse.' if sev in ['Critical','High'] else 'No evidence of unauthorised access was found in access logs, limiting impact.'} Estimated remediation and notification costs total {impact}."
    sentence3 = f"Access was restricted within hours of notification; a full cloud storage permission audit was initiated and affected individuals were notified per breach notification requirements."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_ddos(co, sev, dur, affected, impact):
    vectors = [
        "a volumetric UDP flood peaking at 340 Gbps",
        "an application-layer HTTP flood targeting the checkout API",
        "a DNS amplification attack generating 120 Gbps of traffic",
        "a SYN flood combined with slow-rate HTTP attacks",
    ]
    services = [
        "the customer-facing web portal and mobile application",
        "online banking and customer account portals",
        "the e-commerce checkout and payment processing endpoints",
        "public API endpoints consumed by partner integrations",
    ]
    vector = pick(vectors)
    service = pick(services)
    mitigation = pick([
        "Traffic was scrubbed via the CDN provider's DDoS mitigation service; origin IP addresses were rotated to prevent re-targeting.",
        "Upstream filtering was engaged through the ISP; rate limiting rules were applied at the WAF layer to absorb residual traffic.",
        "Anycast routing pushed attack traffic to scrubbing centres; application-level rate limits were tightened to 50 requests per second per IP.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Distributed Denial of Service
Reporting Entity: {co[0]} Network Operations Center

INCIDENT SUMMARY
On {pick(['Monday','Tuesday','Wednesday','Thursday'])} at {random.randint(8,17):02d}:{random.randint(0,59):02d}, {co[0]}'s NOC detected a sharp spike in inbound traffic consistent with {vector}. The attack targeted {service}, causing complete unavailability for {pick(['12','18','23','35'])} minutes before partial mitigation was achieved. Full service restoration took {dur}.

ATTACK CHARACTERISTICS
Attack Vector: {vector.capitalize()}
Peak Throughput: {pick(['28 Gbps','85 Gbps','210 Gbps','340 Gbps','520 Gbps'])}
Origin: Distributed botnet across {pick(['14','22','37','61'])} countries
Targeted Endpoint(s): {service.capitalize()}

BUSINESS IMPACT
{service.capitalize()} was degraded or unavailable for {dur}. Estimated {affected} customers experienced errors or timeouts. {pick(['Payment processing was unaffected as it runs on a separate infrastructure path.','All in-flight transactions were rolled back safely.','Incomplete transactions during the outage window have been identified and customers contacted.'])} Revenue and reputational impact estimated at {impact}.

MITIGATION ACTIONS
{mitigation}
Post-incident: DDoS protection thresholds reviewed and increased. BGP blackholing procedure documented and tested. Runbook updated with ISP escalation contacts.

RECOMMENDATIONS
1. Implement always-on DDoS protection rather than on-demand activation
2. Conduct quarterly DDoS simulation to validate mitigation runbooks
3. Evaluate secondary CDN provider for failover capacity
"""
    sentence1 = f"A DDoS attack using {vector} disrupted {service} at {co[0]}, causing degraded availability for {dur}."
    sentence2 = f"Approximately {affected} customers were affected, with an estimated business impact of {impact} in lost revenue and recovery costs."
    sentence3 = f"{mitigation}"
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_bec(co, sev, dur, affected, impact):
    methods = [
        "a compromised vendor email account used to send a fraudulent wire transfer request",
        "a lookalike domain impersonating the CFO to redirect a scheduled supplier payment",
        "a hijacked executive email account used to authorise an urgent payroll change",
        "a spoofed CEO email requesting an emergency fund transfer to a new account",
    ]
    method = pick(methods)
    outcome = pick([
        f"A wire transfer of {impact} was initiated before the fraud was identified",
        f"A payroll redirect of {impact} was processed to an attacker-controlled account",
        f"An accounts payable team member authorised a supplier payment of {impact} to a fraudulent account",
    ])
    recovery = pick([
        "The bank was contacted within the recall window and funds were successfully recovered.",
        "A SWIFT recall was initiated; partial fund recovery is pending.",
        "Funds were not recoverable; an insurance claim has been filed.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Business Email Compromise
Reporting Entity: {co[0]} Finance and Security Teams

INCIDENT DESCRIPTION
{co[0]}'s finance team received a request via {method}. {outcome}. The fraud was identified {dur} later when the legitimate counterpart queried the missing payment.

TIMELINE
- T+0:00  Fraudulent email / request received by finance team
- T+{pick(['0:45','1:10','2:00','3:30'])}  Finance team processes payment without secondary verification
- T+{dur}  Discrepancy identified; IT security and management notified
- T+{pick(['1:00','2:00','3:00'])} after detection  Bank notified; recall process initiated

FINANCIAL IMPACT
Amount transferred: {impact}
Recovery status: {recovery.split('.')[0]}
Additional costs: Legal fees, forensic investigation, {pick(['$12K','$18K','$25K','$40K'])} estimated

ROOT CAUSE
{pick(['No secondary verification process existed for wire transfers above a defined threshold.',
       'The finance team was not trained to verify out-of-band requests via phone before processing.',
       'The email gateway did not flag the lookalike domain as suspicious.',
       'Access controls allowed a single approver to authorise large transfers without a second signatory.'])}

REMEDIATION
{recovery} Dual-authorisation controls implemented for all transfers above {pick(['$10K','$25K','$50K'])}. Finance team completed emergency BEC awareness training. Email gateway rules updated to flag lookalike domains.
"""
    sentence1 = f"A business email compromise attack at {co[0]} used {method}, resulting in an unauthorised transfer of {impact}."
    sentence2 = f"The fraud went undetected for {dur}, affecting {affected} finance team members involved in the approval chain."
    sentence3 = f"{recovery} Dual-authorisation controls have been implemented for all high-value transfers and the finance team completed BEC awareness training."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_insider(co, sev, dur, affected, impact):
    roles = ["departing sales manager", "disgruntled IT administrator", "contractor with database access", "former employee whose access was not revoked"]
    data = ["customer contact lists", "proprietary source code", "pricing models and contract terms", "patient health records", "employee salary data"]
    role = pick(roles)
    datum = pick(data)
    detection = pick([
        "a DLP alert triggered by large-volume file transfers to a personal cloud storage account",
        "an anomaly detected by the UEBA system flagging off-hours bulk downloads",
        "a tip from a colleague who noticed unusual USB usage",
        "an automated alert on excessive API calls outside normal working hours",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Insider Threat / Data Exfiltration
Reporting Entity: {co[0]} Security Operations

INCIDENT DESCRIPTION
{co[0]}'s security operations team received {detection} associated with a {role}. Investigation confirmed that {datum} had been exfiltrated over a period of {dur} prior to detection.

TIMELINE
- T-{dur}  Exfiltration activity begins
- T+0:00  {detection.capitalize()} generated
- T+0:30  SOC analyst begins investigation
- T+1:00  HR and Legal notified; account suspended
- T+3:00  Forensic image of device captured
- T+5:00  Scope of exfiltration confirmed

SCOPE
Data Exfiltrated: {datum.capitalize()}
Volume: {pick(['2.3 GB','4.7 GB','890 MB','11 GB'])} ({pick(['14,000','32,000','8,500','51,000'])} records)
Destination: {pick(['Personal Dropbox','Google Drive','USB drive','Personal Gmail via email attachment'])}
Individual Involved: {role.capitalize()}

INVESTIGATION FINDINGS
{pick(['The individual had submitted their resignation 5 days prior to the exfiltration.','The individual had a performance improvement plan issued 3 weeks earlier.','Access should have been revoked upon contract expiry but was not.','No prior disciplinary history was identified.'])} Forensic review confirmed data was accessed and transferred intentionally.

REMEDIATION
Account access was revoked immediately. Legal hold placed on the individual's corporate device. {pick(['Civil proceedings being considered.','Law enforcement notified.','Settlement discussions initiated.','Matter referred to HR for disciplinary proceedings.'])} Offboarding checklist updated to include same-day access revocation. DLP policies tightened to flag bulk downloads exceeding {pick(['500 MB','1 GB','2 GB'])} in a single session.
"""
    sentence1 = f"A {role} at {co[0]} exfiltrated {datum} over {dur}, detected through {detection}."
    sentence2 = f"The exfiltration affected {affected} records and is estimated to carry a business and legal exposure of {impact}."
    sentence3 = f"The individual's access was immediately revoked, a forensic hold placed on their device, and DLP policies and the offboarding checklist were updated to prevent recurrence."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_credential_stuffing(co, sev, dur, affected, impact):
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Credential Stuffing
Reporting Entity: {co[0]} Platform Security

INCIDENT DESCRIPTION
{co[0]}'s authentication infrastructure experienced a credential stuffing attack in which automated tools submitted {pick(['1.2M','3.4M','800K','5.1M'])} login attempts using credentials sourced from third-party data breaches. Approximately {affected} accounts were successfully accessed before rate limiting was activated.

TIMELINE
- T+0:00  Spike in failed login attempts detected ({pick(['12,000','28,000','45,000'])} attempts/minute)
- T+0:15  WAF rate limiting triggered; CAPTCHA enforcement activated
- T+0:45  Affected accounts identified via concurrent session analysis
- T+{dur}  All confirmed compromised accounts locked and owners notified

IMPACT
Accounts compromised: {affected}
{'Attacker accessed order history and initiated fraudulent refunds in ' + str(random.randint(5,30)) + ' accounts.' if sev in ['Critical','High'] else 'No evidence of fraudulent activity detected in affected accounts.'}
Estimated impact: {impact}

ROOT CAUSE
Credentials were sourced from publicly available breach datasets. {pick(['MFA was optional and not enabled on affected accounts.','Legacy API endpoint lacked rate limiting controls.','Bot detection was insufficient to distinguish high-volume scripted attempts.'])}

REMEDIATION
Compromised accounts were locked and password reset emails sent. CAPTCHA and progressive delays implemented on all authentication endpoints. {pick(['MFA enforcement rolled out to all accounts.','Bot management solution evaluated and selected.','Compromised credential monitoring integrated into the login flow.'])}
"""
    sentence1 = f"A credential stuffing attack against {co[0]}'s authentication systems used breach-sourced credentials to compromise {affected} customer accounts before rate limiting was activated after {dur}."
    sentence2 = f"{'Attackers accessed order history and initiated fraudulent activity in a subset of accounts, with an estimated impact of ' + impact + '.' if sev in ['Critical','High'] else 'No fraudulent activity was detected in the affected accounts, though remediation costs totalled ' + impact + '.'}"
    sentence3 = f"Compromised accounts were locked and reset; CAPTCHA and progressive delay controls were applied to all authentication endpoints and MFA enforcement was expanded."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_supply_chain(co, sev, dur, affected, impact):
    vendors = [
        ("a third-party HR software provider", "a malicious update pushed to on-premises agents"),
        ("a managed IT service provider", "compromised remote monitoring and management tooling"),
        ("a software development tool vendor", "a backdoored build pipeline dependency"),
        ("a document management SaaS vendor", "a compromised OAuth integration token"),
    ]
    vendor_desc, vector = pick(vendors)
    root_cause_supply = pick([
        "The vendor did not have code-signing controls on software updates.",
        "The MSP's RMM platform lacked MFA on the administrative console.",
        "The OAuth token granted excessive permissions beyond what the integration required.",
        "The compromised dependency was not flagged by the vendor's SBOM controls.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Supply Chain Compromise
Reporting Entity: {co[0]} Incident Response Team

INCIDENT DESCRIPTION
{co[0]} was notified by {vendor_desc} that their platform had been compromised, resulting in {vector} affecting {co[0]}'s environment. The vendor disclosed the compromise {dur} after the initial intrusion. {affected} internal systems were potentially exposed during that window.

TIMELINE
- T-{dur}  Vendor compromised; malicious code/access introduced
- T+0:00  Vendor notifies {co[0]} of compromise
- T+0:45  {co[0]} IR team activates; vendor integration suspended
- T+2:00  Forensic review of {co[0]} environment initiated
- T+8:00  Scope of internal exposure confirmed

SCOPE
Vendor: {vendor_desc.capitalize()}
Attack Vector: {vector.capitalize()}
Systems Potentially Exposed: {affected}
Evidence of Lateral Movement: {'Yes — attacker pivoted to internal network segment' if sev == 'Critical' else 'No evidence of lateral movement identified'}

ROOT CAUSE
{root_cause_supply}

REMEDIATION
The vendor integration was suspended immediately. {pick(['A forensic review found no evidence of data exfiltration from the company environment.','Forensic review is ongoing.','No sensitive data was accessible to the attacker based on network segmentation controls.'])} Third-party risk assessment process updated to require SOC 2 Type II reports and annual penetration test results from all software vendors. SBOM requirements added to vendor onboarding checklist.
"""
    sentence1 = f"A supply chain compromise of {vendor_desc} introduced {vector} into {co[0]}'s environment, with the intrusion going undetected for {dur}."
    sentence2 = f"{affected} internal systems were potentially exposed, with an estimated impact of {impact} in investigation and remediation costs."
    sentence3 = f"The vendor integration was suspended immediately; a forensic review was initiated and third-party risk assessment requirements were updated to mandate SOC 2 Type II reports and penetration test results from all vendors."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_api_key_leak(co, sev, dur, affected, impact):
    locations = [
        "a public GitHub repository committed by a developer",
        "a Slack message in a public workspace channel",
        "a Confluence page with misconfigured public access",
        "a Docker image pushed to Docker Hub with embedded credentials",
    ]
    location = pick(locations)
    services = ["cloud infrastructure", "production database", "payment processing API", "customer data warehouse"]
    service = pick(services)
    root_cause_api = pick([
        "Pre-commit hooks to block secret commits were not configured in the developer's local environment.",
        "The developer was unaware that the repository visibility had been changed to public.",
        "No secrets management tool was in use; credentials were stored directly in config files.",
        "The CI/CD pipeline did not enforce secret scanning before merge.",
    ])
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — API Key / Credential Leak
Reporting Entity: {co[0]} DevSecOps

INCIDENT DESCRIPTION
A live API key with access to {co[0]}'s {service} was discovered in {location}. The secret had been exposed for {dur} before detection via automated secret scanning. {'Activity logs show the key was accessed from non-company IP addresses.' if sev in ['High','Critical'] else 'Activity logs show no evidence of the key being used by unauthorised parties.'}

TIMELINE
- T-{dur}  Key committed / published
- T+0:00  Secret scanning tool generates alert
- T+0:20  DevSecOps investigates; key invalidated
- T+1:00  Access logs reviewed for unauthorised usage
- T+3:00  Root cause confirmed; developer notified

IMPACT
Service exposed: {service.capitalize()}
Exposure duration: {dur}
Unauthorised API calls detected: {'Yes — ' + str(random.randint(12,500)) + ' calls from external IP addresses' if sev in ['High','Critical'] else 'None detected'}
Estimated impact: {impact}

ROOT CAUSE
{root_cause_api}

REMEDIATION
Key was rotated immediately. {'Unauthorised API calls have been reviewed and no data modification was detected.' if sev in ['Medium','Low'] else 'Unauthorised API calls are being reviewed; scope of any data access is under investigation.'} Pre-commit hooks and mandatory secret scanning gates added to all repositories. Developer security awareness training updated to include secrets management. Vault or AWS Secrets Manager adoption roadmap accelerated.
"""
    sentence1 = f"A live API key granting access to {co[0]}'s {service} was inadvertently exposed in {location} and remained publicly accessible for {dur}."
    sentence2 = f"{'Unauthorised access was detected in activity logs, resulting in an estimated impact of ' + impact + '.' if sev in ['High','Critical'] else 'No unauthorised use of the key was detected; remediation costs totalled ' + impact + '.'}"
    sentence3 = f"The key was rotated immediately; pre-commit secret scanning hooks and mandatory CI pipeline gates have been deployed across all repositories."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


def make_zero_day(co, sev, dur, affected, impact):
    products = [
        ("a perimeter firewall appliance", "remote code execution"),
        ("the corporate VPN gateway", "authentication bypass"),
        ("a widely used file-transfer application", "unauthenticated file read"),
        ("the enterprise email gateway", "server-side request forgery"),
    ]
    product, vuln_type = pick(products)
    report = f"""SECURITY INCIDENT REPORT
Incident ID: INC-{random.randint(10000,99999)}
Classification: {sev} — Zero-Day Exploitation
Reporting Entity: {co[0]} Incident Response

INCIDENT DESCRIPTION
{co[0]} identified active exploitation of a zero-day vulnerability ({vuln_type}) in {product}. The vulnerability was publicly disclosed {pick(['12','36','48','72'])} hours before a vendor patch was available. During the exposure window of {dur}, attackers were able to gain unauthorised access to {affected} internal systems.

TIMELINE
- T-{dur}  Vulnerability disclosed; no patch available
- T+0:00  Threat intelligence team identifies active exploitation in the wild
- T+0:30  Emergency change process initiated; affected device isolated
- T+{pick(['2:00','4:00','6:00'])}  Compensating controls applied; network access restored
- T+{pick(['24h','48h','72h'])} after vendor patch release  Patch applied and validated

TECHNICAL DETAILS
CVE: CVE-2024-{random.randint(10000,59999)}
Vulnerability Type: {vuln_type.capitalize()}
Affected Product: {product.capitalize()}
CVSS Score: {pick(['9.8','9.1','8.8','8.1'])} (Critical)
Patch Available at Detection: No

EVIDENCE OF EXPLOITATION
{'Active exploitation confirmed — attacker established persistent access via webshell.' if sev == 'Critical' else 'Exploitation attempts detected in logs but no evidence of successful access.' if sev == 'High' else 'No exploitation of company systems confirmed; precautionary isolation applied.'}

REMEDIATION
The affected {product} was isolated from the network within {pick(['30','45','60'])} minutes. Compensating controls (WAF rules, IP allowlisting) were applied. Vendor patch was applied within {pick(['4','8','12'])} hours of release. Threat hunt conducted across the environment; {pick(['no additional indicators of compromise found','two additional suspicious connections identified and investigated'])}. Patch management SLA for critical CVEs updated to 24 hours.
"""
    sentence1 = f"A zero-day vulnerability (CVE-2024-{random.randint(10000,59999)}, {vuln_type}) in {product} was actively exploited against {co[0]} during the {dur} window between public disclosure and patch availability."
    sentence2 = f"{'Active exploitation was confirmed, with the attacker gaining access to ' + str(affected) + ' internal systems and an estimated impact of ' + impact + '.' if sev in ['Critical','High'] else 'No successful exploitation of company systems was confirmed; precautionary remediation costs totalled ' + impact + '.'}"
    sentence3 = f"The affected appliance was isolated within 30 minutes; compensating controls were applied immediately and the vendor patch was deployed within hours of release."
    ideal = f"{sentence1} {sentence2} {sentence3}"
    return report.strip(), ideal


# ── generator dispatch ────────────────────────────────────────────────────────

GENERATORS = {
    "ransomware":           make_ransomware,
    "phishing":             make_phishing,
    "data_breach_misconfig": make_data_breach_misconfig,
    "ddos":                 make_ddos,
    "bec":                  make_bec,
    "insider_threat":       make_insider,
    "credential_stuffing":  make_credential_stuffing,
    "supply_chain":         make_supply_chain,
    "api_key_leak":         make_api_key_leak,
    "zero_day":             make_zero_day,
    # types without dedicated generators fall back to ransomware template
    "data_breach_sql":      make_data_breach_misconfig,
    "rdp_brute_force":      make_ransomware,
    "vishing":              make_phishing,
    "malware_dropper":      make_ransomware,
    "cryptojacking":        make_credential_stuffing,
}

# ensure roughly even spread across types
INCIDENT_POOL = []
for t in list(GENERATORS.keys()):
    INCIDENT_POOL.extend([t] * 8)
random.shuffle(INCIDENT_POOL)
INCIDENT_POOL = INCIDENT_POOL[:100]

# ── generate ──────────────────────────────────────────────────────────────────

examples = []
for i, incident_type in enumerate(INCIDENT_POOL):
    co = pick(COMPANIES)
    sev = pickw(SEVERITIES, SEVERITY_WEIGHTS)
    dur = pick(DURATIONS[sev])
    affected_n = pick(AFFECTED_COUNTS[sev])
    affected = f"{affected_n:,}"
    impact = pick(FINANCIAL_IMPACTS[sev])

    gen = GENERATORS[incident_type]
    try:
        report, ideal = gen(co, sev, dur, affected, impact)
    except Exception as e:
        print(f"Error on {i} ({incident_type}): {e}")
        continue

    examples.append({
        "input": report,
        "ideal": ideal,
        "_meta": {"incident_type": incident_type, "company": co[0], "severity": sev},
    })

print(f"Generated {len(examples)} examples")

# ── write JSONL ───────────────────────────────────────────────────────────────

jsonl_path = "/sessions/affectionate-stoic-cori/incidents.jsonl"
with open(jsonl_path, "w") as f:
    for ex in examples:
        record = {
            "messages": [{"role": "user", "content": ex["input"]}],
            "ideal": ex["ideal"],
        }
        f.write(json.dumps(record) + "\n")

print(f"Wrote {jsonl_path}")

# ── write CSV ─────────────────────────────────────────────────────────────────

csv_path = "/sessions/affectionate-stoic-cori/incidents.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["input", "ideal"])
    writer.writeheader()
    for ex in examples:
        writer.writerow({"input": ex["input"], "ideal": ex["ideal"]})

print(f"Wrote {csv_path}")

# ── quick stats ───────────────────────────────────────────────────────────────

from collections import Counter
type_counts = Counter(ex["_meta"]["incident_type"] for ex in examples)
sev_counts = Counter(ex["_meta"]["severity"] for ex in examples)
avg_input_len = sum(len(ex["input"]) for ex in examples) / len(examples)
avg_ideal_len = sum(len(ex["ideal"]) for ex in examples) / len(examples)

print(f"\nSeverity distribution: {dict(sev_counts)}")
print(f"Incident type distribution: {dict(type_counts)}")
print(f"Avg report length: {avg_input_len:.0f} chars")
print(f"Avg ideal length:  {avg_ideal_len:.0f} chars")
print("\nSample ideal:")
print(examples[0]["ideal"])
