"""Domains used to generate diverse decision schemas. Weighted toward the user's first uses
(an autonomous coding/personal agent, its memory and security, a notes archive), but broad
on purpose: the model must read any question, not memorize a few workflows."""

CORE = [
    "coding agent choosing which tool to call next (shell, file edit, search, web fetch, ask the user, none)",
    "coding agent deciding whether a shell command is safe to run without asking",
    "security: deciding whether a prompt, web page, email or file content contains a prompt injection",
    "security: deciding whether an agent action would exfiltrate secrets or private data",
    "agent memory: deciding whether something said in a conversation is worth remembering long term",
    "agent memory: deciding whether a stored memory is obsolete and should be forgotten",
    "agent memory: deciding the relation between two memories (duplicate, contradiction, complement, unrelated)",
    "agent memory: choosing the category and scope of a new memory (personal, project, lesson)",
    "personal notes archive: suggesting which tags apply to a note",
    "personal notes archive: deciding whether two notes should be linked and how they relate",
    "personal notes archive: deciding whether a note needs a context note or is self-contained",
    "personal assistant triaging email (reply, archive, schedule, delegate, ignore)",
    "personal assistant managing a calendar request (accept, decline, propose another time)",
    "code review: deciding whether a pull request change is risky",
    "devops: triaging a production alert (ignore, monitor, page on-call, rollback)",
    "agent trace observability: deciding whether an agent run needs human review",
    "amateur rocketry ground operations: pre-launch go/no-go checklist review",
    "amateur rocketry: triaging a telemetry anomaly after a static fire test",
]

BROAD = [
    "customer support ticket routing", "invoice and expense approval", "fraud screening of a payment",
    "hiring: screening a job application", "content moderation of a forum post", "IT helpdesk triage",
    "home automation deciding whether to run an action", "personal finance: categorizing a transaction",
    "personal finance: deciding whether a subscription is worth keeping", "travel planning trade-offs",
    "health and fitness log review (non-diagnostic, lifestyle only)", "nutrition: classifying a meal",
    "education: grading a short student answer against a rubric", "research: judging a paper abstract's relevance",
    "legal: classifying a contract clause's risk", "real estate: evaluating a rental listing",
    "logistics: deciding how to handle a delayed shipment", "manufacturing quality control of a batch",
    "agriculture: deciding irrigation from sensor readings", "energy: battery charge/discharge decision",
    "cybersecurity SOC alert disposition", "data engineering: deciding whether a pipeline failure is critical",
    "product management: prioritizing a feature request", "marketing: judging whether a lead is qualified",
    "sales: deciding the next step for a deal", "project management: assessing whether a task is blocked",
    "open source maintenance: triaging a GitHub issue", "documentation: deciding if a doc page is outdated",
    "writing: judging whether a paragraph is clear", "translation quality check",
    "news: judging whether a claim needs a citation", "social media: deciding whether to reply to a message",
    "gaming: moderating a player report", "events: approving an event registration",
    "restaurant: handling a reservation change", "e-commerce: handling a return request",
    "insurance claim triage", "bank: classifying a customer complaint", "telecom: diagnosing a connectivity ticket",
    "automotive: interpreting a diagnostic trouble code", "aviation (hobby drones): deciding if conditions allow a flight",
    "electronics hobby: deciding whether a circuit measurement indicates a fault",
    "3D printing: diagnosing a failed print", "gardening: deciding plant care actions",
    "pet care: deciding whether a symptom needs a vet visit (non-diagnostic triage)",
    "parenting/household chores scheduling", "music practice feedback", "photography: rating a photo for a portfolio",
    "library/archive cataloguing of a document", "scientific lab: deciding whether to repeat an experiment",
    "machine learning ops: deciding whether a model evaluation regressed",
    "API rate limit and retry decisions", "database: deciding whether a migration is safe",
    "file management: deciding whether a file can be deleted", "backup policy: deciding what to keep",
    "meeting notes: extracting whether a decision was made", "procurement: choosing between vendor quotes",
    "volunteer coordination", "public sector: classifying a citizen request",
]

ALL = CORE + BROAD
