"""
HemoSmart - Role Permission Matrix (FR-08)
---------------------------------------------
The single source of truth for what each role may do. REST routes
(backend.dependencies.rbac.require_permission), the chat assistant
(agents/chat_agent.py) and the tests all read this table, so the
dashboard and the chatbot can't disagree about who may see what.

Each capability lists the non-Admin roles that hold it. Admin holds
every capability (full access), exactly as `require_role` has always
treated it. Derived from the route guards as they stood before this
module existed, plus one owner decision: Donor Coordinators may read
inventory, to coordinate donor calls against stock (D65).
"""

ADMIN = "Admin"
ROLES = ("Admin", "Blood Bank Manager", "Hospital Staff", "Donor Coordinator", "Auditor")

_BBM, _HS, _DC, _AUD = "Blood Bank Manager", "Hospital Staff", "Donor Coordinator", "Auditor"
_EVERYONE = frozenset({_BBM, _HS, _DC, _AUD})

CAPABILITIES = {
    "patients:read":         frozenset({_HS, _BBM, _AUD}),
    "predict:run":           frozenset({_HS, _BBM}),
    "explain:run":           frozenset({_HS, _BBM}),
    "cases:run":             frozenset({_HS, _BBM}),
    "transfusions:write":    frozenset({_HS, _BBM}),
    "transfusions:read":     frozenset({_HS, _BBM, _AUD}),
    "inventory:read":        frozenset({_HS, _BBM, _AUD, _DC}),
    "inventory:write":       frozenset({_BBM}),
    "forecast:read":         frozenset({_BBM, _AUD}),
    "donors:read":           frozenset({_BBM, _DC}),
    "donor_alerts:send":     frozenset({_BBM, _DC}),
    "donor_responses:write": frozenset({_BBM, _DC}),
    "autonomy:read":         frozenset({_BBM, _AUD}),
    "autonomy:run":          frozenset({_BBM}),
    "agents:monitor":        frozenset({_BBM, _AUD}),  # agent monitor page, read-only (D74)
    "audit:read":            frozenset({_AUD}),
    "users:manage":          frozenset(),
    "chat:use":              _EVERYONE,
    "guidelines:read":       _EVERYONE,  # WHO handbook text only, no patient data
}

# How a capability is named to a person ("Not authorised: <label> ...").
LABELS = {
    "patients:read": "patient records and predictions",
    "predict:run": "transfusion prediction",
    "explain:run": "prediction explanations",
    "cases:run": "AI case processing",
    "transfusions:write": "reserving and issuing blood",
    "transfusions:read": "transfusion records",
    "inventory:read": "blood inventory",
    "inventory:write": "inventory changes",
    "forecast:read": "the demand forecast and supply gap",
    "donors:read": "donor records and alerts",
    "donor_alerts:send": "sending donor alerts",
    "donor_responses:write": "recording donor replies",
    "autonomy:read": "autonomous agent activity",
    "autonomy:run": "running the agents",
    "agents:monitor": "agent monitoring",
    "audit:read": "the audit log",
    "users:manage": "user management",
    "chat:use": "the assistant",
    "guidelines:read": "WHO transfusion guidelines",
}

# What the chatbot can answer, per capability. Built from the matrix, so
# the assistant never suggests a question it would then refuse.
CHAT_TOPICS = (
    "inventory:read", "forecast:read", "predict:run", "explain:run", "patients:read",
    "donors:read", "donor_alerts:send", "autonomy:read", "audit:read", "guidelines:read",
)

SUGGESTIONS = {
    "inventory:read": "How much O- blood do we have?",
    "forecast:read": "Any shortages or wastage in the next 7 days?",
    "predict:run": "Hemoglobin 7.2, platelets 65000, INR 1.9, age 58, Emergency surgery",
    "patients:read": "Show the most recent patients",
    "donors:read": "Any pending donor responses?",
    "donor_alerts:send": "Alert donors for O- platelets",
    "autonomy:read": "What did the last agent cycle do?",
    "audit:read": "Show refused chat requests",
    "guidelines:read": "What does WHO say about platelet transfusion thresholds?",
}


def is_permitted(role: str, capability: str) -> bool:
    """Deny by default: an unknown capability or role is refused."""
    if capability not in CAPABILITIES:
        return False
    return role == ADMIN or role in CAPABILITIES[capability]


def denial(role: str, capability: str) -> str:
    """The fixed refusal, shared by the REST routes (403 detail) and chat."""
    return f"Not authorised: the {role} role doesn't have access to {LABELS.get(capability, 'that request')}."


def capabilities_for(role: str) -> list:
    return [c for c in CAPABILITIES if is_permitted(role, c)]


def chat_topics_for(role: str) -> list:
    """Labels of what this role can ask the assistant about."""
    return [LABELS[c] for c in CHAT_TOPICS if is_permitted(role, c)]


def suggestions_for(role: str, limit: int = 4) -> list:
    return [SUGGESTIONS[c] for c in CHAT_TOPICS if c in SUGGESTIONS and is_permitted(role, c)][:limit]

# Chat intent -> the capability it needs. An intent missing here is
# refused (deny by default). "help"/"unclear" only need chat access.
INTENT_CAPABILITY = {
    "predict": "predict:run",
    "explain": "explain:run",
    "inventory": "inventory:read",
    "forecast": "forecast:read",
    "action": "donor_alerts:send",
    "donors": "donors:read",
    "donor_responses": "donors:read",
    "patients": "patients:read",
    "autonomy": "autonomy:read",
    "audit": "audit:read",
    "guidelines": "guidelines:read",
    "help": "chat:use",
    "smalltalk": "chat:use",
    "unclear": "chat:use",
}
