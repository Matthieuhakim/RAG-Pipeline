"""Query refusal policies.

Screens user queries for PII (emails, phone numbers, SSNs, credit card
numbers) and for requests seeking legal or medical advice.  Returns a
refusal message when a policy is triggered, preventing the pipeline from
processing sensitive or liability-prone queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s\-.]?)?"
    r"(?:\(?\d{3}\)?[\s\-.]?)"
    r"\d{3}[\s\-.]?\d{4}"
    r"(?!\d)"
)
_SSN_RE = re.compile(r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[\s\-]?){13,19}\b")

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email address", _EMAIL_RE),
    ("phone number", _PHONE_RE),
    ("Social Security Number", _SSN_RE),
    ("credit card number", _CREDIT_CARD_RE),
]

# ---------------------------------------------------------------------------
# Legal / medical keyword patterns
# ---------------------------------------------------------------------------

_LEGAL_RE = re.compile(
    r"\b(?:"
    r"legal\s+advice|am\s+i\s+liable|can\s+i\s+(?:sue|be\s+sued)"
    r"|lawyer|attorney|lawsuit|legal\s+rights"
    r"|should\s+i\s+(?:sign|agree\s+to)\s+(?:this|the)\s+contract"
    r"|legal\s+obligation|legal\s+liability"
    r")\b",
    re.IGNORECASE,
)

_MEDICAL_RE = re.compile(
    r"\b(?:"
    r"medical\s+advice"
    r"|(?:should\s+i|can\s+i)\s+(?:take|stop\s+taking)\s+\w+"
    r"|(?:diagnose\s+(?:me|my)|my\s+diagnosis)"
    r"|(?:my|the)\s+(?:prescription|dosage)"
    r"|(?:what\s+are\s+(?:the\s+)?)?symptoms?\s+of"
    r"|am\s+i\s+(?:sick|ill|infected)"
    r"|is\s+it\s+(?:safe|dangerous)\s+to\s+(?:take|mix|use)"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Refusal messages
# ---------------------------------------------------------------------------

PII_REFUSAL_MESSAGE = (
    "Your query appears to contain personally identifiable information "
    "(PII). For your privacy and security, I cannot process queries "
    "that include email addresses, phone numbers, Social Security "
    "Numbers, or credit card numbers. Please remove any personal data "
    "and try again."
)

LEGAL_DISCLAIMER_MESSAGE = (
    "Your query appears to request legal advice. I am an AI assistant "
    "and cannot provide legal counsel. The information in the knowledge "
    "base should not be treated as legal advice. Please consult a "
    "qualified attorney for legal matters."
)

MEDICAL_DISCLAIMER_MESSAGE = (
    "Your query appears to request medical advice. I am an AI assistant "
    "and cannot provide medical diagnoses, treatment recommendations, or "
    "medication guidance. Please consult a qualified healthcare "
    "professional for medical matters."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RefusalResult:
    """Result of screening a query against refusal policies."""

    refused: bool
    reason: str = ""
    message: str = ""


def screen_query(query: str) -> RefusalResult:
    """Check a query against all refusal policies.

    Returns a RefusalResult indicating whether the query should be
    refused and, if so, which policy was triggered.
    """
    # PII check
    for label, pattern in _PII_PATTERNS:
        if pattern.search(query):
            return RefusalResult(
                refused=True,
                reason=f"PII detected: {label}",
                message=PII_REFUSAL_MESSAGE,
            )

    # Legal advice check
    if _LEGAL_RE.search(query):
        return RefusalResult(
            refused=True,
            reason="Legal advice request",
            message=LEGAL_DISCLAIMER_MESSAGE,
        )

    # Medical advice check
    if _MEDICAL_RE.search(query):
        return RefusalResult(
            refused=True,
            reason="Medical advice request",
            message=MEDICAL_DISCLAIMER_MESSAGE,
        )

    return RefusalResult(refused=False)
