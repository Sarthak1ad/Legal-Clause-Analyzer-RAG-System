"""Step 4: document-level metadata: type, effective date, parent agreement.

Pure regex on purpose: dates and document links must be deterministic.
The parent link ("issued under the MSA dated ...") becomes a PARENT_OF edge
in the graph later (Layer 4).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from ..models import ParentRef

_DATE = (r"(?:[A-Z][a-z]{2,8}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
         r"|\d{1,2}(?:st|nd|rd|th)? (?:day of )?[A-Z][a-z]{2,8},? \d{4}"
         r"|\d{4}-\d{2}-\d{2})")

_TYPE_RULES = [  # order matters: 'Amendment to MSA' is an AMENDMENT, not an MSA
    ("AMENDMENT", r"\bamendment\b"),
    ("SOW", r"statement of work|\bSOW\b|work order"),
    ("MSA", r"master (?:services|subscription)? ?agreement|\bMSA\b"),
    ("NDA", r"non-?disclosure|confidentiality agreement|\bNDA\b"),
    ("ORDER_FORM", r"order form"),
]

_PARENT_RE = re.compile(
    r"(?:issued under|pursuant to|subject to|governed by|amends?|to|under)\s+"
    r"(?:that certain\s+|the\s+)?"
    r"((?:Master (?:Services |Subscription )?Agreement)|MSA)"
    r"[^.]{0,80}?dated(?:\s+as\s+of)?\s+(" + _DATE + ")",
    re.IGNORECASE,
)
_EFFECTIVE_RE = re.compile(
    r"(?:effective\s+(?:as\s+of|date)|effective|dated(?:\s+as\s+of)?|entered into (?:as of|on))"
    r"[:\s]*(?:the\s+)?(" + _DATE + ")",
    re.IGNORECASE,
)


def parse_date(s: str) -> Optional[date]:
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s.strip().replace("day of ", ""))
    s = s.replace(",", "").replace(".", "")
    for fmt in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def detect_title(blocks_text: list[str]) -> Optional[str]:
    for t in blocks_text[:5]:
        if 3 <= len(t) <= 120:
            return t
    return None


def detect_doc_type(title: Optional[str], head: str) -> str:
    probe = title or ""
    for dtype, pat in _TYPE_RULES:
        if re.search(pat, probe, re.IGNORECASE):
            return dtype
    for dtype, pat in _TYPE_RULES:  # fall back to the opening paragraph
        if re.search(pat, head[:400], re.IGNORECASE):
            return dtype
    return "OTHER"


def extract_metadata(blocks_text: list[str], full_text: str):
    head = full_text[:4000]
    title = detect_title(blocks_text)
    doc_type = detect_doc_type(title, head)

    parent = None
    masked = head
    pm = _PARENT_RE.search(head)
    if pm:
        parent = ParentRef(doc_type="MSA", date=parse_date(pm.group(2)), raw=pm.group(0))
        masked = head[:pm.start()] + " " * (pm.end() - pm.start()) + head[pm.end():]

    eff = None
    em = _EFFECTIVE_RE.search(masked)  # parent's date must not be mistaken for ours
    if em:
        eff = parse_date(em.group(1))
    return title, doc_type, eff, parent
