"""Jurisdiction normalization and node generation (Layer 4 Part D).

Normalizes governing law text (e.g. 'State of New York' -> 'New York',
'NY' -> 'New York', 'England and Wales' -> 'England') and creates
Jurisdiction nodes.
"""
from __future__ import annotations

import re
from typing import Optional
from .model import JurisdictionNode

_JURISDICTION_ALIASES: dict[str, str] = {
    "ny": "New York",
    "new york": "New York",
    "state of new york": "New York",
    "ca": "California",
    "california": "California",
    "state of california": "California",
    "de": "Delaware",
    "delaware": "Delaware",
    "state of delaware": "Delaware",
    "england": "England",
    "england and wales": "England",
    "uk": "England",
    "united kingdom": "England",
}


def normalize_jurisdiction(raw_text: Optional[str]) -> Optional[str]:
    """Normalize a jurisdiction string into its canonical form.

    Strips leading 'State of', 'the laws of', punctuation, and applies
    alias lookup. Returns None if raw_text is empty.
    """
    if not raw_text:
        return None
    s = raw_text.strip().rstrip(".,;:")
    # Strip phrases like "the laws of", "the State of", "laws of the State of"
    s = re.sub(
        r"^(?:the\s+)?(?:laws?\s+of\s+(?:the\s+)?)?(?:state\s+of\s+)?",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = re.sub(r"\s+(?:state|laws?)$", "", s, flags=re.IGNORECASE).strip()
    key = s.lower()
    return _JURISDICTION_ALIASES.get(key, s)


def make_jurisdiction_node(raw_name: str) -> Optional[JurisdictionNode]:
    """Construct a JurisdictionNode with normalized name and ID."""
    canonical = normalize_jurisdiction(raw_name)
    if not canonical:
        return None
    return JurisdictionNode(
        id=f"jurisdiction:{canonical.lower().replace(' ', '_')}",
        name=canonical,
    )
