"""Party extraction and entity resolution (Part A).

PREAMBLE EXTRACTION
-------------------
The preamble is the prose before the first numbered section.  Nearly every
contract says something like:

    "...between Acme Corp ('Customer') and Northwind Software Ltd ('Supplier')."

We use two regex passes:
  1. With role labels  – captures "Name ('Role')" pairs.
  2. Without role labels – captures "between Name and Name".

We always search for the LAST occurrence of a "between ... and ..." phrase
before the first section boundary, because preambles sometimes have a
"recitals" sentence that uses 'between' idiomatically before the operative
'between X and Y' that names the parties.

NORMALISATION
-------------
"Acme Corp., Inc." -> "acme"  (lower-case, strip punctuation, remove suffixes)
This lets us detect that "Acme Corp" and "ACME Corporation" are the same
entity before going to the LLM.

MATCHING (rapidfuzz token_sort_ratio)
--------------------------------------
>=92  : auto-merge  (same party for certain)
80-91 : ask LLM; if LLM unavailable or ambiguous -> needs_review=True, no merge
<80   : different parties

We also block auto-merge when the two normalized names have a distinguishing
token that differs between them ('acme europe' vs 'acme asia'), unless the LLM
explicitly says same_company=true.
"""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz.fuzz import token_sort_ratio

from ..llm.client import LLMClient, LLMError, NullLLM
from ..models import ParsedDocument, make_span
from ..extraction.schemas import Evidence
from .schemas import PartyMention

# Legal suffixes to strip during normalisation (order matters: longer first).
_LEGAL_SUFFIXES = [
    "incorporated", "corporation", "company", "limited", "pvt",
    "inc", "corp", "ltd", "llc", "plc", "gmbh", "co",
]

# All quote characters we want to recognise (ASCII double/single + Unicode curly).
# IMPORTANT: Do NOT use re.VERBOSE with these patterns -- VERBOSE mode strips
# spaces inside character classes, which would destroy the quote character set.
_QUOTES = '"\'' + "\u201c\u201d\u2018\u2019"  # ASCII + Unicode curly quotes

# Pattern A: company name followed by (QuoteRoleQuote)
# e.g. 'Acme Corp ("Customer")' or 'Acme Corp (\u201cCustomer\u201d)'
# Group 1 = company name; Group 2 = role label
_WITH_ROLE = re.compile(
    r'([A-Z][A-Za-z&\-.,]*(?:\s+[A-Za-z&\-.,]+){0,5})'
    r'\s*\([' + _QUOTES + r']'
    r'([A-Za-z][A-Za-z ]{1,30})'
    r'[' + _QUOTES + r']\)'
)

# Pattern B: "between X and Y"  (no role labels; X and Y are capitalised names)
# Group 1 = first party; Group 2 = second party
_NO_ROLE = re.compile(
    r'\bbetween\s+'
    r'([A-Z][A-Za-z&\-.]*(?:\s+[A-Za-z&\-.]+){0,4})'
    r'\s+and\s+'
    r'([A-Z][A-Za-z&\-.]*(?:\s+[A-Za-z&\-.]+){0,4})'
)

# Schema for LLM merge question: constrained to {"same_company": bool}
_MERGE_SCHEMA = {
    "type": "object",
    "properties": {"same_company": {"type": "boolean"}},
    "required": ["same_company"],
}

MERGE_SYSTEM = (
    "You are a legal entity resolution expert. "
    "Answer with JSON only: {\"same_company\": true/false}."
)


# ------------------------------------------------------------------ normalisation
def normalize_name(name: str) -> str:
    """Normalise a company name for fuzzy comparison.

    Steps:
    1. Lower-case and strip leading/trailing whitespace.
    2. Remove all punctuation (keep letters, digits, spaces).
    3. Remove legal-form suffixes (inc, corp, ltd ...) as whole words.
    4. Collapse runs of whitespace.

    Example: "Acme Corp., Inc." -> "acme"
    """
    s = name.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)        # punctuation -> space
    # Remove legal suffixes as whole words, longest first so "incorporated"
    # is stripped before "inc" would match inside it.
    for suf in _LEGAL_SUFFIXES:
        s = re.sub(rf"\b{re.escape(suf)}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _distinguishing_tokens(norm_a: str, norm_b: str) -> bool:
    """True when both names have a non-shared token (e.g. 'europe' vs 'asia').

    This guard prevents auto-merging "Acme Europe" with "Acme Asia" even when
    the fuzzy score is high -- they share 'acme' but differ on the geo qualifier.
    """
    ta = set(norm_a.split())
    tb = set(norm_b.split())
    common = ta & tb
    only_a = ta - common
    only_b = tb - common
    # Both sides have unique tokens -> likely different entities
    return bool(only_a and only_b)


# ------------------------------------------------------------------ preamble helpers
def _preamble_text(doc: ParsedDocument) -> tuple[str, int]:
    """Return (preamble_text, start_offset_in_full_text).

    The preamble is everything before the first top-level section.
    If there are no sections, use the first 1500 characters.
    """
    if doc.sections:
        end = doc.sections[0].span.start
    else:
        end = min(1500, len(doc.full_text))
    return doc.full_text[:end], 0


# ------------------------------------------------------------------ party extraction
def extract_parties(doc: ParsedDocument) -> list[PartyMention]:
    """Extract party names and role labels from the preamble.

    Returns one PartyMention per party found.  Evidence spans reference the
    original doc.full_text so the round-trip invariant holds:
        doc.full_text[ev.span.start:ev.span.end] == ev.quote
    """
    preamble, base = _preamble_text(doc)
    mentions: list[PartyMention] = []

    # --- Pass A: look for role-labelled parties first ---
    role_matches = list(_WITH_ROLE.finditer(preamble))
    if role_matches:
        for m in role_matches:
            raw_name = m.group(1).strip()
            role = m.group(2).strip()
            quote = m.group(0)
            abs_start = base + m.start()
            abs_end = base + m.end()
            # Sanity check: span must round-trip
            assert doc.full_text[abs_start:abs_end] == quote, (
                f"Span mismatch for {raw_name!r}"
            )
            span = make_span(doc.doc_id, doc.version, 1, "preamble",
                             abs_start, abs_end)
            mentions.append(PartyMention(
                raw_name=raw_name,
                role_label=role,
                evidence=Evidence(quote=quote, span=span),
            ))
        return mentions

    # --- Pass B: no role labels; find "between X and Y" ---
    # Take the LAST match: earlier ones may be recital boilerplate.
    no_role_matches = list(_NO_ROLE.finditer(preamble))
    if no_role_matches:
        m = no_role_matches[-1]
        for grp in (1, 2):
            raw_name = m.group(grp).strip().rstrip(".,;:")  # strip trailing sentence punctuation
            abs_start = base + m.start(grp)
            # Adjust end offset to match the stripped name length
            abs_end = base + m.start(grp) + len(raw_name)
            quote = doc.full_text[abs_start:abs_end]
            span = make_span(doc.doc_id, doc.version, 1, "preamble",
                             abs_start, abs_end)
            mentions.append(PartyMention(
                raw_name=raw_name,
                role_label=None,
                evidence=Evidence(quote=quote, span=span),
            ))

    return mentions


# ------------------------------------------------------------------ fuzzy matching
def _llm_same_company(llm: LLMClient, a: str, b: str) -> Optional[bool]:
    """Ask the LLM whether two company names refer to the same entity.

    Returns True/False from the LLM, or None if the LLM is unavailable.
    We constrain the output to {"same_company": bool} so the small model
    cannot hallucinate anything outside that schema.
    """
    prompt = (
        f"Are these two company names the same legal entity?\n"
        f"Name A: {a}\n"
        f"Name B: {b}\n"
        f"Consider common abbreviations (Corp/Corporation/Inc/Incorporated/Ltd/Limited)."
    )
    try:
        out = llm.generate_json(MERGE_SYSTEM, prompt, _MERGE_SCHEMA)
        result = out.get("same_company")
        if isinstance(result, bool):
            return result
    except LLMError:
        pass
    return None


def should_merge(
    name_a: str,
    name_b: str,
    llm: LLMClient,
) -> tuple[bool, bool, Optional[str]]:
    """Decide whether two raw party names are the same entity.

    Returns:
        (merge: bool, needs_review: bool, reason: str | None)

    The threshold logic:
    - score >= 92: auto-merge (same entity for certain).
    - 80-91: borderline; ask LLM.
        - LLM says True  -> merge.
        - LLM says False -> do NOT merge.
        - LLM unavailable -> do NOT merge, set needs_review=True.
    - < 80: different parties.

    We also block auto-merge when both names have a unique distinguishing token
    (e.g. 'acme europe' vs 'acme asia'), deferring to LLM if in 80-91 range.
    """
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)

    # Identical after normalization -> definitely same
    if norm_a == norm_b:
        return True, False, None

    score = token_sort_ratio(norm_a, norm_b)
    distinguished = _distinguishing_tokens(norm_a, norm_b)

    if score >= 92 and not distinguished:
        # High score, no geographic/qualifier distinction -> auto-merge
        return True, False, None

    if score >= 80:
        # Borderline range: consult the LLM
        result = _llm_same_company(llm, name_a, name_b)
        if result is True:
            return True, False, None
        if result is False:
            return False, False, None
        # LLM unavailable -> flag for review, do NOT merge
        reason = (
            f"Fuzzy score={score:.0f} (borderline 80-92) for "
            f"{name_a!r} vs {name_b!r}; LLM unavailable for confirmation"
        )
        return False, True, reason

    # Low score: clearly different parties
    return False, False, None
