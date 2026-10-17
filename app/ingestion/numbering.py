"""Legal numbering recognition: Article IV / 9. / 9.2 / (a) / (i).

Rule-based on purpose. Numbering is highly regular, rules are fast, free and
deterministic. (An LLM fallback for messy documents is a later improvement.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ARTICLE_RE = re.compile(r"^(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+)\b[\s.:\-–—]*(.*)$")
DECIMAL_RE = re.compile(r"^(\d+(?:\.\d+)*)(\.?)\s+(\S.*)$")
PAREN_RE = re.compile(r"^\(([A-Za-z]+)\)\s+(\S.*)$")

_ROMAN_CHARS = set("ivxlc")
_SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "on", "or", "a", "an", "by", "with"}


@dataclass
class Marker:
    kind: str            # article | decimal | letter | roman
    label: str           # "9.2", "(a)", "Article IV"
    level: int           # article=0, decimal=dots+1; letter/roman decided by tree
    title: Optional[str]
    body: str            # text after the number/title on the same line


def _is_title_like(s: str) -> bool:
    """'Limitation of Liability' yes, 'Supplier shall defend Customer' no."""
    words = s.replace("&", "and").split()
    if not 1 <= len(words) <= 8:
        return False
    return all(w[0].isupper() or w.lower() in _SMALL_WORDS or not w[0].isalpha() for w in words)


def split_title(rest: str) -> tuple[Optional[str], str]:
    """'Indemnity. Supplier shall...' -> ('Indemnity', 'Supplier shall...')."""
    m = re.match(r"^(.{1,80}?)[.:]\s+(\S.*)$", rest)
    if m and _is_title_like(m.group(1)):
        return m.group(1).strip(), m.group(2)
    # whole line is a heading (no sentence after it)
    stripped = rest.rstrip(".:")
    if len(rest) <= 80 and not rest.endswith((";", ",")) and _is_title_like(stripped):
        return stripped, ""
    return None, rest


def is_numbered_line(line: str) -> bool:
    return classify(line, last_letter=None) is not None


def classify(line: str, last_letter: Optional[str]) -> Optional[Marker]:
    m = ARTICLE_RE.match(line)
    if m:
        title = m.group(2).strip().rstrip(".:") or None
        return Marker("article", f"Article {m.group(1)}", 0, title, "")

    m = DECIMAL_RE.match(line)
    if m:
        label, trailing_dot, rest = m.group(1), m.group(2), m.group(3)
        # '30 days after...' is body text, not a clause number
        if "." in label or trailing_dot:
            title, body = split_title(rest)
            return Marker("decimal", label, label.count(".") + 1, title, body)
        return None

    m = PAREN_RE.match(line)
    if m:
        tok, rest = m.group(1), m.group(2)
        low = tok.lower()
        if len(low) > 1:
            if set(low) <= _ROMAN_CHARS:
                return Marker("roman", f"({tok})", -1, None, rest)
            return None
        # single char: '(i)' is roman unless it follows '(h)'
        if low in ("i", "v", "x") and last_letter != chr(ord(low) - 1):
            return Marker("roman", f"({tok})", -1, None, rest)
        return Marker("letter", f"({tok})", -1, None, rest)
    return None
