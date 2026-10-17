"""Deterministic extraction: keyword classifier + regex attribute extractors.

Why rules first? A 3B model is good at "which category is this?" but unreliable
at reading numbers and quotes. Regex is exact, free and explainable, and every
match is a real slice of the clause, so its evidence span is correct by
construction. The LLM is only used where rules cannot decide (see extractor.py).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from ..ingestion.metadata import _DATE, parse_date
from ..models import ClauseCandidate, make_span
from .schemas import Attribute, ClauseType, DefinedTerm, Evidence, Reference

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "thirty": 30,
         "forty-five": 45, "sixty": 60, "ninety": 90}
NUM = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(WORDS, key=len, reverse=True)) + ")"
Q_OPEN, Q_CLOSE = "\"“", "\"”"


# ---------------------------------------------------------------- helpers
def _ev(c: ClauseCandidate, s: int, e: int) -> Evidence:
    """Evidence for c.text[s:e]; offsets are relative to the clause start."""
    base = c.span.start
    span = make_span(c.span.doc_id, c.span.version, c.span.page, c.section_path,
                     base + s, base + e)
    return Evidence(quote=c.text[s:e], span=span)


def _attr(c, name, value, s, e) -> Attribute:
    return Attribute(name=name, value=value, evidence=_ev(c, s, e))


def _digits(m: re.Match, g: int = 2) -> int:
    return int(m.group(g))


# ---------------------------------------------------------------- classification
HEADING_RULES: list[tuple[ClauseType, str]] = [
    (ClauseType.indemnification, r"indemn"),
    (ClauseType.limitation_of_liability, r"limitation of liability|liability cap|^cap$|^exclusions?$"),
    (ClauseType.governing_law, r"governing law|jurisdiction"),
    (ClauseType.order_of_precedence, r"precedence"),
    (ClauseType.termination, r"terminat"),
    (ClauseType.term_renewal, r"\bterm\b|renewal"),
    (ClauseType.payment, r"\bfees?\b|payment|invoice"),
    (ClauseType.confidentiality, r"confidential|non-?disclosure"),
    (ClauseType.ip_ownership, r"intellectual property|ownership|\bip\b"),
    (ClauseType.notices, r"notices?"),
    (ClauseType.definitions, r"definitions?"),
    (ClauseType.services_scope, r"^services?$|scope of (?:work|services)"),
]

BODY_RULES: list[tuple[ClauseType, str]] = [
    (ClauseType.indemnification, r"indemnif"),
    (ClauseType.limitation_of_liability, r"aggregate liability|consequential damages"),
    (ClauseType.governing_law, r"governed by the laws?"),
    (ClauseType.order_of_precedence, r"prevails? over"),
    (ClauseType.term_renewal, r"renews? automatically|expires"),
    (ClauseType.payment, r"invoice|annual fee|\bfees?\b"),
    (ClauseType.confidentiality, r"confidential information"),
    (ClauseType.notices, r"notices? must be"),
]


def classify_rules(c: ClauseCandidate) -> tuple[ClauseType, float]:
    """Return (type, confidence). Heading 0.9, parent heading 0.8, body keywords 0.5."""
    for text, conf in ((c.heading, 0.9), (c.parent_heading, 0.8)):
        if text:
            for ctype, pat in HEADING_RULES:
                if re.search(pat, text.strip(), re.IGNORECASE):
                    return ctype, conf
    hits = [(t, len(re.findall(p, c.text, re.IGNORECASE))) for t, p in BODY_RULES]
    hits = [h for h in hits if h[1] > 0]
    if hits:
        return max(hits, key=lambda h: h[1])[0], 0.5
    return ClauseType.other, 0.0


# ---------------------------------------------------------------- attribute extractors
def _cap(c: ClauseCandidate) -> list[Attribute]:
    out: list[Attribute] = []
    t = c.text
    m = re.search(rf"({NUM})\s+times\s*\((\d+(?:\.\d+)?)x\)", t, re.I)
    if m:
        out.append(_attr(c, "cap_multiple", float(m.group(2)), m.start(), m.end()))
    else:
        m = re.search(r"not\s+exceed\s+the\s+(?:annual\s+)?fees", t, re.I)
        if m:  # 'shall not exceed the annual Fees' means 1x
            out.append(_attr(c, "cap_multiple", 1.0, m.start(), m.end()))
    if out:
        b = re.search(r"\bthe\s+((?:annual\s+)?fees)\b", t, re.I)
        if b:
            out.append(_attr(c, "cap_basis", b.group(1).lower(), b.start(1), b.end(1)))
        lb = re.search(rf"({NUM})\s*\((\d+)\)\s+months", t, re.I)
        if lb:
            out.append(_attr(c, "cap_lookback_months", _digits(lb), lb.start(), lb.end()))
        co = re.search(r"\bexcept\b[^.]{0,80}?\bSection\s+(\d+(?:\.\d+)*(?:\([a-z]+\))*)", t)
        if co:
            out.append(_attr(c, "cap_carve_outs", [co.group(1)], co.start(), co.end()))
    return out


def _survival(c) -> list[Attribute]:
    m = re.search(rf"surviv\w*[^.]{{0,40}}?({NUM})\s*\((\d+)\)\s+years?", c.text, re.I)
    return [_attr(c, "survival_years", _digits(m), m.start(), m.end())] if m else []


def x_indemnification(c) -> list[Attribute]:
    out: list[Attribute] = []
    t = c.text
    m = re.search(r"\b([A-Z][a-z]+) shall (?:defend,?\s+(?:and\s+)?)?indemnify "
                  r"(?:and hold harmless )?([A-Z][a-z]+)", t)
    if m:
        out += [_attr(c, "indemnitor", m.group(1), m.start(), m.end()),
                _attr(c, "indemnitee", m.group(2), m.start(), m.end())]
        each = re.search(r"\b[Ee]ach party\b[^.]{0,40}indemnif", t)
        out.append(_attr(c, "direction", "mutual", each.start(), each.end()) if each
                   else _attr(c, "direction", "one_way", m.start(), m.end()))
        gn = re.search(r"gross negligence", t, re.I)
        out.append(_attr(c, "covers_gross_negligence", True, gn.start(), gn.end()) if gn
                   else Attribute(name="covers_gross_negligence", value=False, source="absence"))
        if c.sub_spans:
            first, last = c.sub_spans[0], c.sub_spans[-1]
            s, e = first.start - c.span.start, last.end - c.span.start
            events = [c.text[x.start - c.span.start:x.end - c.span.start] for x in c.sub_spans]
            out.append(_attr(c, "covered_events", events, s, e))
    return out + _cap(c)


def x_limitation(c) -> list[Attribute]:
    out: list[Attribute] = []
    ind = re.search(r"indirect or consequential", c.text, re.I)
    if ind:
        out.append(_attr(c, "excludes_indirect_damages", True, ind.start(), ind.end()))
    mu = re.search(r"\b(Neither party|Each party)\b", c.text)
    if mu:
        out.append(_attr(c, "mutual", True, mu.start(), mu.end()))
    return out + _cap(c)


def x_term(c) -> list[Attribute]:
    out: list[Attribute] = []
    t = c.text
    m = re.search(rf"(?:continues\s+for|expires|remains\s+in\s+effect\s+for|for\s+a\s+term\s+of|"
                  rf"initial\s+term\s+of)\s+({NUM})\s*\((\d+)\)\s+years?", t, re.I)
    if m:
        out.append(_attr(c, "term_years", _digits(m), m.start(), m.end()))
    m = re.search(r"renews?\s+automatically", t, re.I)
    if m:
        out.append(_attr(c, "auto_renew", True, m.start(), m.end()))
    m = re.search(rf"successive\s+({NUM})\s*\((\d+)\)\s+years?", t, re.I)
    if m:
        out.append(_attr(c, "renewal_term_years", _digits(m), m.start(), m.end()))
    m = re.search(rf"({NUM})\s*\((\d+)\)\s+days['’]?\s+(?:prior\s+)?written\s+notice\s+of\s+non-renewal", t, re.I)
    if m:
        out.append(_attr(c, "non_renewal_notice_days", _digits(m), m.start(), m.end()))
    for name, pat in (("start_date", rf"begins\s+on\s+({_DATE})"),
                      ("end_date", rf"(?:ends|expires)\s+on\s+({_DATE})")):
        m = re.search(pat, t)
        d = parse_date(m.group(1)) if m else None
        if d:
            out.append(_attr(c, name, d.isoformat(), m.start(), m.end()))
    return out + _survival(c)


def x_payment(c) -> list[Attribute]:
    out: list[Attribute] = []
    t = c.text
    m = re.search(rf"within\s+({NUM})\s*\((\d+)\)\s+days", t, re.I)
    if m:
        out.append(_attr(c, "payment_days", _digits(m), m.start(), m.end()))
    m = re.search(r"\((\d+(?:\.\d+)?)%\)\s+per\s+month", t)
    if m:
        out.append(_attr(c, "late_interest_pct_per_month", float(m.group(1)), m.start(), m.end()))
    m = re.search(r"\b(USD|EUR|INR|GBP|\$)\s?(\d[\d,]*(?:\.\d+)?)", t)
    if m:
        out.append(_attr(c, "fee_amount", float(m.group(2).replace(",", "")), m.start(), m.end()))
        out.append(_attr(c, "fee_currency", m.group(1), m.start(1), m.end(1)))
    return out


def x_confidentiality(c) -> list[Attribute]:
    out = _survival(c)
    m = re.search(r"\bEach party\b", c.text)
    if m:
        out.append(_attr(c, "mutual", True, m.start(), m.end()))
    return out


def x_governing_law(c) -> list[Attribute]:
    m = re.search(r"laws?\s+of\s+(?:the\s+)?(?:State\s+of\s+|Commonwealth\s+of\s+|Republic\s+of\s+)?"
                  r"([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)", c.text)
    return [_attr(c, "governing_law", m.group(1), m.start(), m.end())] if m else []


def x_precedence(c) -> list[Attribute]:
    m = re.search(r"(?:this\s+)?(Agreement|Statement\s+of\s+Work)\s+(?:shall\s+)?prevails?\s+over\s+"
                  r"(?:any\s+)?([A-Za-z ]{3,30}?)(?=[.,;])", c.text, re.I)
    if not m:
        return []
    return [_attr(c, "prevailing_document", m.group(1).lower(), m.start(), m.end()),
            _attr(c, "subordinate_document", m.group(2).lower(), m.start(), m.end())]


def x_ip(c) -> list[Attribute]:
    m = re.search(r"\b([A-Z][a-z]+)\s+(?:owns|retains\s+ownership)", c.text)
    return [_attr(c, "owner", m.group(1), m.start(), m.end())] if m else []


EXTRACTORS: dict[ClauseType, Callable] = {
    ClauseType.indemnification: x_indemnification,
    ClauseType.limitation_of_liability: x_limitation,
    ClauseType.term_renewal: x_term,
    ClauseType.payment: x_payment,
    ClauseType.confidentiality: x_confidentiality,
    ClauseType.governing_law: x_governing_law,
    ClauseType.order_of_precedence: x_precedence,
    ClauseType.ip_ownership: x_ip,
}


# ---------------------------------------------------------------- references + defined terms
def find_references(c: ClauseCandidate) -> list[Reference]:
    out = []
    for m in re.finditer(r"\bSections?\s+(\d+(?:\.\d+)*(?:\([a-z]+\))*)", c.text):
        out.append(Reference(kind="section", target=m.group(1), evidence=_ev(c, m.start(), m.end())))
    for m in re.finditer(r"\bSchedule\s+([A-Z])\b", c.text):
        out.append(Reference(kind="schedule", target=m.group(1), evidence=_ev(c, m.start(), m.end())))
    return out


def find_defined_terms(c: ClauseCandidate) -> list[DefinedTerm]:
    out = []
    for m in re.finditer(rf"[{Q_OPEN}]([A-Z][A-Za-z ]{{1,40}})[{Q_CLOSE}]\s+means\s+([^.\n]+)", c.text):
        out.append(DefinedTerm(term=m.group(1), definition=m.group(2).strip(),
                               evidence=_ev(c, m.start(), m.end())))
    for m in re.finditer(rf"\((?:the\s+)?[{Q_OPEN}]([A-Z][A-Za-z ]{{1,40}})[{Q_CLOSE}]\)", c.text):
        out.append(DefinedTerm(term=m.group(1), evidence=_ev(c, m.start(), m.end())))
    return out
