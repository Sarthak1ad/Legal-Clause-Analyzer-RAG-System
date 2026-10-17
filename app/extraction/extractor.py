"""Layer 2 pipeline: ParsedDocument -> ExtractedDocument.

Per clause:
  1. Rules classify the clause (fast, deterministic).
  2. The LLM (qwen2.5:3b) classifies it independently, output forced to the enum.
  3. Agreement -> high confidence. Disagreement or LLM failure -> keep a safe
     answer AND flag the clause for human review. Two cheap, independent
     opinions are how we get reliability out of a small model.
  4. Rules extract attributes with evidence spans.
  5. If a REQUIRED attribute is still missing, ask the LLM a narrow question and
     accept the answer ONLY if its quote really appears in the clause (grounding).
"""
from __future__ import annotations

from typing import Optional

from ..llm.client import LLMClient, LLMError
from ..models import ClauseCandidate, ParsedDocument, make_span
from .rules import (EXTRACTORS, classify_rules, find_defined_terms,
                    find_references)
from .schemas import (Attribute, ClauseType, Evidence, ExtractedClause,
                      ExtractedDocument, TYPE_HELP)

CLASSIFY_SYSTEM = ("You are a contract analyst. Classify the clause into exactly one type. "
                   "Answer with JSON only.")
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"clause_type": {"type": "string", "enum": [t.value for t in ClauseType]}},
    "required": ["clause_type"],
}
FILL_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": ["string", "null"]}, "quote": {"type": ["string", "null"]}},
    "required": ["value", "quote"],
}
# attributes that MUST exist for a clause type; if rules miss them, try the LLM
REQUIRED: dict[ClauseType, dict[str, str]] = {
    ClauseType.governing_law: {"governing_law": "the place whose law governs (just the name, e.g. 'New York')"},
    ClauseType.order_of_precedence: {"prevailing_document": "which document prevails (e.g. 'agreement')"},
}
MAX_CLAUSE_CHARS = 1200  # keep prompts small for a 3B model


def _classify_prompt(c: ClauseCandidate) -> str:
    types = "\n".join(f"- {t.value}: {h}" for t, h in TYPE_HELP.items())
    where = f"Heading: {c.heading or '(none)'}"
    if c.parent_heading:
        where += f" (under: {c.parent_heading})"
    return f"Types:\n{types}\n\n{where}\nClause text:\n{c.text[:MAX_CLAUSE_CHARS]}"


def llm_classify(llm: LLMClient, c: ClauseCandidate) -> ClauseType:
    out = llm.generate_json(CLASSIFY_SYSTEM, _classify_prompt(c), CLASSIFY_SCHEMA)
    try:
        return ClauseType(out["clause_type"])
    except (KeyError, ValueError, TypeError) as e:
        raise LLMError(f"invalid classification output: {out!r}") from e


def combine(rule: ClauseType, rule_conf: float, llm_type: Optional[ClauseType],
            llm_failed: bool) -> tuple[ClauseType, float, list[str]]:
    """Decide the final type, confidence and review reasons."""
    reasons: list[str] = []
    if llm_failed:
        reasons.append("LLM unavailable or invalid; used rules only")
        return rule, min(rule_conf, 0.6), reasons + (["no rule matched"] if rule_conf == 0 else [])
    if rule_conf >= 0.8:                       # strong rule evidence (heading)
        if llm_type == rule:
            return rule, 0.95, reasons
        reasons.append(f"LLM says {llm_type.value}, heading rule says {rule.value}")
        return rule, 0.6, reasons
    if llm_type == rule:                       # weak rule agrees with LLM
        return rule, 0.8, reasons
    if rule_conf > 0:                          # weak rule disagrees with LLM
        reasons.append(f"LLM says {llm_type.value}, keyword rule says {rule.value}")
        return llm_type, 0.5, reasons
    return llm_type, 0.6, reasons              # no rule matched, trust LLM


def grounded(c: ClauseCandidate, value: Optional[str], quote: Optional[str]) -> Optional[Evidence]:
    """Accept an LLM answer only if its quote is verbatim in the clause and supports the value."""
    if not value or not quote:
        return None
    idx = c.text.find(quote)
    if idx < 0:
        return None
    tokens = [w for w in value.lower().replace("-", " ").split() if len(w) >= 3]
    if not all(tok in quote.lower() for tok in tokens):
        return None
    base = c.span.start
    span = make_span(c.span.doc_id, c.span.version, c.span.page, c.section_path,
                     base + idx, base + idx + len(quote))
    return Evidence(quote=quote, span=span)


def llm_fill(llm: LLMClient, c: ClauseCandidate, name: str, desc: str) -> Optional[Attribute]:
    prompt = (f"Clause:\n{c.text[:MAX_CLAUSE_CHARS]}\n\nFind {desc}. "
              f"'value' is your answer. 'quote' must be copied exactly from the clause text. "
              f"If it is not stated, use null for both.")
    out = llm.generate_json("You extract facts from contracts. Answer with JSON only.", prompt, FILL_SCHEMA)
    ev = grounded(c, out.get("value"), out.get("quote"))
    return Attribute(name=name, value=out["value"], evidence=ev, source="llm") if ev else None


def extract_clause(c: ClauseCandidate, llm: LLMClient, use_llm_fill: bool = True) -> ExtractedClause:
    rule_type, rule_conf = classify_rules(c)
    llm_type, failed = None, False
    try:
        llm_type = llm_classify(llm, c)
    except LLMError:
        failed = True
    final, conf, reasons = combine(rule_type, rule_conf, llm_type, failed)
    if final == ClauseType.other:
        reasons.append("could not determine clause type")

    attrs = EXTRACTORS[final](c) if final in EXTRACTORS else []
    have = {a.name for a in attrs}
    for name, desc in REQUIRED.get(final, {}).items():
        if name in have:
            continue
        got = None
        if use_llm_fill and not failed:
            try:
                got = llm_fill(llm, c, name, desc)
            except LLMError:
                pass
        if got:
            attrs.append(got)
        else:
            reasons.append(f"required attribute '{name}' not found or not verifiable")

    return ExtractedClause(
        clause_id=c.clause_id, section_path=c.section_path, heading=c.heading,
        parent_heading=c.parent_heading, span=c.span, clause_type=final,
        type_by_rule=rule_type, type_by_llm=llm_type, confidence=conf,
        attributes=attrs, references=find_references(c), defined_terms=find_defined_terms(c),
        needs_review=bool(reasons), review_reasons=reasons)


def extract_document(doc: ParsedDocument, llm: LLMClient, use_llm_fill: bool = True) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id=doc.doc_id, version=doc.version, doc_type=doc.doc_type,
        effective_date=doc.effective_date, parent_ref=doc.parent_ref, llm_model=llm.model,
        clauses=[extract_clause(c, llm, use_llm_fill) for c in doc.clauses])
