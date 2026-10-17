"""Output of Layer 2. Every attribute can point back to the exact source text."""
from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from ..models import ParentRef, Span


class ClauseType(str, Enum):
    indemnification = "indemnification"
    limitation_of_liability = "limitation_of_liability"
    term_renewal = "term_renewal"
    termination = "termination"
    payment = "payment"
    confidentiality = "confidentiality"
    governing_law = "governing_law"
    order_of_precedence = "order_of_precedence"
    ip_ownership = "ip_ownership"
    notices = "notices"
    definitions = "definitions"
    services_scope = "services_scope"
    other = "other"


TYPE_HELP = {
    ClauseType.indemnification: "one party defends/compensates the other for third-party claims, and caps on that",
    ClauseType.limitation_of_liability: "caps on liability, excluded damages (indirect, consequential)",
    ClauseType.term_renewal: "contract duration, start/end, automatic renewal, non-renewal notice",
    ClauseType.termination: "rights and procedures to end the contract early",
    ClauseType.payment: "fees, invoices, payment deadlines, late interest",
    ClauseType.confidentiality: "keeping non-public information secret, survival period",
    ClauseType.governing_law: "which jurisdiction's law governs the contract",
    ClauseType.order_of_precedence: "which document wins if documents conflict",
    ClauseType.ip_ownership: "who owns deliverables and pre-existing materials",
    ClauseType.notices: "how formal notices must be sent",
    ClauseType.definitions: "definitions of capitalised terms",
    ClauseType.services_scope: "description of the services or work to be performed",
    ClauseType.other: "anything else",
}


class Evidence(BaseModel):
    quote: str
    span: Span


AttrValue = Union[bool, int, float, str, list[str]]


class Attribute(BaseModel):
    name: str
    value: AttrValue
    evidence: Optional[Evidence] = None      # None only for source='absence'
    source: Literal["rule", "llm", "absence"] = "rule"


class Reference(BaseModel):
    kind: Literal["section", "schedule"]
    target: str                              # '12.3', '9.1(b)', 'A'
    evidence: Evidence


class DefinedTerm(BaseModel):
    term: str
    definition: Optional[str] = None
    evidence: Evidence


class ExtractedClause(BaseModel):
    clause_id: str
    section_path: str
    heading: Optional[str] = None
    parent_heading: Optional[str] = None
    span: Span
    clause_type: ClauseType
    type_by_rule: ClauseType
    type_by_llm: Optional[ClauseType] = None
    confidence: float
    attributes: list[Attribute] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    defined_terms: list[DefinedTerm] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    def attr(self, name: str) -> Optional[Attribute]:
        return next((a for a in self.attributes if a.name == name), None)


class ExtractedDocument(BaseModel):
    doc_id: str
    version: int
    doc_type: str
    effective_date: Optional[Date] = None
    parent_ref: Optional[ParentRef] = None
    llm_model: str
    clauses: list[ExtractedClause]
