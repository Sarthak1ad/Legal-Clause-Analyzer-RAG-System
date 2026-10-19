"""Graph data models for Layer 4: Knowledge graph nodes, edges, and views.

Every node and edge has a deterministic string ID and provenance tracking.
The graph is built in-memory first as typed Pydantic models, then exported
to JSON and loaded into Neo4j.
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Typed Node Models
# ----------------------------------------------------------------------

class ContractNode(BaseModel):
    id: str
    label: Literal["Contract"] = "Contract"
    doc_id: str
    version: int
    doc_type: str
    title: Optional[str] = None
    effective_date: Optional[str] = None
    status: str = "active"


class ClauseNode(BaseModel):
    id: str
    label: Literal["Clause"] = "Clause"
    clause_id: str
    section_path: str
    heading: Optional[str] = None
    clause_type: str
    span_id: str
    text_hash: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    needs_review: bool = False
    status: Literal["active", "superseded"] = "active"
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None


class PartyNode(BaseModel):
    id: str
    label: Literal["Party"] = "Party"
    party_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class DefinedTermNode(BaseModel):
    id: str
    label: Literal["DefinedTerm"] = "DefinedTerm"
    term: str
    definition: Optional[str] = None
    span_id: str


class StandardTemplateNode(BaseModel):
    id: str
    label: Literal["StandardTemplate"] = "StandardTemplate"
    template_id: str
    clause_type: str
    version: str


class JurisdictionNode(BaseModel):
    id: str
    label: Literal["Jurisdiction"] = "Jurisdiction"
    name: str


class DateEventNode(BaseModel):
    id: str
    label: Literal["DateEvent"] = "DateEvent"
    kind: str  # effective, initial_term_end, current_term_end, non_renewal_deadline, next_renewal, survival_end
    date: Optional[str] = None
    derivation: str
    input_span_ids: list[str] = Field(default_factory=list)
    passed: Optional[bool] = None


# TODO: Obligation nodes will be added in a future layer.


# ----------------------------------------------------------------------
# Generic Node and Edge for uniform export / graph representation
# ----------------------------------------------------------------------

class Node(BaseModel):
    """Uniform graph node suitable for JSON export and Neo4j loading."""
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    """Uniform graph edge with typed source, target, and provenance."""
    id: str
    type: str  # CONTAINS, PARTY_TO, INSTANCE_OF, REFERENCES, DEFINES, IMPOSES_ON, BENEFITS, PARENT_OF, GOVERNED_BY, HAS_DATE, AMENDS, SUPERSEDES
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: str = "text"
    evidence_span_ids: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    """A flagged issue that requires human attention; never silently dropped."""
    item_id: str
    doc_id: str
    kind: str  # unresolvable_reference, unlinked_parent, ambiguous_parent, ambiguous_amendment, etc.
    description: str
    span_id: Optional[str] = None
    severity: Literal["info", "warning", "error"] = "warning"


class EffectiveClauseEntry(BaseModel):
    """One entry in the materialized effective view of a contract."""
    section_path: str
    effective_clause_id: str
    source_doc: str
    effective_from: str
    superseded_clause_ids: list[str] = Field(default_factory=list)


class EffectiveView(BaseModel):
    """The materialized view of what clauses are in force for a contract as of a date."""
    doc_id: str
    as_of: str
    clauses: list[EffectiveClauseEntry] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    """Complete in-memory knowledge graph representation."""
    as_of: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)

    def node_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.label] = counts.get(n.label, 0) + 1
        return dict(sorted(counts.items()))

    def edge_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.edges:
            counts[e.type] = counts.get(e.type, 0) + 1
        return dict(sorted(counts.items()))


def compute_text_hash(text: str) -> str:
    """Deterministic 12-char SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
