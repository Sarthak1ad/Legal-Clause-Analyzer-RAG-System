"""Output schemas for Layer 3: entity resolution + template matching.

Every fact produced here is grounded in a specific span of the source document
(same Evidence model as Layer 2), so the citation verifier in later layers can
check every claim.
"""
from __future__ import annotations

from datetime import date as Date
from typing import Optional

from pydantic import BaseModel, Field

from ..extraction.schemas import Evidence


class PartyMention(BaseModel):
    """One party as seen in a specific document.

    raw_name       : the exact name that appeared in the text
    role_label     : 'Customer', 'Supplier', etc. from the preamble quote,
                     or None if the preamble had no role labels
    party_id       : canonical ID from the registry (e.g. 'party-0001');
                     set after entity resolution
    needs_review   : True when a fuzzy name match was ambiguous
    review_reason  : explanation if needs_review is True
    evidence       : the preamble quote + span that supports this mention
    """
    raw_name: str
    role_label: Optional[str] = None
    party_id: Optional[str] = None
    needs_review: bool = False
    review_reason: Optional[str] = None
    evidence: Evidence


class RegistryEntry(BaseModel):
    """One canonical party in the cross-document registry (data/parties.json).

    canonical_name : the shortest/most official-looking alias
    aliases        : every raw_name variant seen so far
    appearances    : where this party has appeared across the portfolio
    """
    party_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    appearances: list["AppearanceRecord"] = Field(default_factory=list)


class AppearanceRecord(BaseModel):
    """One document-version where a canonical party appeared."""
    doc_id: str
    version: int
    role_label: Optional[str] = None


class RoleMapping(BaseModel):
    """Per-document: which party_id plays which role."""
    customer_party_id: Optional[str] = None
    supplier_party_id: Optional[str] = None


class TemplateMatch(BaseModel):
    """One extracted clause linked to a standard template (or no template).

    method        : 'type_match' (deterministic) or 'none'
    score         : 1.0 for type_match, 0.0 when no template found
    embedding_sim : optional float from sentence-transformers (feature-flagged)
    needs_review  : forwarded from Layer 2 OR set here if the match itself
                    is ambiguous
    review_reasons: merged list from Layer 2 + any Layer 3 additions
    """
    clause_id: str
    section_path: str
    clause_type: str
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    playbook_version: Optional[str] = None
    method: str = "none"
    score: float = 0.0
    embedding_sim: Optional[float] = None   # only when USE_EMBEDDINGS=true
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class ResolvedDocument(BaseModel):
    """Layer 3 output: parties + roles + template matches for one doc version."""
    doc_id: str
    version: int
    parties: list[PartyMention] = Field(default_factory=list)
    role_mapping: RoleMapping = Field(default_factory=RoleMapping)
    template_matches: list[TemplateMatch] = Field(default_factory=list)


RegistryEntry.model_rebuild()
