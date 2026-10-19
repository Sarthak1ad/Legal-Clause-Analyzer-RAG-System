"""Playbook schema models validated with Pydantic v2.

A "playbook" is our organization's negotiation standard: what we consider
preferred, acceptable, and fallback for every clause type.  These are
data-driven (YAML files), not hard-coded, so legal can update them without
touching Python.

Key design decisions:
- attribute names in each template MUST match Layer 2 attribute names exactly;
  the loader validates this against the known set of attribute names.
- Only ONE approved template per clause_type is allowed: having two would make
  the matcher ambiguous.
- better_when and favours_role drive the deviation engine later (Layer 4+),
  telling it which direction of movement counts as "worse for customer".
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# All attribute names ever produced by Layer 2 rules.  The loader validates
# every template attribute name against this set so a typo is caught immediately.
KNOWN_ATTRIBUTES: frozenset[str] = frozenset({
    # indemnification
    "indemnitor", "indemnitee", "direction", "covers_gross_negligence",
    "covered_events", "cap_multiple", "cap_basis", "cap_lookback_months",
    "cap_carve_outs",
    # limitation_of_liability
    "excludes_indirect_damages", "mutual",
    # term_renewal
    "term_years", "auto_renew", "renewal_term_years", "non_renewal_notice_days",
    "start_date", "end_date", "survival_years",
    # payment
    "payment_days", "late_interest_pct_per_month", "fee_amount", "fee_currency",
    # confidentiality  (survival_years already above, mutual already above)
    # governing_law
    "governing_law",
    # order_of_precedence
    "prevailing_document", "subordinate_document",
    # ip_ownership
    "owner",
})

AttrPreferred = Union[float, int, bool, str, list[str], None]


class TemplateAttribute(BaseModel):
    """One attribute standard inside a template.

    better_when: "higher" / "lower" / "true" / "false" / "equals"
        Tells the deviation engine which direction of change is worse.
    favours_role: "customer" / "supplier" / "neutral"
        Whose interest is served when the value is at 'preferred'.
    """
    name: str
    value_kind: Literal["number", "bool", "enum", "text"]
    preferred: AttrPreferred = None
    acceptable: AttrPreferred = None
    fallback: AttrPreferred = None
    better_when: Literal["higher", "lower", "true", "false", "equals"]
    favours_role: Literal["customer", "supplier", "neutral"]
    note: Optional[str] = None


class StandardTemplate(BaseModel):
    """One standard clause template in the playbook library."""
    template_id: str
    clause_type: str          # validated against ClauseType enum by loader
    version: str
    status: Literal["approved", "draft", "retired"]
    effective_from: str       # ISO date string, e.g. "2024-01-01"
    standard_text: str
    attributes: list[TemplateAttribute] = Field(default_factory=list)

    @field_validator("attributes")
    @classmethod
    def _attr_names_must_be_known(cls, attrs: list[TemplateAttribute]) -> list[TemplateAttribute]:
        # This is a per-object guard; the loader also validates globally
        # with a better error message.  Here we just ensure model integrity.
        for a in attrs:
            if a.name not in KNOWN_ATTRIBUTES:
                raise ValueError(
                    f"Unknown attribute name {a.name!r}.  "
                    f"Must be one of: {sorted(KNOWN_ATTRIBUTES)}"
                )
        return attrs


class DealRecord(BaseModel):
    """One historical negotiated deal, used for precedent lookups."""
    deal_id: str
    counterparty: str
    clause_type: str
    attribute: str
    our_ask: Union[float, int, str]
    final_value: Union[float, int, str]
    outcome: Literal["accepted", "conceded", "rejected"]


class PlaybookConfig(BaseModel):
    """Top-level config from playbook/config.yaml."""
    our_organization: str
    our_role: Literal["customer", "supplier"]
    default_perspective: Literal["customer", "supplier"]
    playbook_version: str


class Playbook(BaseModel):
    """The whole loaded playbook: config + templates + history."""
    config: PlaybookConfig
    templates: list[StandardTemplate]
    history: list[DealRecord] = Field(default_factory=list)

    def approved_template(self, clause_type: str) -> Optional[StandardTemplate]:
        """Return the single approved template for a clause type, or None."""
        matches = [t for t in self.templates
                   if t.clause_type == clause_type and t.status == "approved"]
        return matches[0] if matches else None
