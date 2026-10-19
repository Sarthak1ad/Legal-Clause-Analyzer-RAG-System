"""Tests for Layer 4 Part C: Amendments and effective-state resolution."""
from __future__ import annotations

from datetime import date
from app.graph.effective import AmendmentOperation, resolve_effective_clauses
from app.graph.model import ClauseNode
from app.extraction.schemas import ClauseType, ExtractedClause, Attribute
from app.models import make_span


def _make_dummy_clause(doc_id: str, section_path: str, attrs: dict) -> ClauseNode:
    return ClauseNode(
        id=f"{doc_id}:{section_path}",
        clause_id=f"{doc_id}:{section_path}",
        section_path=section_path,
        heading="Section Heading",
        clause_type="indemnification",
        span_id=f"{doc_id}:v1:p1:{section_path}:100-200",
        text_hash="abc123456789",
        attributes=attrs,
        needs_review=False,
        status="active",
    )


def test_effective_clauses_before_and_after_amendment():
    """Effective clauses before amendment (2026-02-01) vs after amendment (2026-03-01)."""
    orig_clauses = [
        _make_dummy_clause("msa-acme", "9.2", {"cap_multiple": 1.0}),
        _make_dummy_clause("msa-acme", "2.1", {"term_years": 3}),
    ]

    span_92 = make_span("amend-01", 1, 1, "9.2", 100, 200)
    span_21 = make_span("amend-01", 1, 1, "2.1", 200, 300)

    ext_92 = ExtractedClause(
        clause_id="amend:9.2",
        section_path="9.2",
        heading="Cap on Indemnity",
        span=span_92,
        clause_type=ClauseType.indemnification,
        type_by_rule=ClauseType.indemnification,
        confidence=0.9,
        attributes=[
            Attribute(name="cap_multiple", value=2.0),
            Attribute(name="cap_basis", value="annual fees"),
            Attribute(name="cap_lookback_months", value=12),
        ],
    )
    ext_21 = ExtractedClause(
        clause_id="amend:2.1",
        section_path="2.1",
        heading="Term",
        span=span_21,
        clause_type=ClauseType.term_renewal,
        type_by_rule=ClauseType.term_renewal,
        confidence=0.9,
        attributes=[Attribute(name="term_years", value=5)],
    )

    amend_ops = [
        AmendmentOperation(
            target_path="9.2",
            replacement_text="Cap on indemnity 2x",
            heading="Cap on Indemnity",
            extracted_clause=ext_92,
            span_id=span_92.span_id,
            source_doc_id="amend-01",
            effective_date=date(2026, 3, 1),
        ),
        AmendmentOperation(
            target_path="2.1",
            replacement_text="Term 5 years",
            heading="Term",
            extracted_clause=ext_21,
            span_id=span_21.span_id,
            source_doc_id="amend-01",
            effective_date=date(2026, 3, 1),
        ),
    ]

    # Before amendment: 2026-02-01
    view_before, eff_before, rev_before = resolve_effective_clauses(
        "msa-acme", orig_clauses, amend_ops, as_of="2026-02-01"
    )
    assert eff_before["9.2"].attributes["cap_multiple"] == 1.0
    assert eff_before["2.1"].attributes["term_years"] == 3
    assert len(rev_before) == 0

    # After amendment: 2026-03-01
    view_after, eff_after, rev_after = resolve_effective_clauses(
        "msa-acme", orig_clauses, amend_ops, as_of="2026-03-01"
    )
    assert eff_after["9.2"].attributes["cap_multiple"] == 2.0
    assert eff_after["2.1"].attributes["term_years"] == 5
    assert len(rev_after) == 0


def test_ambiguity_conflicting_same_date_amendments():
    """Two amendments with identical effective date targeting the same section triggers review."""
    orig_clauses = [_make_dummy_clause("msa-acme", "9.2", {"cap_multiple": 1.0})]
    span = make_span("amend-A", 1, 1, "9.2", 100, 200)

    ext = ExtractedClause(
        clause_id="amend:9.2",
        section_path="9.2",
        span=span,
        clause_type=ClauseType.indemnification,
        type_by_rule=ClauseType.indemnification,
        confidence=0.9,
        attributes=[Attribute(name="cap_multiple", value=2.0)],
    )

    amend_ops = [
        AmendmentOperation("9.2", "text A", None, ext, span.span_id, "amend-A", date(2026, 3, 1)),
        AmendmentOperation("9.2", "text B", None, ext, span.span_id, "amend-B", date(2026, 3, 1)),
    ]

    view, eff_map, reviews = resolve_effective_clauses(
        "msa-acme", orig_clauses, amend_ops, as_of="2026-10-19"
    )

    # Should flag ambiguous amendment review and keep original without guessing
    assert any(r.kind == "ambiguous_amendment" for r in reviews)
    assert eff_map["9.2"].attributes["cap_multiple"] == 1.0


def test_amendment_missing_effective_date_triggers_review():
    """Amendment operation with None effective_date is flagged for review."""
    orig_clauses = [_make_dummy_clause("msa-acme", "9.2", {"cap_multiple": 1.0})]
    span = make_span("amend-nodate", 1, 1, "9.2", 100, 200)
    ext = ExtractedClause(
        clause_id="amend:9.2",
        section_path="9.2",
        span=span,
        clause_type=ClauseType.indemnification,
        type_by_rule=ClauseType.indemnification,
        confidence=0.9,
    )
    amend_ops = [
        AmendmentOperation("9.2", "text", None, ext, span.span_id, "amend-nodate", None)
    ]

    view, eff_map, reviews = resolve_effective_clauses(
        "msa-acme", orig_clauses, amend_ops, as_of="2026-10-19"
    )
    assert any(r.kind == "amendment_missing_effective_date" for r in reviews)


def test_amendment_target_not_in_parent_triggers_review():
    """Amendment targeting non-existent section triggers review."""
    orig_clauses = [_make_dummy_clause("msa-acme", "9.2", {"cap_multiple": 1.0})]
    span = make_span("amend-badtarget", 1, 1, "99.9", 100, 200)
    ext = ExtractedClause(
        clause_id="amend:99.9",
        section_path="99.9",
        span=span,
        clause_type=ClauseType.indemnification,
        type_by_rule=ClauseType.indemnification,
        confidence=0.9,
    )
    amend_ops = [
        AmendmentOperation("99.9", "text", None, ext, span.span_id, "amend-badtarget", date(2026, 3, 1))
    ]

    view, eff_map, reviews = resolve_effective_clauses(
        "msa-acme", orig_clauses, amend_ops, as_of="2026-10-19"
    )
    assert any(r.kind == "amendment_target_not_found" for r in reviews)
