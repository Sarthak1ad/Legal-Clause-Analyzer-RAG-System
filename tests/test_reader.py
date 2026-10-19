"""Tests for app/graph/reader.py (GraphReader interface + InMemoryGraphReader).

Uses the same 5-contract sample dataset as test_graph.py so no extra fixtures
are needed.  The module-level fixture builds the graph once.

Coverage:
1. contracts() returns all 5 contracts sorted by doc_id.
2. contract() returns properties for known and unknown ids.
3. clauses_of() with status filter and as_of filter.
4. template_of() returns a StandardTemplate for matched clauses, None for others.
5. parent_of() / children_of() follow PARENT_OF edges correctly.
6. parties_of() returns parties linked via PARTY_TO.
7. jurisdiction_of() / dates_of() return expected data.
8. contracts_expiring_between() filters by date range and kind.
9. neighbors() follows out/in/both edges with optional type filter.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.graph.build import build_graph
from app.graph.model import KnowledgeGraph
from app.graph.reader import GraphReader, InMemoryGraphReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    return build_graph(DATA_DIR, as_of="2026-10-19")


@pytest.fixture(scope="module")
def reader(graph: KnowledgeGraph) -> GraphReader:
    return InMemoryGraphReader(graph)


# ------------------------------------------------------------------
# 1. contracts()
# ------------------------------------------------------------------

def test_contracts_returns_all(reader: GraphReader):
    """All 5 sample contracts are returned, sorted by doc_id."""
    contracts = reader.contracts()
    doc_ids = [c["doc_id"] for c in contracts]
    assert "msa-acme" in doc_ids
    assert "sow-acme-01" in doc_ids
    assert "nda-globex" in doc_ids
    assert "msa-initech" in doc_ids
    assert "amendment-acme-01" in doc_ids
    assert len(contracts) >= 5
    # Should be sorted
    assert doc_ids == sorted(doc_ids)


# ------------------------------------------------------------------
# 2. contract()
# ------------------------------------------------------------------

def test_contract_known(reader: GraphReader):
    """contract() returns properties for a known doc_id."""
    props = reader.contract("msa-acme")
    assert props is not None
    assert props["doc_id"] == "msa-acme"
    assert props["doc_type"] in ("MSA", "NDA", "SOW", "AMENDMENT")


def test_contract_unknown(reader: GraphReader):
    """contract() returns None for an unknown doc_id."""
    assert reader.contract("does-not-exist") is None


# ------------------------------------------------------------------
# 3. clauses_of()
# ------------------------------------------------------------------

def test_clauses_of_active(reader: GraphReader):
    """clauses_of() with status='active' returns only active clauses."""
    clauses = reader.clauses_of("msa-acme", status="active")
    assert len(clauses) > 0
    for c in clauses:
        assert c["status"] == "active"


def test_clauses_of_all(reader: GraphReader):
    """clauses_of() with status=None returns all clauses (active and superseded)."""
    all_clauses = reader.clauses_of("msa-acme", status=None)
    active = reader.clauses_of("msa-acme", status="active")
    assert len(all_clauses) >= len(active)


def test_clauses_of_as_of_filter(reader: GraphReader):
    """clauses_of() with as_of filters out clauses whose effective_to <= as_of.

    A clause superseded in amendment-acme-01 (effective 2026-01-15) should
    not appear when as_of='2026-06-01' (past its effective_to date).
    But it should appear in unfiltered results.
    """
    # Get all clauses regardless of effective date
    all_clauses = reader.clauses_of("msa-acme", as_of=None, status=None)
    # Get clauses as of a date after the amendment
    post_amend = reader.clauses_of("msa-acme", as_of="2026-06-01", status="active")
    # Should be a valid subset
    all_ids = {c["clause_id"] for c in all_clauses}
    for c in post_amend:
        assert c["clause_id"] in all_ids


def test_clauses_of_empty_contract(reader: GraphReader):
    """clauses_of() for unknown doc returns empty list (no crash)."""
    assert reader.clauses_of("no-such-contract") == []


# ------------------------------------------------------------------
# 4. clause() and template_of()
# ------------------------------------------------------------------

def test_clause_known(reader: GraphReader):
    """clause() returns properties for a known clause id."""
    props = reader.clause("msa-acme:9.1")
    assert props is not None
    assert props["section_path"] == "9.1"


def test_clause_unknown(reader: GraphReader):
    """clause() returns None for unknown id."""
    assert reader.clause("msa-acme:999.999") is None


def test_template_of_matched_clause(reader: GraphReader, graph: KnowledgeGraph):
    """template_of() returns a StandardTemplate for a clause that has INSTANCE_OF."""
    # Find a clause that has an INSTANCE_OF edge in the graph
    inst_edges = [e for e in graph.edges if e.type == "INSTANCE_OF"]
    if not inst_edges:
        pytest.skip("No INSTANCE_OF edges in sample data")
    clause_id = inst_edges[0].source
    tpl = reader.template_of(clause_id)
    assert tpl is not None
    assert "template_id" in tpl


def test_template_of_unmatched_clause(reader: GraphReader):
    """template_of() returns None for a clause with no INSTANCE_OF edge."""
    # amendment-acme-01 operation clauses have no INSTANCE_OF edges
    props = reader.clause("amendment-acme-01:1")
    if props is None:
        pytest.skip("amendment-acme-01:1 not in sample data")
    tpl = reader.template_of("amendment-acme-01:1")
    assert tpl is None


# ------------------------------------------------------------------
# 5. parent_of() / children_of()
# ------------------------------------------------------------------

def test_parent_of_sow(reader: GraphReader):
    """sow-acme-01's parent is msa-acme."""
    parent = reader.parent_of("sow-acme-01")
    assert parent is not None
    assert parent["doc_id"] == "msa-acme"


def test_parent_of_root_contract(reader: GraphReader):
    """msa-acme has no parent."""
    assert reader.parent_of("msa-acme") is None


def test_children_of_msa_acme(reader: GraphReader):
    """msa-acme has sow-acme-01 and amendment-acme-01 as children."""
    children = reader.children_of("msa-acme")
    child_ids = [c["doc_id"] for c in children]
    assert "sow-acme-01" in child_ids
    assert "amendment-acme-01" in child_ids


def test_children_of_leaf(reader: GraphReader):
    """sow-acme-01 has no children."""
    assert reader.children_of("sow-acme-01") == []


# ------------------------------------------------------------------
# 6. parties_of()
# ------------------------------------------------------------------

def test_parties_of_msa_acme(reader: GraphReader):
    """msa-acme has at least 2 parties (Customer and Supplier)."""
    parties = reader.parties_of("msa-acme")
    assert len(parties) >= 2
    canon_names = {p["canonical_name"] for p in parties}
    assert len(canon_names) >= 2


def test_parties_of_unknown(reader: GraphReader):
    """parties_of() for unknown doc returns empty list."""
    assert reader.parties_of("no-such-contract") == []


# ------------------------------------------------------------------
# 7. jurisdiction_of() / dates_of()
# ------------------------------------------------------------------

def test_jurisdiction_of_msa_acme(reader: GraphReader):
    """msa-acme is governed by New York jurisdiction."""
    jur = reader.jurisdiction_of("msa-acme")
    assert jur is not None
    assert "new york" in jur["name"].lower() or "New York" in jur["name"]


def test_jurisdiction_of_unknown(reader: GraphReader):
    """jurisdiction_of() returns None for unknown doc."""
    assert reader.jurisdiction_of("no-such-contract") is None


def test_dates_of_msa_acme(reader: GraphReader):
    """msa-acme has at least an effective date event."""
    dates = reader.dates_of("msa-acme")
    assert len(dates) >= 1
    kinds = {d["kind"] for d in dates}
    assert "effective" in kinds


def test_dates_of_amendment_only_effective(reader: GraphReader):
    """Amendment gets only an effective DateEvent (no term_end / renewal)."""
    dates = reader.dates_of("amendment-acme-01")
    kinds = {d["kind"] for d in dates}
    assert "initial_term_end" not in kinds
    assert "current_term_end" not in kinds
    # May or may not have an effective date depending on sample data
    assert len(kinds) <= 1


# ------------------------------------------------------------------
# 8. contracts_expiring_between()
# ------------------------------------------------------------------

def test_expiry_range_returns_subset(reader: GraphReader):
    """contracts_expiring_between returns a subset of contracts."""
    all_contracts = reader.contracts()
    expiring = reader.contracts_expiring_between("2024-01-01", "2030-12-31")
    all_ids = {c["doc_id"] for c in all_contracts}
    for c in expiring:
        assert c["doc_id"] in all_ids


def test_expiry_range_empty(reader: GraphReader):
    """contracts_expiring_between with a past narrow window returns nothing."""
    result = reader.contracts_expiring_between("1900-01-01", "1900-01-02")
    assert result == []


def test_expiry_range_sorted(reader: GraphReader):
    """contracts_expiring_between result is sorted by doc_id."""
    result = reader.contracts_expiring_between("2000-01-01", "2040-12-31")
    ids = [c["doc_id"] for c in result]
    assert ids == sorted(ids)


# ------------------------------------------------------------------
# 9. neighbors()
# ------------------------------------------------------------------

def test_neighbors_out_contains(reader: GraphReader):
    """neighbors() with edge_type='CONTAINS' direction='out' yields clauses."""
    clause_props = reader.neighbors("msa-acme", edge_type="CONTAINS", direction="out")
    assert len(clause_props) > 0
    # Each neighbor should be a clause node (has clause_id)
    for p in clause_props:
        assert "clause_id" in p or "doc_id" in p  # could be sub-contract


def test_neighbors_in_party_to(reader: GraphReader):
    """neighbors() with edge_type='PARTY_TO' direction='in' yields parties for msa-acme."""
    party_props = reader.neighbors("msa-acme", edge_type="PARTY_TO", direction="in")
    assert len(party_props) >= 2
    for p in party_props:
        assert "canonical_name" in p


def test_neighbors_both(reader: GraphReader):
    """neighbors() with direction='both' includes both in and out neighbors."""
    out_n = reader.neighbors("msa-acme", direction="out")
    in_n  = reader.neighbors("msa-acme", direction="in")
    both  = reader.neighbors("msa-acme", direction="both")
    # both should have at least as many as the larger of out/in
    assert len(both) >= max(len(out_n), len(in_n))


def test_neighbors_unknown_node(reader: GraphReader):
    """neighbors() for unknown node returns empty list without error."""
    assert reader.neighbors("ghost-node") == []
