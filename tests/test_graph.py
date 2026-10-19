"""Comprehensive tests for Layer 4 Knowledge Graph (app/graph/build.py).

Tests:
1. Build is deterministic: building twice yields byte-identical graph.json.
2. Node/edge counts for the 5 sample contracts; every Clause has exactly one CONTAINS parent; every matched clause has INSTANCE_OF.
3. Reference resolution: msa-acme 5.1 -> 12.3, 10.2 -> 9.1 (from "9.1(b)"); self-reference ignored; an unresolvable reference lands in review_items.
4. PARENT_OF: sow-acme-01 and the amendment point to msa-acme; Initech MSA does not link to Acme MSA.
5. Role inheritance for the SOW (Customer = Acme Corp via parent).
6. IMPOSES_ON / BENEFITS for msa-acme 9.1: Supplier party imposed, Customer benefits.
7. Jurisdiction edges: msa-acme New York, sow-acme-01 California.
8. Evidence: every span_id stored in the graph resolves back to real text.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.graph.build import build_graph, load_parsed_docs
from app.graph.model import KnowledgeGraph

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    """Build knowledge graph once for the test module."""
    return build_graph(DATA_DIR, as_of="2026-10-19")


# 1. Determinism
def test_build_is_deterministic(tmp_path):
    """Building the graph twice produces byte-identical graph.json."""
    g1 = build_graph(DATA_DIR, as_of="2026-10-19")
    path1 = DATA_DIR / "graph" / "graph.json"
    bytes1 = path1.read_bytes()

    g2 = build_graph(DATA_DIR, as_of="2026-10-19")
    bytes2 = path1.read_bytes()

    assert bytes1 == bytes2, "Graph JSON export is not deterministic across runs"
    assert g1.model_dump() == g2.model_dump()


# 2. Node/Edge Counts & Structure
def test_graph_counts_and_structure(graph):
    """Node/edge counts for 5 contracts; every Clause has one CONTAINS parent."""
    counts = graph.node_counts()
    assert counts.get("Contract") == 5
    assert counts.get("Party") == 4
    assert counts.get("StandardTemplate") == 7
    assert counts.get("Jurisdiction") == 2
    assert counts.get("Clause", 0) >= 40

    # Every Clause has exactly one CONTAINS parent
    contains_edges = [e for e in graph.edges if e.type == "CONTAINS"]
    parent_map: dict[str, list[str]] = {}
    for e in contains_edges:
        parent_map.setdefault(e.target, []).append(e.source)

    clause_ids = [n.id for n in graph.nodes if n.label == "Clause"]
    for cid in clause_ids:
        parents = parent_map.get(cid, [])
        assert len(parents) == 1, f"Clause {cid} has {len(parents)} CONTAINS parents, expected 1"

    # Clauses with template matches have INSTANCE_OF
    instance_edges = [e for e in graph.edges if e.type == "INSTANCE_OF"]
    assert len(instance_edges) >= 25


# 3. Reference Resolution
def test_reference_resolution(graph):
    """msa-acme 5.1 -> 12.3, 10.2 -> 9.1 (from '9.1(b)'); self-reference ignored; unresolvable in review_items."""
    ref_edges = [
        (e.source, e.target)
        for e in graph.edges
        if e.type == "REFERENCES"
    ]

    # 5.1 -> 12.3
    assert ("msa-acme:5.1", "msa-acme:12.3") in ref_edges
    # 10.2 -> 9.1
    assert ("msa-acme:10.2", "msa-acme:9.1") in ref_edges

    # No self-references
    for src, tgt in ref_edges:
        assert src != tgt, f"Self-reference found on {src}"

    # Unresolvable references land in review_items
    unres_items = [r for r in graph.review_items if r.kind == "unresolvable_reference"]
    assert len(unres_items) >= 2


# 4. PARENT_OF Document Linking
def test_parent_of_linking(graph):
    """sow-acme-01 and amendment point to msa-acme; initech MSA is not linked to acme MSA."""
    parent_edges = [
        (e.source, e.target)
        for e in graph.edges
        if e.type == "PARENT_OF"
    ]

    assert ("msa-acme", "sow-acme-01") in parent_edges
    assert ("msa-acme", "amendment-acme-01") in parent_edges

    # Initech should NOT link to Acme
    for p, c in parent_edges:
        if c == "msa-initech":
            assert p != "msa-acme"
        if p == "msa-initech":
            assert "acme" not in c


# 5. Role Inheritance
def test_role_inheritance_for_sow(graph):
    """Child SOW inherits Customer=Acme Corp, Supplier=Northwind from parent."""
    sow_party_edges = [
        e for e in graph.edges
        if e.type == "PARTY_TO" and e.target == "sow-acme-01"
    ]
    assert len(sow_party_edges) == 2

    by_pid = {e.source: e for e in sow_party_edges}
    assert by_pid["party-0001"].properties.get("role") == "Customer"
    assert by_pid["party-0001"].provenance == "inherited"

    assert by_pid["party-0002"].properties.get("role") == "Supplier"
    assert by_pid["party-0002"].provenance == "inherited"


# 6. IMPOSES_ON / BENEFITS
def test_imposes_on_and_benefits(graph):
    """msa-acme 9.1: Supplier party imposed, Customer party benefits."""
    imposes = [
        (e.source, e.target)
        for e in graph.edges
        if e.type == "IMPOSES_ON" and e.source == "msa-acme:9.1"
    ]
    benefits = [
        (e.source, e.target)
        for e in graph.edges
        if e.type == "BENEFITS" and e.source == "msa-acme:9.1"
    ]

    assert len(imposes) == 1
    assert imposes[0][1] == "party-0002"  # Northwind (Supplier)

    assert len(benefits) == 1
    assert benefits[0][1] == "party-0001"  # Acme (Customer)


# 7. Jurisdiction
def test_jurisdiction_edges(graph):
    """msa-acme New York, sow-acme-01 California."""
    gov_edges = {
        e.source: e.target
        for e in graph.edges
        if e.type == "GOVERNED_BY"
    }

    assert "msa-acme" in gov_edges
    assert "new_york" in gov_edges["msa-acme"].lower()

    assert "sow-acme-01" in gov_edges
    assert "california" in gov_edges["sow-acme-01"].lower()


# 8. Evidence Span Round-Trip
def test_evidence_span_roundtrip(graph):
    """Every evidence span_id stored in the graph resolves back to real document text."""
    parsed_docs = load_parsed_docs(DATA_DIR)

    all_span_ids: set[str] = set()

    for e in graph.edges:
        for sid in e.evidence_span_ids:
            all_span_ids.add(sid)

    for n in graph.nodes:
        if n.label == "Clause":
            all_span_ids.add(n.properties["span_id"])
        elif n.label == "DefinedTerm":
            all_span_ids.add(n.properties["span_id"])
        elif n.label == "DateEvent":
            for sid in n.properties.get("input_span_ids", []):
                all_span_ids.add(sid)

    assert len(all_span_ids) > 0

    for sid in all_span_ids:
        # Format: {doc_id}:v{version}:p{page}:{path}:{start}-{end}
        parts = sid.split(":")
        doc_id = parts[0]
        offsets = parts[-1].split("-")
        start, end = int(offsets[0]), int(offsets[1])

        doc = parsed_docs[doc_id]
        quote = doc.full_text[start:end]
        assert len(quote) == (end - start), f"Span {sid} length mismatch"
        assert len(quote) > 0, f"Span {sid} has empty quote"
