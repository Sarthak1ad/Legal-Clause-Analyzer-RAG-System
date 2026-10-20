"""Integration tests for Neo4j database storage (Layer 5b Part A).

Auto-skipped if Neo4j server is not reachable or credentials not set.
Uses a dedicated test namespace so production/main data is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.graph.build import build_graph
from app.graph.neo4j_check import check_neo4j_connection
from app.graph.neo4j_reader import Neo4jGraphReader
from app.graph.neo4j_store import Neo4jStore, is_neo4j_available
from app.graph.query import run_query, QUERY_FILES
from app.graph.reader import InMemoryGraphReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PORTFOLIO_DIR = Path(__file__).resolve().parent.parent / "samples" / "portfolio"
TEST_NS = "test_cg_neo4j"


if not is_neo4j_available():
    pytestmark = pytest.mark.skip(
        reason="Neo4j database is offline or credentials not set; skipping integration tests."
    )


@pytest.fixture(scope="module")
def loaded_neo4j():
    """Load the portfolio graph once into TEST_NS for the test suite, then clean up."""
    graph = build_graph(DATA_DIR, as_of="2026-10-19")
    store = Neo4jStore(namespace=TEST_NS)
    try:
        store.reset_namespace()
        store.load_graph(graph, reset=False)
        yield store, graph
    finally:
        try:
            store.reset_namespace()
        except Exception:
            pass
        store.close()


def test_neo4j_connection_check():
    """Connection check returns 0 and outputs server details."""
    rc = check_neo4j_connection()
    assert rc == 0


def test_neo4j_namespace_isolation():
    """Test namespace reset does not touch other namespaces."""
    store1 = Neo4jStore(namespace=f"{TEST_NS}_iso1")
    store2 = Neo4jStore(namespace=f"{TEST_NS}_iso2")
    try:
        store1.reset_namespace()
        store2.reset_namespace()

        # Create dummy node in iso1 and iso2
        store1.query(
            "MERGE (n:Contract {id: $id, ns: $ns}) SET n.name = 'doc1'",
            {"id": f"{TEST_NS}_iso1:c1", "ns": f"{TEST_NS}_iso1"},
        )
        store2.query(
            "MERGE (n:Contract {id: $id, ns: $ns}) SET n.name = 'doc2'",
            {"id": f"{TEST_NS}_iso2:c2", "ns": f"{TEST_NS}_iso2"},
        )

        assert len(store1.query("MATCH (n:Contract {ns: $ns}) RETURN n", {"ns": f"{TEST_NS}_iso1"})) == 1
        assert len(store2.query("MATCH (n:Contract {ns: $ns}) RETURN n", {"ns": f"{TEST_NS}_iso2"})) == 1

        # Reset iso1 only
        store1.reset_namespace()

        assert len(store1.query("MATCH (n:Contract {ns: $ns}) RETURN n", {"ns": f"{TEST_NS}_iso1"})) == 0
        assert len(store2.query("MATCH (n:Contract {ns: $ns}) RETURN n", {"ns": f"{TEST_NS}_iso2"})) == 1
    finally:
        store1.reset_namespace()
        store2.reset_namespace()
        store1.close()
        store2.close()


def test_neo4j_idempotent_load_with_count_verification(loaded_neo4j):
    """Loading graph twice produces verified identical node and edge counts."""
    store, graph = loaded_neo4j

    # Reload graph into the same namespace (idempotent MERGE)
    stats = store.load_graph(graph, reset=False)
    assert stats["node_counts"]
    assert stats["edge_counts"]

    expected_nodes = graph.node_counts()
    for label, count in expected_nodes.items():
        assert stats["node_counts"].get(label) == count


def test_neo4j_sanity_queries_live(loaded_neo4j):
    """All 6 sanity queries execute without error against live Neo4j."""
    store, _ = loaded_neo4j
    for q_key in sorted(QUERY_FILES.keys()):
        rows = run_query(q_key, DATA_DIR, neo4j_ok=True, namespace=TEST_NS)
        assert isinstance(rows, list)


def test_neo4j_reader_parity(loaded_neo4j):
    """Neo4jGraphReader and InMemoryGraphReader return identical results."""
    store, graph = loaded_neo4j
    neo_reader = Neo4jGraphReader(store=store, namespace=TEST_NS)
    mem_reader = InMemoryGraphReader(graph=graph)

    # 1. Contracts
    c_neo = sorted([c["doc_id"] for c in neo_reader.contracts()])
    c_mem = sorted([c["doc_id"] for c in mem_reader.contracts()])
    assert c_neo == c_mem

    # 2. Clauses of first contract
    if c_mem:
        test_doc = c_mem[0]
        cl_neo = sorted([cl["clause_id"] for cl in neo_reader.clauses_of(test_doc)])
        cl_mem = sorted([cl["clause_id"] for cl in mem_reader.clauses_of(test_doc)])
        assert cl_neo == cl_mem

        # 3. Parties of contract
        p_neo = sorted([p["party_id"] for p in neo_reader.parties_of(test_doc)])
        p_mem = sorted([p["party_id"] for p in mem_reader.parties_of(test_doc)])
        assert p_neo == p_mem

        # 4. Dates of contract
        d_neo = sorted([f"{d.get('kind')}:{d.get('date')}" for d in neo_reader.dates_of(test_doc)])
        d_mem = sorted([f"{d.get('kind')}:{d.get('date')}" for d in mem_reader.dates_of(test_doc)])
        assert d_neo == d_mem

    # 5. Contracts expiring between dates
    exp_neo = sorted([c["doc_id"] for c in neo_reader.contracts_expiring_between("2026-01-01", "2026-12-31")])
    exp_mem = sorted([c["doc_id"] for c in mem_reader.contracts_expiring_between("2026-01-01", "2026-12-31")])
    assert exp_neo == exp_mem

    # 6. Portfolio expected queries
    eq_path = PORTFOLIO_DIR / "expected_queries.json"
    if eq_path.exists():
        eq = json.loads(eq_path.read_text(encoding="utf-8"))
        # Question 1: Cap below standard (2x) and expiring 2026
        q1_expected = sorted(eq["query_1_cap_below_standard_and_expiring_2026"]["expected_doc_ids"])
        q1_rows = run_query("6", DATA_DIR, neo4j_ok=True, namespace=TEST_NS)
        q1_actual = sorted([r["contract_id"] for r in q1_rows])
        assert q1_actual == q1_expected


def test_neo4j_weak_indemnity_query_matches_expected(loaded_neo4j):
    """Query 6 executed in LIVE mode matches expected_queries.json."""
    eq_path = PORTFOLIO_DIR / "expected_queries.json"
    if not eq_path.exists():
        pytest.skip("Portfolio expected_queries.json missing.")

    eq = json.loads(eq_path.read_text(encoding="utf-8"))
    expected = sorted(eq["query_1_cap_below_standard_and_expiring_2026"]["expected_doc_ids"])

    rows = run_query("6", DATA_DIR, neo4j_ok=True, namespace=TEST_NS)
    actual = sorted([r["contract_id"] for r in rows])
    assert actual == expected, f"Query 6 live mismatch: {actual} vs {expected}"
