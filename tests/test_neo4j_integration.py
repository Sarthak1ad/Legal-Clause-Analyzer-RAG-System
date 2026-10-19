"""Integration tests for Neo4j database storage (Layer 4 Part E).

Auto-skipped if Neo4j server is not reachable.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from app.graph.build import build_graph
from app.graph.neo4j_store import Neo4jStore, is_neo4j_available
from app.graph.query import run_query, QUERY_FILES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def require_neo4j():
    """Skip test module if Neo4j is not available."""
    if not is_neo4j_available():
        pytest.skip("Neo4j database is offline; skipping integration tests.")


def test_neo4j_loading_idempotent():
    """Loading graph twice produces identical node and relationship counts."""
    graph = build_graph(DATA_DIR, as_of="2026-10-19")
    store = Neo4jStore()
    try:
        store.reset_database()
        store.load_graph(graph)

        # Count nodes and relationships
        c1 = store.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        r1 = store.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

        # Load second time
        store.load_graph(graph)
        c2 = store.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        r2 = store.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

        assert c1 == c2, f"Node count changed on reload: {c1} vs {c2}"
        assert r1 == r2, f"Rel count changed on reload: {r1} vs {r2}"
    finally:
        store.close()


def test_neo4j_sanity_queries():
    """All 5 sanity queries execute without error against Neo4j."""
    store = Neo4jStore()
    try:
        for q_key in sorted(QUERY_FILES.keys()):
            rows = run_query(q_key, DATA_DIR, neo4j_ok=True)
            assert isinstance(rows, list)
    finally:
        store.close()
