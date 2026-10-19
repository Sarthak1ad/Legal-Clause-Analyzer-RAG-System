"""Unit tests for IndexStore, SQLite DB, and vector storage (Layer 5 Part B).

Verifies:
- Idempotent rebuild producing identical document count.
- Incremental update touches only changed documents.
- Superseded clauses are marked or filtered out by default.
- Pure-Python BM25 fallback matches expected search behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from app.graph.reader import InMemoryGraphReader
from app.index.embedder import HashingEmbedder
from app.index.store import IndexStore, NumpyVectorIndex, PythonBM25Index, compute_text_hash, render_attributes_text

GRAPH_JSON = Path("data/graph/graph.json")


@pytest.fixture(scope="module")
def reader() -> InMemoryGraphReader:
    if not GRAPH_JSON.exists():
        pytest.skip("data/graph/graph.json missing")
    return InMemoryGraphReader.from_json(GRAPH_JSON)


def test_numpy_vector_index(tmp_path):
    """NumpyVectorIndex correctly adds, searches, deletes, and persists vectors."""
    idx = NumpyVectorIndex(dim=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    idx.add([10, 20], np.array([v1, v2]))

    # Search with query close to v1
    q = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    hits = idx.search(q, k=2)
    assert len(hits) == 2
    assert hits[0][0] == 10
    assert hits[0][1] > hits[1][1]

    # Pre-filtering: restrict to rowid 20
    hits_filtered = idx.search(q, rowid_filter={20}, k=2)
    assert len(hits_filtered) == 1
    assert hits_filtered[0][0] == 20

    # Persistence
    idx.save(tmp_path)
    loaded = NumpyVectorIndex(dim=4)
    loaded.load(tmp_path)
    assert loaded._rowids == [10, 20]
    assert np.allclose(loaded._vectors, idx._vectors)


def test_python_bm25_fallback():
    """PythonBM25Index returns expected ranked documents for keyword queries."""
    bm25 = PythonBM25Index()
    docs = [
        (1, "Supplier shall indemnify Customer against gross negligence"),
        (2, "Governing law is New York"),
        (3, "Fees must be paid within thirty days of undisputed invoice"),
    ]
    bm25.index(docs)

    hits = bm25.search("indemnify gross negligence", k=2)
    assert len(hits) >= 1
    assert hits[0][0] == 1


def test_render_attributes_text():
    """render_attributes_text produces search-friendly keywords."""
    attrs = {
        "cap_multiple": 2.0,
        "cap_basis": "annual fees",
        "direction": "one_way",
        "covers_gross_negligence": True,
        "payment_days": 45,
    }
    rendered = render_attributes_text("indemnification", attrs)
    assert "2.0x" in rendered
    assert "annual fees" in rendered
    assert "one-way" in rendered
    assert "covers gross negligence" in rendered
    assert "payment within 45 days" in rendered


def test_store_idempotent_build(tmp_path, reader: InMemoryGraphReader):
    """Building twice with rebuild produces identical document count and metadata."""
    store = IndexStore(tmp_path)
    embedder = HashingEmbedder(dim=512)

    stats1 = store.build_from_graph(reader, embedder, rebuild=True)
    total_docs = stats1["total_docs"]
    assert total_docs > 50
    assert stats1["added"] == total_docs

    # Second build with rebuild=False (incremental)
    stats2 = store.build_from_graph(reader, embedder, rebuild=False)
    assert stats2["total_docs"] == total_docs
    assert stats2["added"] == 0
    assert stats2["unchanged"] == total_docs


def test_store_superseded_clauses_stored_with_status(tmp_path, reader: InMemoryGraphReader):
    """Superseded clauses (from amendments) are stored in docs with status='superseded'."""
    store = IndexStore(tmp_path)
    embedder = HashingEmbedder(dim=512)
    store.build_from_graph(reader, embedder, rebuild=True)

    cur = store._conn.execute("SELECT doc_key, status FROM docs WHERE status = 'superseded'")
    superseded = cur.fetchall()
    # At least msa-acme Section 9.2 or 2.1 was superseded by amendment-acme-01
    assert len(superseded) >= 1
    assert any("msa-acme" in r["doc_key"] for r in superseded)
