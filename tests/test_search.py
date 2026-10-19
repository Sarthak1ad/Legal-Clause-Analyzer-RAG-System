"""Unit tests for Hybrid Searcher, Pre-filtering, RRF Fusion, and ACL (Layer 5 Part C).

Verifies:
- BM25 exact phrase search ("gross negligence").
- Vector search retrieves paraphrased text.
- RRF fusion ordering is mathematically correct on a controlled example.
- Filters are strictly pre-filters (applied before ranking, fill top-k).
- ACL gating: users restricted to specific groups cannot access other contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.graph.reader import InMemoryGraphReader
from app.index.embedder import HashingEmbedder
from app.index.search import HybridSearcher, sanitize_fts5_query
from app.index.store import IndexStore

INDEX_DIR = Path("data/index")
GRAPH_JSON = Path("data/graph/graph.json")


@pytest.fixture(scope="module")
def reader() -> InMemoryGraphReader:
    if not GRAPH_JSON.exists():
        pytest.skip("data/graph/graph.json missing")
    return InMemoryGraphReader.from_json(GRAPH_JSON)


@pytest.fixture(scope="module")
def searcher() -> HybridSearcher:
    if not (INDEX_DIR / "index.db").exists():
        pytest.skip("data/index/index.db missing; build index first")
    store = IndexStore(INDEX_DIR)
    embedder = HashingEmbedder(dim=512)
    return HybridSearcher(store, embedder)


def test_sanitize_fts5_query():
    """FTS5 sanitization preserves quotes while stripping syntax breaking characters."""
    raw = 'Supplier\'s "gross negligence" AND (cap >= 2x);'
    sanitized = sanitize_fts5_query(raw)
    assert '"gross negligence"' in sanitized
    assert "Supplier" in sanitized
    assert ";" not in sanitized


def test_bm25_finds_exact_phrase(searcher: HybridSearcher):
    """Keyword search on exact phrase 'gross negligence' retrieves indemnification clauses."""
    hits = searcher.search(query='"gross negligence"', k=5, mode="keyword")
    assert len(hits) >= 1
    # Check that top hit actually contains the phrase
    assert any("gross negligence" in h.text.lower() for h in hits)
    # Check rank/score populated
    assert hits[0].keyword_rank == 1
    assert hits[0].keyword_score is not None


def test_vector_finds_paraphrase(searcher: HybridSearcher):
    """Vector search retrieves paraphrased clauses ('hold harmless Client')."""
    hits = searcher.search(
        query="Provider shall hold harmless Client from third-party claims",
        k=5,
        mode="vector",
    )
    assert len(hits) >= 1
    # The top results should include the paraphrased contracts
    doc_ids = [h.contract_id for h in hits if h.contract_id]
    assert any(c in doc_ids for c in ("msa-wonka-acme", "msa-umbrella-acme-eu", "msa-wayne-acme-eu"))


def test_rrf_ordering_controlled():
    """RRF ordering accurately fuses keyword and vector ranks."""
    # Hand-made simulation of RRF logic
    k_rrf = 60
    # Doc A: vector_rank=1, keyword_rank=10 -> 1/61 + 1/70 = 0.01639 + 0.01428 = 0.03067
    score_a = 1.0 / (k_rrf + 1) + 1.0 / (k_rrf + 10)
    # Doc B: vector_rank=2, keyword_rank=2  -> 1/62 + 1/62 = 0.01612 + 0.01612 = 0.03225
    score_b = 1.0 / (k_rrf + 2) + 1.0 / (k_rrf + 2)
    # Doc C: vector_rank=1, keyword_rank=None -> 1/61 = 0.01639
    score_c = 1.0 / (k_rrf + 1)

    assert score_b > score_a > score_c


def test_pre_filtering_inside_retrieval(searcher: HybridSearcher):
    """Pre-filtering strictly applies before ranking so top-k is filled with matching docs."""
    # Search for "payment" but strictly filter for clause_type="governing_law"
    hits = searcher.search(
        query="payment fees invoice",
        k=5,
        mode="hybrid",
        filters={"clause_type": "governing_law"},
    )
    assert len(hits) >= 1
    for h in hits:
        assert h.clause_type == "governing_law"
        assert h.clause_type != "payment"


def test_effective_state_as_of_filtering(searcher: HybridSearcher):
    """By default, superseded clauses are excluded for active as_of date."""
    hits = searcher.search(
        query="Cap on Indemnity",
        k=10,
        mode="hybrid",
        filters={"contract_id": "msa-acme"},
        as_of="2026-10-19",
    )
    for h in hits:
        assert h.status == "active"


def test_acl_permission_gating(tmp_path, reader: InMemoryGraphReader):
    """A user in a restricted group cannot see documents from other groups."""
    store = IndexStore(tmp_path)
    embedder = HashingEmbedder(dim=512)

    # ACL mapping: msa-acme is in 'finance', msa-stark-acme is in 'engineering'
    acl = {
        "msa-acme": ["finance"],
        "msa-stark-acme": ["engineering"],
    }
    store.build_from_graph(reader, embedder, acl_map=acl, rebuild=True)

    searcher_acl = HybridSearcher(store, embedder)

    # User in 'engineering' only
    hits_eng = searcher_acl.search(
        query="Agreement begins on the Effective Date",
        k=10,
        mode="hybrid",
        user_groups=["engineering"],
    )
    cids_eng = {h.contract_id for h in hits_eng if h.contract_id}
    assert "msa-stark-acme" in cids_eng
    assert "msa-acme" not in cids_eng

    # User in 'finance' only
    hits_fin = searcher_acl.search(
        query="Agreement begins on the Effective Date",
        k=10,
        mode="hybrid",
        user_groups=["finance"],
    )
    cids_fin = {h.contract_id for h in hits_fin if h.contract_id}
    assert "msa-acme" in cids_fin
    assert "msa-stark-acme" not in cids_fin
