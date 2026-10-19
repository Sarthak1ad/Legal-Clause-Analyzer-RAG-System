"""Unit tests for Embedder interface and implementations (Layer 5 Part A).

Verifies:
- Deterministic output across multiple calls.
- L2-normalized vector output.
- Automatic fallback when sentence-transformers is missing or fails.
- IndexStore refuses an embedder that does not match stored metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.index.embedder import HashingEmbedder, SentenceTransformerEmbedder, get_embedder
from app.index.store import IndexStore


def test_hashing_embedder_deterministic():
    """HashingEmbedder produces byte-identical vectors for the same input across instances."""
    e1 = HashingEmbedder(dim=512)
    e2 = HashingEmbedder(dim=512)

    texts = [
        "Supplier shall defend and indemnify Customer against third-party claims.",
        "Either party may terminate this Agreement upon ninety (90) days notice.",
    ]

    v1 = e1.embed(texts)
    v2 = e2.embed(texts)

    assert v1.shape == (2, 512)
    assert np.allclose(v1, v2, atol=1e-7)


def test_hashing_embedder_l2_normalized():
    """All non-empty embedded texts have L2 norm of exactly 1.0."""
    embedder = HashingEmbedder(dim=512)
    texts = [
        "Governing law shall be New York.",
        "Payment within thirty (30) days.",
    ]
    vectors = embedder.embed(texts)
    for v in vectors:
        norm = np.linalg.norm(v)
        assert norm == pytest.approx(1.0, rel=1e-5)


def test_hashing_embedder_empty_input():
    """Empty list produces (0, dim) array; empty string produces zero vector."""
    embedder = HashingEmbedder(dim=256)
    v_empty = embedder.embed([])
    assert v_empty.shape == (0, 256)

    v_blank = embedder.embed([""])
    assert v_blank.shape == (1, 256)
    assert np.all(v_blank[0] == 0.0)


def test_sentence_transformer_fallback():
    """SentenceTransformerEmbedder falls back gracefully to HashingEmbedder when model missing."""
    # Attempt to load a nonexistent local model
    st = SentenceTransformerEmbedder(model_name="nonexistent/fake-model-xyz")
    assert st.dim == 512
    assert "hashing" in st.name or "fallback" in st.name
    vecs = st.embed(["test clause text"])
    assert vecs.shape == (1, 512)
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, rel=1e-5)


def test_index_store_refuses_mismatched_embedder(tmp_path):
    """IndexStore raises ValueError if queried or updated with an embedder different from build."""
    store = IndexStore(tmp_path)
    store.set_meta("embedder_name", "hashing-512")
    store.set_meta("embedder_dim", "512")

    wrong_embedder = HashingEmbedder(dim=256, name="hashing-256")
    with pytest.raises(ValueError, match="Embedder mismatch"):
        store.verify_embedder(wrong_embedder)

    correct_embedder = HashingEmbedder(dim=512, name="hashing-512")
    # Should not raise
    store.verify_embedder(correct_embedder)
