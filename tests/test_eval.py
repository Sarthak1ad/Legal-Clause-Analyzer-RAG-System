"""Regression tests for retrieval evaluation thresholds (Layer 5 Part F).

Verifies:
- Benchmark retrieval evaluation on tests/data/retrieval_eval.json.
- Hybrid mode achieves >= each single mode on recall@5.
- Metrics meet measured regression thresholds with HashingEmbedder.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.index.embedder import HashingEmbedder
from app.index.eval import run_evaluation
from app.index.search import HybridSearcher
from app.index.store import IndexStore

INDEX_DIR = Path("data/index")
EVAL_FILE = Path("tests/data/retrieval_eval.json")


@pytest.fixture(scope="module")
def searcher() -> HybridSearcher:
    if not (INDEX_DIR / "index.db").exists():
        pytest.skip("data/index/index.db missing; build index first")
    store = IndexStore(INDEX_DIR)
    embedder = HashingEmbedder(dim=512)
    return HybridSearcher(store, embedder)


@pytest.fixture(scope="module")
def eval_queries() -> list[dict]:
    if not EVAL_FILE.exists():
        pytest.skip(f"Eval file {EVAL_FILE} missing")
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def test_retrieval_eval_thresholds(searcher: HybridSearcher, eval_queries: list[dict]):
    """Ensure retrieval metrics meet empirical regression thresholds."""
    assert len(eval_queries) >= 15, "Evaluation suite must have at least 15 queries"

    results = run_evaluation(searcher, eval_queries, k=5)

    kw_rec = results["keyword"]["recall@5"]
    vec_rec = results["vector"]["recall@5"]
    hyb_rec = results["hybrid"]["recall@5"]
    hyb_mrr = results["hybrid"]["mrr"]

    # Hybrid must be >= each single mode on recall@5
    assert hyb_rec >= kw_rec, f"Hybrid recall ({hyb_rec}) < keyword recall ({kw_rec})"
    assert hyb_rec >= vec_rec, f"Hybrid recall ({hyb_rec}) < vector recall ({vec_rec})"

    # Measured values: keyword=62.5%, vector=93.8%, hybrid=100.0%, hybrid MRR=0.969
    # Thresholds set slightly below measured:
    assert kw_rec >= 0.50, f"Keyword recall below threshold: {kw_rec}"
    assert vec_rec >= 0.80, f"Vector recall below threshold: {vec_rec}"
    assert hyb_rec >= 0.95, f"Hybrid recall below threshold: {hyb_rec}"
    assert hyb_mrr >= 0.85, f"Hybrid MRR below threshold: {hyb_mrr}"
