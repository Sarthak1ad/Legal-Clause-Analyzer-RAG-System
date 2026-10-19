"""Retrieval evaluation framework (app/index/eval.py).

Computes Recall@5 and MRR (Mean Reciprocal Rank) for:
- Keyword (FTS5 BM25)
- Vector (Cosine similarity)
- Hybrid (RRF fusion)

Usage:
    python -m app.index.eval [--eval-file tests/data/retrieval_eval.json] [--embedder hashing|st]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .embedder import Embedder, get_embedder
from .search import HybridSearcher
from .store import IndexStore, DEFAULT_INDEX_DIR

DEFAULT_EVAL_FILE = Path("tests/data/retrieval_eval.json")


def run_evaluation(
    searcher: HybridSearcher,
    queries: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Evaluate keyword, vector, and hybrid modes on test queries.

    Returns:
        dict: {mode -> {"recall@5": float, "mrr": float}}
    """
    modes = ["keyword", "vector", "hybrid"]
    metrics: dict[str, dict[str, float]] = {
        m: {"recall@5": 0.0, "mrr": 0.0} for m in modes
    }

    n = len(queries)
    if n == 0:
        return metrics

    for item in queries:
        q = item["query"]
        expected = set(item.get("expected_clause_ids", []))
        if not expected:
            continue

        for m in modes:
            hits = searcher.search(query=q, k=k, mode=m)
            # 1. Recall@5 (did at least one relevant document land in top-5?)
            hit_in_top_k = any(h.clause_id in expected for h in hits[:k])
            if hit_in_top_k:
                metrics[m]["recall@5"] += 1.0 / n

            # 2. MRR (reciprocal rank of the FIRST relevant document)
            first_rank = None
            for idx, h in enumerate(hits, start=1):
                if h.clause_id in expected:
                    first_rank = idx
                    break
            if first_rank is not None:
                metrics[m]["mrr"] += (1.0 / first_rank) / n

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Layer 5 retrieval modes")
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE), help="Path to retrieval eval dataset")
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "st"], help="Embedder to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank for recall")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"[ERROR] Eval dataset not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    queries = json.loads(eval_path.read_text(encoding="utf-8"))
    store = IndexStore(DEFAULT_INDEX_DIR)
    embedder = get_embedder(args.embedder)
    searcher = HybridSearcher(store, embedder)

    print(f"\nRETRIEVAL EVALUATION ({len(queries)} queries, embedder: {embedder.name})")
    print(f"Eval dataset: {eval_path}\n")

    results = run_evaluation(searcher, queries, k=args.k)

    print(f"  {'Mode':<12} {'Recall@5':<12} {'MRR':<12}")
    print("  " + "-" * 36)
    for m in ["keyword", "vector", "hybrid"]:
        rec = results[m]["recall@5"]
        mrr = results[m]["mrr"]
        print(f"  {m:<12} {rec*100:>8.1f}%   {mrr:>8.3f}")
    print()


if __name__ == "__main__":
    main()
