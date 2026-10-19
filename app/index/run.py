"""Command-line interface for Layer 5 Indexes (app/index/run.py).

Usage:
  python -m app.index.run build [--rebuild] [--embedder hashing|st] [--no-narrative]
  python -m app.index.run search "cap on supplier liability" --k 5 --mode hybrid
        [--clause-type indemnification] [--as-of 2026-10-19] [--groups all]
  python -m app.index.run communities
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..graph.reader import InMemoryGraphReader
from ..llm.client import NullLLM
from .communities import CommunityBuilder, COMMUNITIES_PATH
from .embedder import get_embedder
from .search import HybridSearcher
from .store import IndexStore, DEFAULT_INDEX_DIR

GRAPH_JSON = Path("data/graph/graph.json")


def cmd_build(args: argparse.Namespace) -> None:
    """Build or rebuild vector, keyword, and community indexes."""
    if not GRAPH_JSON.exists():
        print(f"[ERROR] Knowledge graph not found at {GRAPH_JSON}. Run pipeline first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  BUILDING LAYER 5 INDEXES (embedder: {args.embedder}, rebuild={args.rebuild})")
    print(f"{'='*65}\n")

    reader = InMemoryGraphReader.from_json(GRAPH_JSON)
    embedder = get_embedder(args.embedder)
    store = IndexStore(DEFAULT_INDEX_DIR)

    # 1. Build community summaries first
    print("[1/2] Detecting and summarizing portfolio communities ...")
    llm = NullLLM()
    if not args.no_narrative:
        try:
            from ..llm.client import OllamaClient  # type: ignore
            llm = OllamaClient()
        except Exception:
            llm = NullLLM()

    comm_builder = CommunityBuilder(reader=reader, llm=llm)
    comms = comm_builder.save_and_index(COMMUNITIES_PATH)
    print(f"      Built {len(comms)} communities -> {COMMUNITIES_PATH}")

    # 2. Build index store
    print(f"[2/2] Indexing clauses, templates, and summaries with {embedder.name} ...")
    stats = store.build_from_graph(
        reader=reader,
        embedder=embedder,
        rebuild=args.rebuild,
        communities_file=COMMUNITIES_PATH,
    )

    print("\nINDEX BUILD SUMMARY")
    print(f"  Total docs in index : {stats['total_docs']}")
    print(f"  Newly added docs    : {stats['added']}")
    print(f"  Updated docs        : {stats['updated']}")
    print(f"  Unchanged docs      : {stats['unchanged']}")
    print(f"  Database path       : {store.db_path}")
    print(f"  Embedder used       : {embedder.name} (dim={embedder.dim})")
    print("Index build complete.\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the hybrid index with optional filters and score breakdown."""
    store = IndexStore(DEFAULT_INDEX_DIR)
    embedder_name = store.get_meta("embedder_name") or "hashing"
    embedder_dim = int(store.get_meta("embedder_dim") or "512")
    embedder = get_embedder("hashing" if "hashing" in embedder_name else "st", dim=embedder_dim)

    searcher = HybridSearcher(store, embedder)

    filters: dict[str, Any] = {}
    if args.clause_type:
        filters["clause_type"] = args.clause_type
    if args.doc_type:
        filters["doc_type"] = args.doc_type
    if args.contract_id:
        filters["contract_id"] = args.contract_id
    if args.kind:
        filters["kind"] = args.kind

    user_groups = [g.strip() for g in args.groups.split(",")] if args.groups else ["all"]

    hits = searcher.search(
        query=args.query,
        k=args.k,
        mode=args.mode,
        filters=filters,
        as_of=args.as_of,
        user_groups=user_groups,
    )

    print(f"\nSEARCH RESULTS (mode={args.mode}, k={args.k}, results={len(hits)})")
    print(f"Query: '{args.query}'\n")

    if not hits:
        print("  (no matching documents found)")
        return

    # Header
    print(f"  {'#':<3} {'Score':<8} {'V-Rnk':<6} {'K-Rnk':<6} {'Kind':<9} {'Contract / Clause':<26} {'Heading / Snippet'}")
    print("  " + "-" * 85)

    for i, h in enumerate(hits, start=1):
        v_rank = str(h.vector_rank) if h.vector_rank is not None else "-"
        k_rank = str(h.keyword_rank) if h.keyword_rank is not None else "-"
        ref = f"{h.contract_id or 'global'}:{h.clause_id}" if h.contract_id else h.clause_id
        ref = ref[:25]
        score_str = f"{h.rrf_score:.4f}"
        print(f"  {i:<3} {score_str:<8} {v_rank:<6} {k_rank:<6} {h.kind:<9} {ref:<26} {h.snippet[:65]}")


def cmd_communities(args: argparse.Namespace) -> None:
    """List portfolio communities and key stats."""
    if not COMMUNITIES_PATH.exists():
        print(f"[ERROR] Communities not found at {COMMUNITIES_PATH}. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
    comms = data.get("communities", [])

    print(f"\nPORTFOLIO COMMUNITIES ({len(comms)})")
    print(f"  Source file: {COMMUNITIES_PATH}\n")

    print(f"  {'ID':<26} {'Kind':<8} {'Group':<6} {'Contracts':<10} {'Clauses':<8} {'Key Summary / Stats'}")
    print("  " + "-" * 90)

    for c in comms:
        cid = c["id"][:25]
        kind = c.get("kind", "")
        group = ",".join(c.get("permission_groups", ["all"]))[:6]
        c_count = len(c.get("member_contract_ids", []))
        cl_count = len(c.get("member_clause_ids", []))
        narr = c.get("narrative", "")[:55]
        print(f"  {cid:<26} {kind:<8} {group:<6} {c_count:<10} {cl_count:<8} {narr}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 5 Index and Search CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build or rebuild indexes")
    p_build.add_argument("--rebuild", action="store_true", help="Force complete rebuild")
    p_build.add_argument("--embedder", default="hashing", choices=["hashing", "st"], help="Embedder backend")
    p_build.add_argument("--no-narrative", action="store_true", help="Skip LLM narrative generation")

    # search
    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument("--k", type=int, default=5, help="Number of results to return")
    p_search.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"], help="Search mode")
    p_search.add_argument("--clause-type", help="Filter by clause_type")
    p_search.add_argument("--doc-type", help="Filter by doc_type (e.g. MSA, SOW)")
    p_search.add_argument("--contract-id", help="Filter by contract_id")
    p_search.add_argument("--kind", help="Filter by kind (clause, template, summary)")
    p_search.add_argument("--as-of", default="2026-10-19", help="Evaluate effective state as of date")
    p_search.add_argument("--groups", default="all", help="User ACL groups (comma-separated)")

    # communities
    subparsers.add_parser("communities", help="List portfolio communities and summary stats")

    args = parser.parse_args()

    if args.subcommand == "build":
        cmd_build(args)
    elif args.subcommand == "search":
        cmd_search(args)
    elif args.subcommand == "communities":
        cmd_communities(args)


if __name__ == "__main__":
    main()
