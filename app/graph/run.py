"""CLI runner for Layer 4/5b Knowledge Graph (app/graph/run.py).

Usage:
  python -m app.graph.run --data-dir data --as-of 2026-10-19
  python -m app.graph.run --data-dir data --neo4j --namespace main --rebuild
  python -m app.graph.run --data-dir data --neo4j --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from .build import build_graph
from .neo4j_store import Neo4jStore, is_neo4j_available


def main():
    parser = argparse.ArgumentParser(
        description="Build contract knowledge graph, resolve effective state, derive dates"
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument(
        "--as-of", default="2026-10-19", help="Evaluation date for effective state (YYYY-MM-DD)"
    )
    parser.add_argument("--neo4j", action="store_true", help="Load graph into Neo4j database")
    parser.add_argument(
        "--namespace", default="main",
        help="Neo4j namespace (default: main). Use a different name for tests."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete only nodes in this namespace before loading"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Reset namespace and perform full reload (same as --reset + --neo4j)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    as_of = args.as_of

    print(f"\n{'='*65}")
    print(f"  CONTRACT KNOWLEDGE GRAPH BUILDER (as_of: {as_of})")
    print(f"{'='*65}\n")

    # 1. Build In-Memory Graph and Export JSON
    graph = build_graph(data_dir=data_dir, as_of=as_of)

    # 2. Print Node and Edge Counts
    print("GRAPH SUMMARY")
    print("  Nodes per type:")
    for label, count in graph.node_counts().items():
        print(f"    {label:<20} : {count}")
    print(f"    {'TOTAL NODES':<20} : {len(graph.nodes)}")

    print("\n  Edges per type:")
    for rel_type, count in graph.edge_counts().items():
        print(f"    {rel_type:<20} : {count}")
    print(f"    {'TOTAL EDGES':<20} : {len(graph.edges)}")

    # 3. Print Derived Dates per Contract
    print("\nDERIVED DATES PER CONTRACT")
    dates_by_contract: dict[str, list[dict]] = {}
    nodes_by_id = {n.id: n for n in graph.nodes}
    for e in graph.edges:
        if e.type == "HAS_DATE":
            cid = e.source
            d_node = nodes_by_id.get(e.target)
            if d_node:
                dates_by_contract.setdefault(cid, []).append(d_node.properties)

    for cid in sorted(dates_by_contract.keys()):
        print(f"  [{cid}]")
        dev_list = sorted(dates_by_contract[cid], key=lambda x: str(x.get("kind")))
        for dev in dev_list:
            kind = dev.get("kind")
            d_val = dev.get("date") or "(null)"
            passed = dev.get("passed")
            flag = ""
            if passed is True:
                flag = " [PASSED]"
            elif passed is False:
                flag = " [PENDING]"
            derivation = dev.get("derivation", "")
            print(f"    {kind:<22} : {d_val:<12}{flag}")
            if derivation:
                print(f"       formula: {derivation}")

    # 4. Print Effective Changes from Amendments
    print("\nEFFECTIVE CHANGES (AMENDMENTS)")
    supersedes_edges = [e for e in graph.edges if e.type == "SUPERSEDES"]
    if supersedes_edges:
        for se in supersedes_edges:
            amend_cl = nodes_by_id.get(se.source)
            orig_cl = nodes_by_id.get(se.target)
            amend_props = amend_cl.properties if amend_cl else {}
            orig_props = orig_cl.properties if orig_cl else {}
            path = orig_props.get("section_path", "")
            print(
                f"  Section {path}: replaced by {se.source} (superseded {se.target})"
            )
            print(f"    New attributes: {amend_props.get('attributes')}")
    else:
        print("  (no superseded clauses as of this date)")

    # 5. Print Review Items grouped by severity
    errors   = [r for r in graph.review_items if r.severity == "error"]
    warnings = [r for r in graph.review_items if r.severity == "warning"]
    infos    = [r for r in graph.review_items if r.severity == "info"]

    print(f"\nREVIEW ITEMS ({len(graph.review_items)})  "
          f"[errors={len(errors)}, warnings={len(warnings)}, info={len(infos)}]")

    for group_label, group in [("ERROR", errors), ("WARNING", warnings), ("INFO", infos)]:
        if not group:
            continue
        print(f"\n  [{group_label}]")
        for r in group:
            span_str = f" [span: {r.span_id}]" if r.span_id else ""
            print(f"    ! [{r.kind}] ({r.doc_id}){span_str}: {r.description}")

    if not graph.review_items:
        print("  (no review items generated)")

    print(f"\nSaved graph export -> {data_dir}/graph/graph.json")
    print(f"Saved effective views -> {data_dir}/graph/effective/")

    # 6. Optional Neo4j Loading
    load_neo4j = args.neo4j or args.rebuild
    if load_neo4j:
        print("\nNEO4J DATABASE LOADING")
        if not is_neo4j_available():
            print("  WARNING: Neo4j server is not reachable. Skipping Neo4j load.")
            print("  Check that Neo4j is running (e.g. docker-compose up -d).")
            print("  Or run:  python -m app.graph.neo4j_check")
        else:
            ns = args.namespace
            do_reset = args.reset or args.rebuild
            print(f"  Namespace : {ns}")
            print(f"  Reset     : {do_reset}")

            with Neo4jStore(namespace=ns) as store:
                try:
                    stats = store.load_graph(graph, reset=do_reset)
                    print("\n  NEO4J NODE COUNTS (verified):")
                    for label, cnt in sorted(stats["node_counts"].items()):
                        print(f"    {label:<20} : {cnt}")
                    print("\n  NEO4J EDGE COUNTS (verified):")
                    for rel_type, cnt in sorted(stats["edge_counts"].items()):
                        print(f"    {rel_type:<20} : {cnt}")
                except ValueError as exc:
                    print(f"\n  [FAIL] {exc}", file=sys.stderr)
                    sys.exit(1)

    print()


if __name__ == "__main__":
    main()
