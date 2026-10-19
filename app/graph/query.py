"""Query runner CLI for Layer 4 Knowledge Graph (app/graph/query.py).

Runs ready-made Cypher queries from queries/ against Neo4j if reachable,
or evaluates the equivalent query logic against data/graph/graph.json
with clear notification.

Usage:
    python -m app.graph.query 1
    python -m app.graph.query all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .neo4j_store import Neo4jStore, is_neo4j_available

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_DIR = ROOT / "queries"


QUERY_FILES: dict[str, str] = {
    "1": "1_initial_term_2026.cypher",
    "2": "2_clauses_with_templates.cypher",
    "3": "3_sows_with_parent_msa.cypher",
    "4": "4_governing_law_per_contract.cypher",
    "5": "5_superseded_clauses.cypher",
}

QUERY_DESCRIPTIONS: dict[str, str] = {
    "1": "Contracts whose initial_term_end falls in 2026",
    "2": "Every clause with its standard template (INSTANCE_OF)",
    "3": "SOWs and their parent MSA (PARENT_OF)",
    "4": "Governing law per contract (eyeball NY vs CA mismatch)",
    "5": "Clauses superseded by an amendment",
}


def _run_json_fallback(query_key: str, data_dir: Path) -> list[dict[str, Any]]:
    """Evaluate query logic directly against data/graph/graph.json."""
    graph_path = data_dir / "graph" / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}. Run build first.")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    if query_key == "1":
        # Contracts whose initial_term_end falls in 2026
        results = []
        for e in edges:
            if e["type"] == "HAS_DATE":
                c = nodes.get(e["source"])
                d = nodes.get(e["target"])
                if c and d and d["properties"].get("kind") == "initial_term_end":
                    date_val = str(d["properties"].get("date") or "")
                    if date_val.startswith("2026"):
                        results.append({
                            "contract_id": c["properties"].get("doc_id"),
                            "doc_type": c["properties"].get("doc_type"),
                            "initial_term_end": date_val,
                            "derivation": d["properties"].get("derivation"),
                        })
        results.sort(key=lambda r: r["contract_id"])
        return results

    elif query_key == "2":
        # Every clause with its template
        # Build clause -> contract map
        clause_contract: dict[str, str] = {}
        for e in edges:
            if e["type"] == "CONTAINS":
                clause_contract[e["target"]] = e["source"]

        results = []
        for e in edges:
            if e["type"] == "INSTANCE_OF":
                cl = nodes.get(e["source"])
                t = nodes.get(e["target"])
                if cl and t:
                    cid = cl["id"]
                    doc_id = clause_contract.get(cid, "unknown")
                    results.append({
                        "contract_id": doc_id,
                        "section": cl["properties"].get("section_path"),
                        "clause_type": cl["properties"].get("clause_type"),
                        "template_id": t["properties"].get("template_id"),
                    })
        results.sort(key=lambda r: (r["contract_id"], r["section"]))
        return results

    elif query_key == "3":
        # SOWs and their parent MSA
        results = []
        for e in edges:
            if e["type"] == "PARENT_OF":
                parent = nodes.get(e["source"])
                child = nodes.get(e["target"])
                if parent and child and child["properties"].get("doc_type") == "SOW":
                    results.append({
                        "sow_id": child["properties"].get("doc_id"),
                        "parent_msa_id": parent["properties"].get("doc_id"),
                        "parent_date": parent["properties"].get("effective_date"),
                    })
        results.sort(key=lambda r: r["sow_id"])
        return results

    elif query_key == "4":
        # Governing law per contract
        results = []
        for e in edges:
            if e["type"] == "GOVERNED_BY":
                c = nodes.get(e["source"])
                j = nodes.get(e["target"])
                if c and j:
                    results.append({
                        "contract_id": c["properties"].get("doc_id"),
                        "doc_type": c["properties"].get("doc_type"),
                        "governing_law": j["properties"].get("name"),
                    })
        results.sort(key=lambda r: r["contract_id"])
        return results

    elif query_key == "5":
        # Clauses superseded by an amendment
        clause_contract = {}
        for e in edges:
            if e["type"] == "CONTAINS":
                clause_contract[e["target"]] = e["source"]

        results = []
        for e in edges:
            if e["type"] == "SUPERSEDES":
                amend_cl = nodes.get(e["source"])
                orig_cl = nodes.get(e["target"])
                if amend_cl and orig_cl:
                    orig_doc = clause_contract.get(orig_cl["id"], "unknown")
                    amend_doc = clause_contract.get(amend_cl["id"], "unknown")
                    results.append({
                        "original_contract": orig_doc,
                        "section": orig_cl["properties"].get("section_path"),
                        "original_status": orig_cl["properties"].get("status"),
                        "amending_doc": amend_doc,
                        "replacement_clause": amend_cl["properties"].get("clause_id"),
                    })
        results.sort(key=lambda r: (r["original_contract"], r["section"]))
        return results

    return []


def format_table(rows: list[dict[str, Any]]) -> str:
    """Format list of dict rows as an aligned text table."""
    if not rows:
        return "  (no rows returned)"
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            val_str = str(r.get(h, ""))
            col_widths[h] = max(col_widths[h], len(val_str))

    lines = []
    header_line = "  " + "  ".join(f"{h:<{col_widths[h]}}" for h in headers)
    sep_line = "  " + "  ".join("-" * col_widths[h] for h in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for r in rows:
        row_str = "  " + "  ".join(f"{str(r.get(h, '')):<{col_widths[h]}}" for h in headers)
        lines.append(row_str)
    return "\n".join(lines)


def run_query(query_key: str, data_dir: Path, neo4j_ok: Optional[bool] = None) -> list[dict[str, Any]]:
    """Run specified sanity query by key ('1'-'5')."""
    if query_key not in QUERY_FILES:
        raise ValueError(f"Unknown query {query_key!r}. Valid keys: {list(QUERY_FILES.keys())}")

    q_file = QUERIES_DIR / QUERY_FILES[query_key]
    desc = QUERY_DESCRIPTIONS[query_key]

    print(f"\n[{query_key}] {desc}")
    print(f"File: {q_file.relative_to(ROOT)}")

    has_neo4j = is_neo4j_available() if neo4j_ok is None else neo4j_ok

    if has_neo4j:
        print("(Mode: Connected to Neo4j database)")
        store = Neo4jStore()
        try:
            cypher = q_file.read_text(encoding="utf-8")
            rows = store.query(cypher)
        finally:
            store.close()
    else:
        print("(Mode: Neo4j server offline; evaluated against JSON graph)")
        rows = _run_json_fallback(query_key, data_dir)

    print(format_table(rows))
    print(f"({len(rows)} row(s) returned)\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Layer 4 sanity queries")
    parser.add_argument("query", help="Query number (1-5), name, or 'all'")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    target = args.query.strip().lower()

    neo4j_ok = is_neo4j_available()

    if target in ("all", "*"):
        for k in sorted(QUERY_FILES.keys()):
            run_query(k, data_dir, neo4j_ok=neo4j_ok)
    elif target in QUERY_FILES:
        run_query(target, data_dir, neo4j_ok=neo4j_ok)
    else:
        # Check if match filename
        matched = None
        for k, fname in QUERY_FILES.items():
            if target in fname.lower():
                matched = k
                break
        if matched:
            run_query(matched, data_dir)
        else:
            print(f"Error: Unknown query {args.query!r}. Use 1-5 or 'all'.")
            sys.exit(1)


if __name__ == "__main__":
    main()
