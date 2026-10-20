"""Query runner CLI for Layer 4/5b Knowledge Graph (app/graph/query.py).

Runs ready-made Cypher queries from queries/ against Neo4j if reachable
(LIVE mode), or evaluates equivalent logic against data/graph/graph.json
(OFFLINE mode) when --offline is specified or Neo4j is unreachable.

Usage:
    python -m app.graph.query 1
    python -m app.graph.query all
    python -m app.graph.query all --offline
    python -m app.graph.query 6 --namespace main
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from .neo4j_store import Neo4jStore, is_neo4j_available

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_DIR = ROOT / "queries"


QUERY_FILES: dict[str, str] = {
    "1": "1_initial_term_2026.cypher",
    "2": "2_clauses_with_templates.cypher",
    "3": "3_sows_with_parent_msa.cypher",
    "4": "4_governing_law_per_contract.cypher",
    "5": "5_superseded_clauses.cypher",
    "6": "6_weak_indemnity_expiring_2026.cypher",
}

QUERY_DESCRIPTIONS: dict[str, str] = {
    "1": "Contracts whose initial_term_end falls in 2026",
    "2": "Every ACTIVE clause with its standard template (INSTANCE_OF)",
    "3": "SOWs and their parent MSA (PARENT_OF)",
    "4": "Governing law per contract (eyeball NY vs CA mismatch)",
    "5": "Clauses superseded by an amendment (status=superseded)",
    "6": "Contracts: ACTIVE indemnity cap < 2x AND current term ends 2026",
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
        # Only ACTIVE clauses with their template
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
                    # Filter superseded clauses
                    if cl["properties"].get("status") == "superseded":
                        continue
                    cid = cl["id"]
                    doc_id = clause_contract.get(cid, "unknown")
                    results.append({
                        "contract_id": doc_id,
                        "section": cl["properties"].get("section_path"),
                        "clause_type": cl["properties"].get("clause_type"),
                        "template_id": t["properties"].get("template_id"),
                        "status": cl["properties"].get("status", "active"),
                    })
        results.sort(key=lambda r: (r["contract_id"], r["section"]))
        return results

    elif query_key == "3":
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

    elif query_key == "6":
        # Contracts with ACTIVE indemnification cap < 2.0 AND current_term_end in 2026
        # Build contract -> date map
        contract_dates: dict[str, list] = {}
        for e in edges:
            if e["type"] == "HAS_DATE":
                d_node = nodes.get(e["target"])
                if d_node and d_node["properties"].get("kind") == "current_term_end":
                    contract_dates.setdefault(e["source"], []).append(
                        d_node["properties"].get("date", "")
                    )
        # Build contract -> clause map (ACTIVE indemnification)
        clause_contract = {}
        for e in edges:
            if e["type"] == "CONTAINS":
                clause_contract[e["target"]] = e["source"]

        results = []
        seen_contracts: set[str] = set()
        for node in nodes.values():
            p = node["properties"]
            if node["label"] != "Clause":
                continue
            if p.get("clause_type") != "indemnification":
                continue
            if p.get("status") != "active":
                continue
            cap = p.get("cap_multiple")
            if cap is None and isinstance(p.get("attributes"), dict):
                cap = p["attributes"].get("cap_multiple")
            if cap is None:
                continue
            try:
                cap_f = float(cap)
            except (ValueError, TypeError):
                continue
            if cap_f >= 2.0:
                continue

            # Find parent contract
            contract_node_id = clause_contract.get(node["id"])
            if not contract_node_id:
                continue
            if contract_node_id in seen_contracts:
                continue

            # Check current_term_end in 2026
            dates_for_c = contract_dates.get(contract_node_id, [])
            matching_dates = [d for d in dates_for_c if str(d).startswith("2026")]
            if not matching_dates:
                continue

            c_node = nodes.get(contract_node_id)
            if not c_node:
                continue
            cp = c_node["properties"]
            results.append({
                "contract_id": cp.get("doc_id"),
                "doc_type": cp.get("doc_type"),
                "current_term_end": matching_dates[0],
                "cap_multiple": cap_f,
                "clause_id": p.get("clause_id"),
            })
            seen_contracts.add(contract_node_id)

        results.sort(key=lambda r: r["contract_id"])
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


def run_query(
    query_key: str,
    data_dir: Path,
    neo4j_ok: Optional[bool] = None,
    force_offline: bool = False,
    namespace: str = "main",
) -> list[dict[str, Any]]:
    """Run specified sanity query by key ('1'-'6').

    In LIVE mode (Neo4j reachable and not --offline): executes the .cypher
    file directly against the database.  In OFFLINE mode: evaluates
    equivalent Python logic against graph.json.
    """
    if query_key not in QUERY_FILES:
        raise ValueError(f"Unknown query {query_key!r}. Valid keys: {list(QUERY_FILES.keys())}")

    q_file = QUERIES_DIR / QUERY_FILES[query_key]
    desc = QUERY_DESCRIPTIONS[query_key]

    print(f"\n[{query_key}] {desc}")
    print(f"File: {q_file.relative_to(ROOT)}")

    use_neo4j = (not force_offline) and (
        is_neo4j_available() if neo4j_ok is None else neo4j_ok
    )

    if use_neo4j:
        print(f"(Mode: LIVE – Neo4j database, namespace={namespace})")
        with Neo4jStore(namespace=namespace) as store:
            cypher = q_file.read_text(encoding="utf-8")
            # Strip Cypher comment lines (start with //) for cleaner output
            cypher_clean = "\n".join(
                line for line in cypher.splitlines() if not line.strip().startswith("//")
            )
            params: dict[str, Any] = {}
            if namespace:
                params["ns"] = namespace
                if "{ns:" not in cypher_clean and "$ns" not in cypher_clean:
                    cypher_clean = re.sub(
                        r"MATCH \(([a-zA-Z0-9_]+):Contract",
                        r"MATCH (\1:Contract {ns: $ns}",
                        cypher_clean,
                        count=1,
                    )
            rows = store.query(cypher_clean, parameters=params)
    else:
        print("(Mode: OFFLINE – evaluated against JSON graph)")
        rows = _run_json_fallback(query_key, data_dir)

    print(format_table(rows))
    print(f"({len(rows)} row(s) returned)\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Layer 4/5b sanity queries")
    parser.add_argument("query", help="Query number (1-6), name, or 'all'")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument(
        "--offline", action="store_true",
        help="Force offline JSON fallback even if Neo4j is available"
    )
    parser.add_argument(
        "--namespace", default="main",
        help="Neo4j namespace to query against (default: main)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    target = args.query.strip().lower()
    force_offline = args.offline

    neo4j_ok = (not force_offline) and is_neo4j_available()

    if target in ("all", "*"):
        for k in sorted(QUERY_FILES.keys()):
            run_query(k, data_dir, neo4j_ok=neo4j_ok, force_offline=force_offline, namespace=args.namespace)
    elif target in QUERY_FILES:
        run_query(target, data_dir, neo4j_ok=neo4j_ok, force_offline=force_offline, namespace=args.namespace)
    else:
        # Try matching by filename fragment
        matched = None
        for k, fname in QUERY_FILES.items():
            if target in fname.lower():
                matched = k
                break
        if matched:
            run_query(matched, data_dir, neo4j_ok=neo4j_ok, force_offline=force_offline, namespace=args.namespace)
        else:
            print(f"Error: Unknown query {args.query!r}. Use 1-6 or 'all'.")
            sys.exit(1)


if __name__ == "__main__":
    main()
