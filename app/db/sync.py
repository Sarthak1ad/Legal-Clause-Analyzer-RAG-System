"""Sync local data artifacts to Supabase (app/db/sync.py).

Reads local data/parsed, extracted, resolved, graph, and the Layer 5
local index, then upserts into Postgres tables.

SECURITY GUARD: the sync REFUSES to run unless ALLOW_CLOUD_SYNC=true.
A clear warning is printed before any data leaves this machine.

Usage:
    python -m app.db.sync --data-dir data [--schema public]
        [--only documents,clauses,index,communities,review]

Design:
- Idempotent: rows whose text_hash is unchanged are skipped.
- Reuses local embeddings from IndexStore; does NOT re-embed.
- All SQL is parameterized.
- Counts per table printed at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR_DEFAULT = ROOT / "data"


def _check_sync_guard() -> None:
    """Refuse to proceed unless ALLOW_CLOUD_SYNC=true."""
    if os.environ.get("ALLOW_CLOUD_SYNC", "").lower() != "true":
        print("[ERROR] Cloud sync is disabled.")
        print()
        print("  Data leaving this machine requires explicit opt-in.")
        print("  Set ALLOW_CLOUD_SYNC=true in your .env file to proceed.")
        print()
        print("  WARNING: Only sync synthetic sample data during development.")
        print("           Real contracts are confidential.")
        sys.exit(1)


def _print_cloud_warning() -> None:
    """Print a prominent warning before data is sent to Supabase."""
    print()
    print("!" * 65)
    print("  WARNING: DATA WILL LEAVE THIS MACHINE")
    print("  Contract data is being uploaded to Supabase (cloud).")
    print("  Only synthetic/sample data should be synced during dev.")
    print("!" * 65)
    print()


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------

def _upsert_documents(
    conn: Any, data_dir: Path
) -> tuple[int, int, int]:
    """Upsert documents from data/parsed/*.json. Returns (inserted, updated, skipped)."""
    parsed_dir = data_dir / "parsed"
    if not parsed_dir.exists():
        return 0, 0, 0

    from ..models import ParsedDocument

    inserted = updated = skipped = 0

    for p in sorted(parsed_dir.glob("*.json")):
        doc = ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))

        # Check existing
        cur = conn.execute(
            "SELECT sha256 FROM documents WHERE doc_id = %s AND version = %s",
            (doc.doc_id, doc.version),
        )
        existing = cur.fetchone()
        sha256 = doc.sha256 if hasattr(doc, "sha256") else None

        if existing:
            if existing[0] == sha256:
                skipped += 1
                continue
            conn.execute(
                """
                UPDATE documents SET sha256=%s, doc_type=%s, title=%s,
                    effective_date=%s, status=%s, source_file=%s, updated_at=NOW()
                WHERE doc_id=%s AND version=%s
                """,
                (sha256, doc.doc_type, doc.title, doc.metadata.effective_date if doc.metadata else None,
                 "active", str(p), doc.doc_id, doc.version),
            )
            updated += 1
        else:
            conn.execute(
                """
                INSERT INTO documents (doc_id, version, sha256, doc_type, title,
                    effective_date, status, source_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, version) DO NOTHING
                """,
                (doc.doc_id, doc.version, sha256, doc.doc_type, doc.title,
                 doc.metadata.effective_date if doc.metadata else None,
                 "active", str(p)),
            )
            inserted += 1

    conn.commit()
    return inserted, updated, skipped


def _upsert_clauses(
    conn: Any, data_dir: Path
) -> tuple[int, int, int]:
    """Upsert clauses from data/extracted/*.json. Returns (inserted, updated, skipped)."""
    from ..extraction.schemas import ExtractedDocument
    import hashlib

    ext_dir = data_dir / "extracted"
    if not ext_dir.exists():
        return 0, 0, 0

    inserted = updated = skipped = 0

    for p in sorted(ext_dir.glob("*.json")):
        edoc = ExtractedDocument.model_validate_json(p.read_text(encoding="utf-8"))
        for clause in edoc.clauses:
            text = clause.text or ""
            text_hash = hashlib.md5(text.encode()).hexdigest()
            attrs = clause.attributes.model_dump() if clause.attributes else {}

            cur = conn.execute(
                "SELECT text_hash FROM clauses WHERE clause_id = %s",
                (clause.clause_id,),
            )
            existing = cur.fetchone()

            if existing:
                if existing[0] == text_hash:
                    skipped += 1
                    continue
                conn.execute(
                    """
                    UPDATE clauses SET doc_id=%s, section_path=%s, heading=%s,
                        clause_type=%s, span_id=%s, text=%s, text_hash=%s,
                        status=%s, attributes=%s, updated_at=NOW()
                    WHERE clause_id=%s
                    """,
                    (edoc.doc_id, clause.section_path, clause.heading,
                     clause.clause_type, clause.span_id, text, text_hash,
                     "active", json.dumps(attrs), clause.clause_id),
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO clauses (clause_id, doc_id, section_path, heading,
                        clause_type, span_id, text, text_hash, status, needs_review, attributes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (clause_id) DO NOTHING
                    """,
                    (clause.clause_id, edoc.doc_id, clause.section_path, clause.heading,
                     clause.clause_type, clause.span_id, text, text_hash,
                     "active", clause.needs_review if hasattr(clause, "needs_review") else False,
                     json.dumps(attrs)),
                )
                inserted += 1

    conn.commit()
    return inserted, updated, skipped


def _upsert_index(
    conn: Any, data_dir: Path, embedder_name: str, embed_dim: int
) -> tuple[int, int, int]:
    """Upsert index_docs from local SQLite IndexStore. Returns (inserted, updated, skipped)."""
    from ..index.store import IndexStore, DEFAULT_INDEX_DIR
    import numpy as np

    index_dir = data_dir / "index" if (data_dir / "index").exists() else DEFAULT_INDEX_DIR
    if not index_dir.exists():
        return 0, 0, 0

    store = IndexStore(index_dir)

    # Validate embedder match
    stored_name = store.get_meta("embedder_name") or ""
    stored_dim = int(store.get_meta("embedder_dim") or "0")

    # Check PG embedder meta
    cur = conn.execute("SELECT value FROM index_meta WHERE key='embedder_name'")
    pg_embedder = cur.fetchone()

    if pg_embedder and pg_embedder[0] != stored_name:
        print(f"[ERROR] Embedder mismatch: Postgres has '{pg_embedder[0]}' but local index uses '{stored_name}'.")
        print("  Run: python -m app.index.run build --rebuild, then sync again.")
        return 0, 0, 0

    # Record embedder in pg
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES ('embedder_name', %s)"
        " ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()",
        (stored_name,),
    )
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES ('embedder_dim', %s)"
        " ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()",
        (str(stored_dim),),
    )

    # Fetch all rows from local SQLite
    rows = store._conn.execute(
        "SELECT rowid, doc_key, kind, ref_id, contract_id, doc_type, clause_type, "
        "section_path, heading, text, span_id, status, effective_from, effective_to, "
        "party_ids, acl_groups, attributes_json, text_hash FROM docs"
    ).fetchall()

    # Build rowid -> vector map
    vec_map: dict[int, Any] = {}
    for i, rowid in enumerate(store.vectors._rowids):
        if i < store.vectors._vectors.shape[0]:
            vec_map[rowid] = store.vectors._vectors[i]

    inserted = updated = skipped = 0

    for r in rows:
        rowid = r["rowid"]
        doc_key = r["doc_key"]
        text_hash = r["text_hash"]

        # Check existing
        cur = conn.execute(
            "SELECT text_hash FROM index_docs WHERE doc_key = %s", (doc_key,)
        )
        existing = cur.fetchone()

        vec = vec_map.get(rowid)
        vec_list = vec.tolist() if vec is not None else None

        party_ids = json.loads(r["party_ids"]) if r["party_ids"] else []
        acl_groups_str = r["acl_groups"]
        acl_groups = json.loads(acl_groups_str) if acl_groups_str else ["all"]
        attrs_text = r["attributes_text"] if "attributes_text" in r.keys() else ""

        if existing:
            if existing[0] == text_hash:
                skipped += 1
                continue
            conn.execute(
                """
                UPDATE index_docs SET kind=%s, ref_id=%s, contract_id=%s, doc_type=%s,
                    clause_type=%s, section_path=%s, heading=%s, text=%s, span_id=%s,
                    status=%s, effective_from=%s, effective_to=%s, party_ids=%s,
                    acl_groups=%s, attributes_text=%s, text_hash=%s, embedder=%s,
                    embedding=%s, updated_at=NOW()
                WHERE doc_key=%s
                """,
                (r["kind"], r["ref_id"], r["contract_id"], r["doc_type"],
                 r["clause_type"], r["section_path"], r["heading"], r["text"],
                 r["span_id"], r["status"], r["effective_from"], r["effective_to"],
                 json.dumps(party_ids), acl_groups, attrs_text, text_hash,
                 stored_name, vec_list, doc_key),
            )
            updated += 1
        else:
            conn.execute(
                """
                INSERT INTO index_docs (doc_key, kind, ref_id, contract_id, doc_type,
                    clause_type, section_path, heading, text, span_id, status,
                    effective_from, effective_to, party_ids, acl_groups, attributes_text,
                    text_hash, embedder, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (doc_key) DO NOTHING
                """,
                (doc_key, r["kind"], r["ref_id"], r["contract_id"], r["doc_type"],
                 r["clause_type"], r["section_path"], r["heading"], r["text"],
                 r["span_id"], r["status"], r["effective_from"], r["effective_to"],
                 json.dumps(party_ids), acl_groups, attrs_text, text_hash,
                 stored_name, vec_list),
            )
            inserted += 1

    conn.commit()
    return inserted, updated, skipped


def _upsert_communities(
    conn: Any, data_dir: Path
) -> tuple[int, int, int]:
    """Upsert community summaries from data/index/communities.json."""
    import hashlib
    c_path = data_dir / "index" / "communities.json"
    if not c_path.exists():
        return 0, 0, 0

    data = json.loads(c_path.read_text(encoding="utf-8"))
    communities = data.get("communities", [])
    inserted = updated = skipped = 0

    for comm in communities:
        cid = comm["id"]
        summary_str = json.dumps(comm.get("stats") or {})
        source_hash = hashlib.md5(summary_str.encode()).hexdigest()

        cur = conn.execute(
            "SELECT source_hash FROM communities WHERE community_id = %s", (cid,)
        )
        existing = cur.fetchone()
        pg_str = ",".join(comm.get("permission_groups") or ["all"])

        if existing:
            if existing[0] == source_hash:
                skipped += 1
                continue
            conn.execute(
                """
                UPDATE communities SET kind=%s, permission_group=%s, summary=%s,
                    source_hash=%s, stale=FALSE, updated_at=NOW()
                WHERE community_id=%s
                """,
                (comm.get("kind",""), pg_str, json.dumps(comm),
                 source_hash, cid),
            )
            updated += 1
        else:
            conn.execute(
                """
                INSERT INTO communities (community_id, kind, permission_group,
                    summary, source_hash, stale)
                VALUES (%s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (community_id) DO NOTHING
                """,
                (cid, comm.get("kind",""), pg_str, json.dumps(comm), source_hash),
            )
            inserted += 1

    conn.commit()
    return inserted, updated, skipped


def _upsert_review(
    conn: Any, data_dir: Path
) -> tuple[int, int, int]:
    """Upsert review items from graph review_items into review_queue."""
    graph_path = data_dir / "graph" / "graph.json"
    if not graph_path.exists():
        return 0, 0, 0

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    review_items = graph.get("review_items", [])
    inserted = skipped = 0

    for item in review_items:
        severity = item.get("severity", "warning")
        try:
            conn.execute(
                """
                INSERT INTO review_queue (source_layer, doc_id, clause_id,
                    severity, reason, details, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'open')
                ON CONFLICT DO NOTHING
                """,
                ("graph", item.get("doc_id"), item.get("span_id"),
                 severity, item.get("kind",""), json.dumps(item)),
            )
            inserted += 1
        except Exception:
            skipped += 1

    conn.commit()
    return inserted, 0, skipped


# ---------------------------------------------------------------------------
# Main sync function
# ---------------------------------------------------------------------------

def sync(
    data_dir: Path = DATA_DIR_DEFAULT,
    schema: str = "public",
    only: Optional[list[str]] = None,
    url: Optional[str] = None,
) -> dict[str, dict[str, int]]:
    """Sync all local artifacts to Supabase.

    Returns a dict of {table: {inserted, updated, skipped}}.
    """
    _check_sync_guard()
    _print_cloud_warning()

    from .conn import get_conn, get_database_url
    url = url or get_database_url()
    conn = get_conn(url, schema=schema)

    only_set = set(only) if only else None
    results: dict[str, dict[str, int]] = {}

    def _should(name: str) -> bool:
        return only_set is None or name in only_set

    if _should("documents"):
        print("Syncing: documents ...", end=" ", flush=True)
        i, u, s = _upsert_documents(conn, data_dir)
        results["documents"] = {"inserted": i, "updated": u, "skipped": s}
        print(f"inserted={i}, updated={u}, skipped={s}")

    if _should("clauses"):
        print("Syncing: clauses ...", end=" ", flush=True)
        i, u, s = _upsert_clauses(conn, data_dir)
        results["clauses"] = {"inserted": i, "updated": u, "skipped": s}
        print(f"inserted={i}, updated={u}, skipped={s}")

    if _should("index"):
        print("Syncing: index_docs ...", end=" ", flush=True)
        embedder_name = "hashing-512"
        embed_dim = 512
        i, u, s = _upsert_index(conn, data_dir, embedder_name, embed_dim)
        results["index_docs"] = {"inserted": i, "updated": u, "skipped": s}
        print(f"inserted={i}, updated={u}, skipped={s}")

    if _should("communities"):
        print("Syncing: communities ...", end=" ", flush=True)
        i, u, s = _upsert_communities(conn, data_dir)
        results["communities"] = {"inserted": i, "updated": u, "skipped": s}
        print(f"inserted={i}, updated={u}, skipped={s}")

    if _should("review"):
        print("Syncing: review_queue ...", end=" ", flush=True)
        i, u, s = _upsert_review(conn, data_dir)
        results["review_queue"] = {"inserted": i, "updated": u, "skipped": s}
        print(f"inserted={i}, updated={u}, skipped={s}")

    conn.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local artifacts to Supabase")
    parser.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT), help="Local data directory")
    parser.add_argument("--schema", default="public", help="Postgres schema (default: public)")
    parser.add_argument(
        "--only",
        help="Comma-separated subset: documents,clauses,index,communities,review",
    )
    args = parser.parse_args()

    only = [x.strip() for x in args.only.split(",")] if args.only else None
    results = sync(
        data_dir=Path(args.data_dir),
        schema=args.schema,
        only=only,
    )

    print()
    print("SYNC SUMMARY")
    print(f"  {'Table':<20} {'Inserted':>10} {'Updated':>10} {'Skipped':>10}")
    print("  " + "-" * 54)
    for table, counts in results.items():
        print(
            f"  {table:<20} {counts['inserted']:>10} {counts['updated']:>10} {counts['skipped']:>10}"
        )


from typing import Optional  # noqa: E402

if __name__ == "__main__":
    main()
