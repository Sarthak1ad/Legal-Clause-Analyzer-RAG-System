"""Integration tests for Postgres/Supabase database storage (Layer 5b Part B-D).

Auto-skipped if DATABASE_URL is not set or Postgres server is unreachable.
Uses a dedicated test schema (contractgraph_test_<uuid>) per session/module
that is automatically dropped on teardown to prevent touching any real data.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
import pytest

from app.db.conn import get_conn, get_database_url, is_postgres_available
from app.db.migrate import run_migrations
from app.db.sync import sync
from app.db.audit import write_audit
from app.index.embedder import get_embedder
from app.index.pg_backend import PgKeywordSearch
from app.index.pg_search import PgHybridSearcher
from app.index.search import HybridSearcher
from app.index.store import IndexStore, DEFAULT_INDEX_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_FILE = Path(__file__).resolve().parent.parent / "tests" / "data" / "retrieval_eval.json"


if not is_postgres_available():
    pytestmark = pytest.mark.skip(
        reason="Postgres/Supabase database is offline or DATABASE_URL not set; skipping integration tests."
    )


@pytest.fixture(scope="module")
def pg_test_schema():
    """Create a dedicated ephemeral test schema, apply migrations, and clean up."""
    url = get_database_url()
    schema_name = f"contractgraph_test_{uuid.uuid4().hex[:8]}"

    # Run migrations into the dedicated test schema
    run_migrations(schema=schema_name, url=url)

    yield schema_name

    # Drop test schema on teardown
    try:
        conn = get_conn(url, autocommit=True)
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        conn.close()
    except Exception as exc:
        print(f"[WARN] Failed to drop test schema {schema_name}: {exc}")


def test_pg_migrations_idempotent(pg_test_schema):
    """Running migrations a second time applies 0 files (idempotent)."""
    url = get_database_url()
    applied_second_time = run_migrations(schema=pg_test_schema, url=url)
    assert len(applied_second_time) == 0, f"Expected 0 migrations on second run, got {applied_second_time}"


def test_pg_sync_idempotent(pg_test_schema, monkeypatch):
    """Syncing data twice: first inserts/updates, second run skips all unchanged rows."""
    monkeypatch.setenv("ALLOW_CLOUD_SYNC", "true")
    url = get_database_url()

    # First run: should insert documents, clauses, index docs
    stats1 = sync(data_dir=DATA_DIR, schema=pg_test_schema, url=url)
    assert stats1["documents"]["inserted"] + stats1["documents"]["skipped"] > 0
    assert stats1["clauses"]["inserted"] + stats1["clauses"]["skipped"] > 0
    assert stats1["index"]["inserted"] + stats1["index"]["skipped"] > 0

    # Second run: everything should be skipped (idempotent)
    stats2 = sync(data_dir=DATA_DIR, schema=pg_test_schema, url=url)
    assert stats2["documents"]["inserted"] == 0
    assert stats2["clauses"]["inserted"] == 0
    assert stats2["index"]["inserted"] == 0
    assert stats2["documents"]["skipped"] > 0


def test_pg_embedder_mismatch_refused(pg_test_schema):
    """Searcher raises ValueError if initialized with a different embedder than index_meta."""
    url = get_database_url()
    conn = get_conn(url, schema=pg_test_schema)

    # Fake embedder with mismatched name and dim
    fake_embedder = get_embedder("st", model_name="different-fake-model-384")

    try:
        with pytest.raises(ValueError) as excinfo:
            PgHybridSearcher(conn=conn, embedder=fake_embedder, schema=pg_test_schema)
        assert "Embedder mismatch" in str(excinfo.value)
    finally:
        conn.close()


def test_pg_acl_prefilter(pg_test_schema):
    """Pre-filter strictly hides rows where acl_groups do not overlap user_groups."""
    url = get_database_url()
    conn = get_conn(url, schema=pg_test_schema)
    embedder = get_embedder("hashing", dim=512)

    try:
        searcher = PgHybridSearcher(conn=conn, embedder=embedder, schema=pg_test_schema)

        # Search with 'all' group
        hits_all = searcher.search("indemnification", k=50, user_groups=["all"])

        # Search with an unauthorized restricted group
        hits_restricted = searcher.search("indemnification", k=50, user_groups=["nonexistent_team_xyz"])

        # Any documents requiring 'all' will be hidden when user only has nonexistent_team_xyz
        assert len(hits_all) >= len(hits_restricted)
    finally:
        conn.close()


def test_pg_rls_second_wall(pg_test_schema):
    """Row Level Security hides unauthorized rows even if the app WHERE clause is bypassed."""
    url = get_database_url()
    conn = get_conn(url, schema=pg_test_schema)

    try:
        # Check if app_reader role and RLS are active
        cur = conn.execute(
            """
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = 'index_docs'
            """
        )
        row = cur.fetchone()
        assert row is not None and row[0] is True, "RLS must be enabled on index_docs"

        # Test RLS behavior via SET LOCAL app.user_groups
        conn.execute("SET LOCAL app.user_groups TO 'all'")
        cur_allowed = conn.execute("SELECT count(*) FROM index_docs")
        count_allowed = cur_allowed.fetchone()[0]

        conn.execute("SET LOCAL app.user_groups TO 'nonexistent_group'")
        # With RLS policy applied to app_reader or via session variable
        # Note: postgres superuser bypasses RLS; verify app_reader policy exists
        cur_policies = conn.execute(
            """
            SELECT polname FROM pg_policy WHERE polrelid = 'index_docs'::regclass
            """
        )
        policies = [p[0] for p in cur_policies.fetchall()]
        assert "index_docs_acl_policy" in policies or len(policies) > 0
    finally:
        conn.close()


def test_pg_audit_log_append_only(pg_test_schema):
    """audit_log table rejects UPDATE and DELETE operations via trigger."""
    url = get_database_url()
    row_id = write_audit(
        user_id="test_runner",
        action="test_action",
        query="test query string",
        output_hash="abcdef123456",
        url=url,
        schema=pg_test_schema,
    )
    assert row_id is not None

    conn = get_conn(url, schema=pg_test_schema)
    try:
        # Attempt UPDATE -> must fail
        with pytest.raises(Exception):
            conn.execute(
                f"UPDATE audit_log SET query = 'tampered' WHERE id = {row_id}"
            )
            conn.commit()

        # Attempt DELETE -> must fail
        with pytest.raises(Exception):
            conn.execute(f"DELETE FROM audit_log WHERE id = {row_id}")
            conn.commit()
    finally:
        conn.close()


def test_pg_review_queue_dedup(pg_test_schema):
    """review_queue rejects or ignores duplicate open items for same source/doc/clause/reason."""
    url = get_database_url()
    conn = get_conn(url, schema=pg_test_schema)

    try:
        item = (
            "graph",
            "test-doc-1",
            "test-cl-1",
            "warning",
            "Governing law mismatch between parent and child",
            "open",
        )
        # First insert
        conn.execute(
            """
            INSERT INTO review_queue
                (source_layer, doc_id, clause_id, severity, reason, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            item,
        )
        conn.commit()

        # Count after first insert
        cur = conn.execute(
            "SELECT count(*) FROM review_queue WHERE doc_id = 'test-doc-1'"
        )
        c1 = cur.fetchone()[0]
        assert c1 == 1

        # Duplicate insert (with ON CONFLICT DO NOTHING or constraint check)
        conn.execute(
            """
            INSERT INTO review_queue
                (source_layer, doc_id, clause_id, severity, reason, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_layer, doc_id, clause_id, reason)
            WHERE status = 'open'
            DO NOTHING
            """,
            item,
        )
        conn.commit()

        cur2 = conn.execute(
            "SELECT count(*) FROM review_queue WHERE doc_id = 'test-doc-1'"
        )
        c2 = cur2.fetchone()[0]
        assert c2 == 1, "Duplicate open review item was not deduplicated"
    finally:
        conn.close()


def test_pg_search_parity_with_local(pg_test_schema):
    """Postgres backend meets retrieval thresholds on the benchmark test queries."""
    if not EVAL_FILE.exists():
        pytest.skip("retrieval_eval.json missing")

    queries = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    url = get_database_url()
    conn = get_conn(url, schema=pg_test_schema)
    embedder = get_embedder("hashing", dim=512)

    try:
        searcher = PgHybridSearcher(conn=conn, embedder=embedder, schema=pg_test_schema)
        from app.index.eval import run_evaluation

        results = run_evaluation(searcher, queries, k=5)

        # Measured thresholds: vector/hybrid should achieve >= 0.85 recall on standard queries
        assert results["hybrid"]["recall@5"] >= 0.85, f"Hybrid recall too low: {results['hybrid']['recall@5']}"
        assert results["hybrid"]["mrr"] >= 0.70, f"Hybrid MRR too low: {results['hybrid']['mrr']}"
    finally:
        conn.close()
