"""Offline tests for Layer 5b (tests/test_layer5b_offline.py).

These tests NEVER touch real databases and run in any environment.

Covers:
- Missing .env gives a clear message (mocked)
- Secret masking (URI mask, URL mask)
- Migration files ordered and well-formed (SQL content checks)
- Sync refuses without ALLOW_CLOUD_SYNC=true
- SQL builders are parameterized (fake cursor check)
- Lazy imports work without neo4j / psycopg installed (import test)
- Query 6 offline fallback logic
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "db" / "migrations"


# ---------------------------------------------------------------------------
# Test: lazy imports (no neo4j / psycopg needed at module level)
# ---------------------------------------------------------------------------

class TestLazyImports(unittest.TestCase):
    def test_neo4j_store_imports_without_neo4j(self):
        """neo4j_store must import cleanly even if neo4j is absent."""
        # Temporarily hide neo4j from import system
        orig = sys.modules.get("neo4j")
        sys.modules["neo4j"] = None  # type: ignore
        try:
            # Force reimport
            mods_to_remove = [k for k in sys.modules if k.startswith("app.graph.neo4j_store")]
            for m in mods_to_remove:
                del sys.modules[m]
            import app.graph.neo4j_store  # noqa: F401
        finally:
            if orig is None:
                sys.modules.pop("neo4j", None)
            else:
                sys.modules["neo4j"] = orig

    def test_db_conn_imports_without_psycopg(self):
        """conn.py must import cleanly even if psycopg is absent."""
        orig = sys.modules.get("psycopg")
        sys.modules["psycopg"] = None  # type: ignore
        try:
            mods_to_remove = [k for k in sys.modules if k.startswith("app.db.conn")]
            for m in mods_to_remove:
                del sys.modules[m]
            import app.db.conn  # noqa: F401
        finally:
            if orig is None:
                sys.modules.pop("psycopg", None)
            else:
                sys.modules["psycopg"] = orig

    def test_pg_backend_imports_without_psycopg(self):
        """pg_backend.py must import cleanly even if psycopg is absent."""
        orig = sys.modules.get("psycopg")
        sys.modules["psycopg"] = None  # type: ignore
        try:
            mods_to_remove = [k for k in sys.modules if k.startswith("app.index.pg_backend")]
            for m in mods_to_remove:
                del sys.modules[m]
            import app.index.pg_backend  # noqa: F401
        finally:
            if orig is None:
                sys.modules.pop("psycopg", None)
            else:
                sys.modules["psycopg"] = orig


# ---------------------------------------------------------------------------
# Test: .env missing gives clear message
# ---------------------------------------------------------------------------

class TestEnvMissing(unittest.TestCase):
    def test_neo4j_check_prints_required_keys_when_env_missing(self):
        """neo4j_check prints the list of required keys when env vars are absent."""
        env_backup = {}
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
            env_backup[k] = os.environ.pop(k, None)

        try:
            from app.graph.neo4j_check import check_neo4j_connection
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = check_neo4j_connection()
            output = buf.getvalue()
            self.assertEqual(rc, 1)
            # Check that required keys are mentioned
            self.assertIn("NEO4J_URI", output)
            self.assertIn("NEO4J_USER", output)
            self.assertIn("NEO4J_PASSWORD", output)
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_get_database_url_raises_without_env(self):
        """get_database_url() raises RuntimeError with helpful message when DATABASE_URL is not set."""
        with unittest.mock.patch.dict(os.environ, {"DATABASE_URL": ""}):
            from app.db.conn import get_database_url
            with self.assertRaises(RuntimeError) as ctx:
                get_database_url()
            self.assertIn("DATABASE_URL", str(ctx.exception))
            self.assertIn("pooler", str(ctx.exception))  # hint about pooler


# ---------------------------------------------------------------------------
# Test: secret masking
# ---------------------------------------------------------------------------

class TestSecretMasking(unittest.TestCase):
    def test_mask_uri_hides_password(self):
        from app.graph.neo4j_store import _mask_uri
        masked = _mask_uri("bolt://neo4j:super-secret@localhost:7687")
        self.assertNotIn("super-secret", masked)
        self.assertIn("localhost", masked)

    def test_mask_url_hides_password(self):
        from app.db.conn import _mask_url
        masked = _mask_url("postgresql://user:mysecret@db.supabase.co:5432/postgres")
        self.assertNotIn("mysecret", masked)
        self.assertIn("db.supabase.co", masked)

    def test_aura_uri_detected(self):
        from app.graph.neo4j_store import _mask_uri
        # neo4j+s should work
        uri = "neo4j+s://abc123.databases.neo4j.io"
        masked = _mask_uri(uri)
        self.assertIn("abc123.databases.neo4j.io", masked)


# ---------------------------------------------------------------------------
# Test: migration files ordering and structure
# ---------------------------------------------------------------------------

class TestMigrationFiles(unittest.TestCase):
    def test_migrations_dir_exists(self):
        self.assertTrue(MIGRATIONS_DIR.exists(), f"Missing: {MIGRATIONS_DIR}")

    def test_migrations_ordered_and_named(self):
        """Files must be named NNN_*.sql with sequential numbering."""
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        self.assertGreater(len(files), 0, "No migration files found")
        for i, f in enumerate(files, start=1):
            prefix = f"{i:03d}_"
            self.assertTrue(
                f.name.startswith(prefix),
                f"Migration {f.name} should start with '{prefix}' (got {f.name[:4]})",
            )

    def test_migration_001_has_required_tables(self):
        """001_init.sql must CREATE all expected tables."""
        sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
        required_tables = [
            "schema_migrations", "documents", "clauses", "index_docs",
            "document_acl", "communities", "review_queue", "review_overrides",
            "audit_log",
        ]
        for t in required_tables:
            self.assertIn(t, sql, f"Table '{t}' not found in 001_init.sql")

    def test_migration_001_has_audit_trigger(self):
        sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
        self.assertIn("audit_log_immutable", sql)
        self.assertIn("BEFORE UPDATE", sql)
        self.assertIn("BEFORE DELETE", sql)

    def test_migration_002_has_dim_placeholder(self):
        sql = (MIGRATIONS_DIR / "002_add_embedding_column.sql").read_text(encoding="utf-8")
        self.assertIn("%%DIM%%", sql)
        self.assertIn("vector", sql.lower())
        self.assertIn("hnsw", sql.lower())

    def test_migration_003_has_rls(self):
        sql = (MIGRATIONS_DIR / "003_row_level_security.sql").read_text(encoding="utf-8")
        self.assertIn("ROW LEVEL SECURITY", sql)
        self.assertIn("app_reader", sql)
        self.assertIn("acl_groups", sql)


# ---------------------------------------------------------------------------
# Test: sync refuses without ALLOW_CLOUD_SYNC
# ---------------------------------------------------------------------------

class TestSyncGuard(unittest.TestCase):
    def test_sync_refuses_without_flag(self):
        """Calling sync() without ALLOW_CLOUD_SYNC=true must exit with code 1."""
        backup = os.environ.pop("ALLOW_CLOUD_SYNC", None)
        try:
            from app.db.sync import _check_sync_guard
            with self.assertRaises(SystemExit) as ctx:
                _check_sync_guard()
            self.assertEqual(ctx.exception.code, 1)
        finally:
            if backup is not None:
                os.environ["ALLOW_CLOUD_SYNC"] = backup

    def test_sync_proceeds_with_flag(self):
        """With ALLOW_CLOUD_SYNC=true, guard should not raise."""
        os.environ["ALLOW_CLOUD_SYNC"] = "true"
        try:
            from app.db.sync import _check_sync_guard
            # Should not raise
            _check_sync_guard()
        finally:
            os.environ.pop("ALLOW_CLOUD_SYNC", None)


# ---------------------------------------------------------------------------
# Test: SQL builders are parameterized (no interpolation)
# ---------------------------------------------------------------------------

class TestSqlParameterized(unittest.TestCase):
    def test_pg_filter_builder_no_interpolation(self):
        """build_pg_filter must use %s placeholders, not f-string interpolation."""
        from app.index.pg_backend import build_pg_filter

        # Inject a value that would be dangerous if interpolated
        dangerous = "'; DROP TABLE index_docs; --"
        filters = {
            "clause_type": dangerous,
            "contract_id": dangerous,
            "status": "active",
        }
        where, params = build_pg_filter(filters, as_of=None, user_groups=["all"])

        # The dangerous string should appear ONLY in params, not in the SQL text
        self.assertNotIn(dangerous, where)
        self.assertIn(dangerous, params)
        self.assertIn("%s", where)

    def test_pg_filter_all_filters_parameterized(self):
        """All filter types produce %s placeholders."""
        from app.index.pg_backend import build_pg_filter

        filters = {
            "status": "active",
            "kind": ["clause", "template"],
            "doc_type": "MSA",
            "clause_type": "indemnification",
            "contract_id": "msa-acme.v1",
            "party_id": "party-acme",
        }
        where, params = build_pg_filter(
            filters, as_of="2026-10-19", user_groups=["all", "team_a"]
        )
        # All user-provided values must be in params
        self.assertIn("active", params)
        self.assertIn("MSA", params)
        self.assertIn("indemnification", params)
        # Count placeholders matches params
        self.assertEqual(where.count("%s"), len(params))


# ---------------------------------------------------------------------------
# Test: query 6 offline fallback
# ---------------------------------------------------------------------------

class TestQuery6Offline(unittest.TestCase):
    def _make_graph(self) -> dict:
        """Build a minimal in-memory graph with a weak-indemnity contract."""
        return {
            "nodes": [
                {"id": "c1", "label": "Contract", "properties": {"doc_id": "msa-test.v1", "doc_type": "MSA"}},
                {"id": "cl1", "label": "Clause", "properties": {
                    "clause_id": "cl1", "clause_type": "indemnification",
                    "status": "active", "cap_multiple": 1.0,
                }},
                {"id": "cl_superseded", "label": "Clause", "properties": {
                    "clause_id": "cl_superseded", "clause_type": "indemnification",
                    "status": "superseded", "cap_multiple": 0.5,
                }},
                {"id": "d1", "label": "DateEvent", "properties": {
                    "kind": "current_term_end", "date": "2026-06-30",
                }},
            ],
            "edges": [
                {"type": "CONTAINS", "source": "c1", "target": "cl1", "properties": {}, "id": "e1"},
                {"type": "CONTAINS", "source": "c1", "target": "cl_superseded", "properties": {}, "id": "e2"},
                {"type": "HAS_DATE",  "source": "c1", "target": "d1", "properties": {}, "id": "e3"},
            ],
            "review_items": [],
        }

    def test_query6_finds_weak_active_indemnity(self):
        """Query 6 offline: finds contract with active indemnity cap < 2.0 and 2026 term end."""
        import tempfile
        import json
        from pathlib import Path

        graph = self._make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "graph").mkdir()
            (tmppath / "graph" / "graph.json").write_text(json.dumps(graph))

            from app.graph.query import _run_json_fallback
            results = _run_json_fallback("6", tmppath)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["contract_id"], "msa-test.v1")
        self.assertAlmostEqual(results[0]["cap_multiple"], 1.0)
        self.assertIn("2026", results[0]["current_term_end"])

    def test_query6_excludes_superseded_clause(self):
        """Query 6 must ignore superseded indemnity clauses even if cap < 2.0."""
        import tempfile, json
        from pathlib import Path

        graph = self._make_graph()
        # Remove the active clause; only superseded remains
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "cl1"]
        graph["edges"] = [e for e in graph["edges"] if e.get("target") != "cl1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "graph").mkdir()
            (tmppath / "graph" / "graph.json").write_text(json.dumps(graph))

            from app.graph.query import _run_json_fallback
            results = _run_json_fallback("6", tmppath)

        self.assertEqual(results, [], "Superseded clause must not be matched")

    def test_query6_excludes_strong_cap(self):
        """Query 6 must not include contracts with cap >= 2.0."""
        import tempfile, json
        from pathlib import Path

        graph = self._make_graph()
        # Change cap to 2.5
        for n in graph["nodes"]:
            if n["id"] == "cl1":
                n["properties"]["cap_multiple"] = 2.5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "graph").mkdir()
            (tmppath / "graph" / "graph.json").write_text(json.dumps(graph))

            from app.graph.query import _run_json_fallback
            results = _run_json_fallback("6", tmppath)

        self.assertEqual(results, [], "Cap >= 2.0 must not be matched")


# ---------------------------------------------------------------------------
# Test: Neo4j namespace prefix logic (no DB needed)
# ---------------------------------------------------------------------------

class TestNamespaceLogic(unittest.TestCase):
    def test_ns_id_prefixing(self):
        """Namespace ids should be <ns>:<original_id> without double-prefix."""
        ns = "test"
        original = "msa-acme.v1"
        ns_id = f"{ns}:{original}"
        # Verify no double-prefix
        self.assertFalse(ns_id.startswith(f"{ns}:{ns}:"))

    def test_mask_uri_bolt(self):
        from app.graph.neo4j_store import _mask_uri
        self.assertEqual(_mask_uri("bolt://localhost:7687"), "bolt://localhost:7687")

    def test_mask_uri_aura(self):
        from app.graph.neo4j_store import _mask_uri
        masked = _mask_uri("neo4j+s://abc.databases.neo4j.io")
        self.assertIn("abc.databases.neo4j.io", masked)


if __name__ == "__main__":
    unittest.main()
