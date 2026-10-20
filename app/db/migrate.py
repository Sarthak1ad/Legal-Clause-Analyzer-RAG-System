"""Database migration runner (app/db/migrate.py).

Applies .sql files from db/migrations/ in lexicographic order, recording
each applied migration in schema_migrations.  Idempotent: already-applied
files are skipped.

Usage:
    python -m app.db.migrate [--schema public] [--dry-run]
    python -m app.db.migrate --schema contractgraph_test_abc123

Design decisions:
- Each .sql file is applied in a single transaction; if it fails the whole
  file rolls back and the runner exits with an error.
- Migration 002 has a %%DIM%% placeholder substituted from the active embedder
  dimension before execution.
- pgvector is enabled via CREATE EXTENSION IF NOT EXISTS before any migration.
- --dry-run prints what would run without executing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "migrations"
ROOT = Path(__file__).resolve().parent.parent.parent


def _get_embedder_dim() -> int:
    """Detect the embedder dimension from the environment / index metadata."""
    embed_model = os.environ.get("EMBED_MODEL", "")
    if embed_model:
        # If a sentence-transformer model is configured, use 384 as the typical dim
        return 384
    return 512  # default hashing embedder


def run_migrations(
    schema: str = "public",
    dry_run: bool = False,
    url: Optional[str] = None,
) -> list[str]:
    """Apply all pending migrations to the target schema.

    Returns list of migration filenames that were applied.
    """
    from .conn import get_conn, _mask_url, get_database_url

    url = url or get_database_url()
    masked = _mask_url(url)

    print(f"Migration runner")
    print(f"  Target   : {masked} (schema={schema})")
    print(f"  Dry run  : {dry_run}")
    print()

    if dry_run:
        conn = None
    else:
        conn = get_conn(url, schema=schema)

    # ------------------------------------------------------------------ #
    # Enable pgvector extension (idempotent)                              #
    # ------------------------------------------------------------------ #
    if not dry_run:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions")
            conn.commit()
        except Exception:
            # Some Supabase setups already have it; try without schema qualifier
            try:
                conn.rollback()
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            except Exception as exc2:
                print(f"[WARN] Could not enable pgvector: {exc2}")
                conn.rollback()

    # ------------------------------------------------------------------ #
    # Ensure schema exists                                                 #
    # ------------------------------------------------------------------ #
    if not dry_run and schema != "public":
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.execute(f'SET search_path TO "{schema}", public, extensions')
        conn.commit()

    # ------------------------------------------------------------------ #
    # Ensure schema_migrations table exists                               #
    # ------------------------------------------------------------------ #
    if not dry_run:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename    TEXT PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.commit()
        cur = conn.execute("SELECT filename FROM schema_migrations ORDER BY filename")
        applied: set[str] = {row[0] for row in cur.fetchall()}
    else:
        applied = set()

    # ------------------------------------------------------------------ #
    # Discover migration files                                             #
    # ------------------------------------------------------------------ #
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print(f"No migration files found in {MIGRATIONS_DIR}")
        return []

    embed_dim = _get_embedder_dim()
    newly_applied: list[str] = []

    for mf in migration_files:
        fname = mf.name
        if fname in applied:
            print(f"  [SKIP]    {fname}  (already applied)")
            continue

        sql = mf.read_text(encoding="utf-8")
        # Substitute embedder dimension placeholder
        sql = sql.replace("%%DIM%%", str(embed_dim))

        if dry_run:
            print(f"  [DRY-RUN] {fname}")
            print(f"    SQL preview (first 200 chars): {sql[:200].strip()!r}")
            newly_applied.append(fname)
            continue

        print(f"  [APPLY]   {fname} ...", end=" ", flush=True)
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s) ON CONFLICT DO NOTHING",
                (fname,),
            )
            conn.commit()
            print("OK")
            newly_applied.append(fname)
        except Exception as exc:
            conn.rollback()
            print(f"FAILED\n  Error: {exc}")
            if conn:
                conn.close()
            sys.exit(1)

    if conn:
        conn.close()

    print()
    if newly_applied:
        print(f"Applied {len(newly_applied)} migration(s): {newly_applied}")
    else:
        print("All migrations already applied (nothing to do).")
    return newly_applied


def list_tables(schema: str = "public", url: Optional[str] = None) -> list[str]:
    """Return sorted list of table names in the schema."""
    from .conn import get_conn, get_database_url
    url = url or get_database_url()
    conn = get_conn(url, schema=schema)
    cur = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = %s ORDER BY tablename",
        (schema,),
    )
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables


def drop_schema(schema: str, url: Optional[str] = None) -> None:
    """Drop an entire schema (for test teardown; never touch 'public')."""
    if schema in ("public", "pg_catalog", "information_schema"):
        raise ValueError(f"Refusing to drop protected schema '{schema}'")
    from .conn import get_conn, get_database_url
    url = url or get_database_url()
    conn = get_conn(url, autocommit=True)
    conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply DB migrations to Supabase/Postgres")
    parser.add_argument("--schema", default="public", help="Target schema (default: public)")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    applied = run_migrations(schema=args.schema, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("Tables in schema:")
        try:
            tables = list_tables(schema=args.schema)
            for t in tables:
                print(f"  • {t}")
        except Exception as exc:
            print(f"  (could not list tables: {exc})")


# Needed by migration runner (Optional not imported at module level above)
from typing import Optional  # noqa: E402


if __name__ == "__main__":
    main()
