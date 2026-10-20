"""Audit log writer for Layer 5b (app/db/audit.py).

write_audit() appends one immutable row to the audit_log table.
The table has a trigger that REJECTS UPDATE and DELETE, so rows are
permanently preserved for compliance.

Design:
- The function is a no-op if Postgres is unavailable, so earlier layers
  that call it work fine without the DB backend.
- All values are parameterized (no SQL interpolation of user data).
- output_hash is the SHA-256 of the model output so we can detect
  tampering post-hoc.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hash_output(output: str) -> str:
    """Deterministic SHA-256 of the output string."""
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def write_audit(
    user_id: Optional[str],
    action: str,
    query: Optional[str] = None,
    plan: Optional[dict[str, Any]] = None,
    retrieved_span_ids: Optional[list[str]] = None,
    output_hash: Optional[str] = None,
    model_versions: Optional[dict[str, str]] = None,
    url: Optional[str] = None,
    schema: str = "public",
) -> Optional[int]:
    """Append one row to audit_log and return its id.

    Args:
        user_id:             Identifier of the user or service making the call.
        action:              Short action label, e.g. 'search', 'query', 'sync'.
        query:               The user's natural-language or Cypher query (if any).
        plan:                Retrieval / reasoning plan as a dict.
        retrieved_span_ids:  Span IDs of retrieved evidence chunks.
        output_hash:         SHA-256 of the LLM/query output for tamper detection.
        model_versions:      Dict of {model_role: version_string}.
        url:                 Override DATABASE_URL.
        schema:              Postgres schema to write to.

    Returns:
        The new row id, or None if Postgres is unavailable.
    """
    try:
        from .conn import get_conn, get_database_url, is_postgres_available
        if not is_postgres_available(url):
            return None
        conn = get_conn(url or get_database_url(), schema=schema)
    except Exception as exc:
        logger.debug("audit_log: skipping (no DB): %s", exc)
        return None

    try:
        cur = conn.execute(
            """
            INSERT INTO audit_log
                (ts, user_id, action, query, plan, retrieved_span_ids,
                 output_hash, model_versions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                datetime.now(timezone.utc),
                user_id,
                action,
                query,
                json.dumps(plan) if plan else None,
                retrieved_span_ids or [],
                output_hash,
                json.dumps(model_versions) if model_versions else None,
            ),
        )
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    except Exception as exc:
        logger.warning("audit_log: write failed: %s", exc)
        conn.rollback()
        return None
    finally:
        conn.close()
