"""Review queue CLI (app/db/review.py).

Manages human review of flagged clauses/documents.

Usage:
    python -m app.db.review list [--status open] [--schema public]
    python -m app.db.review resolve <id> --by NAME [--note TEXT]
    python -m app.db.review dismiss <id> [--by NAME]
    python -m app.db.review override --target-type clause --target-id ID
        --field F --new-value JSON --by NAME [--reason TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass


def _get_conn(schema: str) -> Any:
    from .conn import get_conn, get_database_url
    return get_conn(get_database_url(), schema=schema)


def cmd_list(args: argparse.Namespace) -> None:
    conn = _get_conn(args.schema)
    status_filter = args.status or "open"

    if status_filter == "all":
        cur = conn.execute(
            "SELECT id, source_layer, doc_id, clause_id, severity, reason, status, created_at "
            "FROM review_queue ORDER BY created_at DESC LIMIT 100"
        )
    else:
        cur = conn.execute(
            "SELECT id, source_layer, doc_id, clause_id, severity, reason, status, created_at "
            "FROM review_queue WHERE status = %s ORDER BY created_at DESC LIMIT 100",
            (status_filter,),
        )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No review items with status='{status_filter}'.")
        return

    print(f"\nREVIEW QUEUE  (status={status_filter}, {len(rows)} item(s))")
    print(f"  {'ID':>5}  {'Sev':<8}  {'Layer':<10}  {'Doc':<20}  {'Reason'}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r[0]:>5}  {r[4]:<8}  {r[1]:<10}  {str(r[2] or ''):<20}  {r[5]}")


def cmd_resolve(args: argparse.Namespace) -> None:
    conn = _get_conn(args.schema)
    note = args.note or ""
    conn.execute(
        """
        UPDATE review_queue
           SET status='resolved', resolved_by=%s, resolved_at=NOW(),
               details = details || jsonb_build_object('note', %s)
         WHERE id=%s AND status='open'
        """,
        (args.by, note, args.id),
    )
    if conn.pgconn.cmd_tuples == 0:  # type: ignore[attr-defined]
        print(f"[WARN] Item {args.id} not found or not open.")
    else:
        print(f"Resolved item {args.id} by {args.by}.")
    conn.commit()
    conn.close()


def cmd_dismiss(args: argparse.Namespace) -> None:
    conn = _get_conn(args.schema)
    by = getattr(args, "by", "anonymous")
    conn.execute(
        "UPDATE review_queue SET status='dismissed', resolved_by=%s, resolved_at=NOW() "
        "WHERE id=%s AND status='open'",
        (by, args.id),
    )
    print(f"Dismissed item {args.id}.")
    conn.commit()
    conn.close()


def cmd_override(args: argparse.Namespace) -> None:
    conn = _get_conn(args.schema)
    try:
        new_val = json.loads(args.new_value)
    except json.JSONDecodeError:
        new_val = args.new_value  # treat as string

    conn.execute(
        """
        INSERT INTO review_overrides
            (target_type, target_id, field, old_value, new_value, reviewer, reason, applied)
        VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
        """,
        (args.target_type, args.target_id, args.field,
         None,           # old_value: could be fetched, left NULL here for simplicity
         json.dumps(new_val), args.by, getattr(args, "reason", "")),
    )
    conn.commit()
    cur = conn.execute("SELECT lastval()")
    new_id = cur.fetchone()[0]
    conn.close()
    print(f"Override recorded (id={new_id}): {args.target_type}/{args.target_id}.{args.field} -> {new_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review queue and override manager")
    parser.add_argument("--schema", default="public")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List review items")
    p_list.add_argument("--status", default="open", help="Filter by status (open/resolved/dismissed/all)")

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve a review item")
    p_resolve.add_argument("id", type=int)
    p_resolve.add_argument("--by", required=True, help="Reviewer name")
    p_resolve.add_argument("--note", default="", help="Resolution note")

    # dismiss
    p_dismiss = sub.add_parser("dismiss", help="Dismiss a review item")
    p_dismiss.add_argument("id", type=int)
    p_dismiss.add_argument("--by", default="anonymous")

    # override
    p_override = sub.add_parser("override", help="Record a human override")
    p_override.add_argument("--target-type", required=True)
    p_override.add_argument("--target-id", required=True)
    p_override.add_argument("--field", required=True)
    p_override.add_argument("--new-value", required=True, help="JSON value")
    p_override.add_argument("--by", required=True, help="Reviewer name")
    p_override.add_argument("--reason", default="")

    args = parser.parse_args()

    from .conn import is_postgres_available
    if not is_postgres_available():
        print("[ERROR] Postgres is not available. Check DATABASE_URL in .env.")
        sys.exit(1)

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "resolve":
        cmd_resolve(args)
    elif args.cmd == "dismiss":
        cmd_dismiss(args)
    elif args.cmd == "override":
        cmd_override(args)


if __name__ == "__main__":
    main()
