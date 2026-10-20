"""Postgres connection helper for Supabase/pgvector backend (app/db/conn.py).

Design decisions:
- Uses psycopg 3 (psycopg[binary]) for async-compatible connection.
- prepare_threshold=None disables prepared statements so the connection
  works through Supabase's Session or Transaction pooler.
- sslmode=require for Supabase; localhost connections relax this.
- URL is masked in error messages (host visible, password hidden).
- pgvector types registered at connection time via the pgvector package.
- Lazy imports: the module loads cleanly even if psycopg/pgvector are absent.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

# Load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# ---- Sentinel so callers can check availability without import errors -------
_PSYCOPG_AVAILABLE = False
try:
    import psycopg  # type: ignore
    _PSYCOPG_AVAILABLE = True
except ImportError:
    pass

_PGVECTOR_AVAILABLE = False
try:
    import pgvector.psycopg  # type: ignore  # noqa: F401
    _PGVECTOR_AVAILABLE = True
except ImportError:
    pass


def _mask_url(url: str) -> str:
    """Return URL with password replaced by ***."""
    try:
        p = urlparse(url)
        masked_netloc = f"{p.hostname}" + (f":{p.port}" if p.port else "")
        if p.username:
            masked_netloc = f"{p.username}:***@{masked_netloc}"
        return f"{p.scheme}://{masked_netloc}{p.path}"
    except Exception:
        # Regex fallback
        return re.sub(r"(?<=:)[^:@]+(?=@)", "***", url)


def _is_local(url: str) -> bool:
    """True if the host is localhost or 127.x.x.x (relaxes sslmode)."""
    try:
        host = urlparse(url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1")
    except Exception:
        return False


def get_database_url() -> str:
    """Return DATABASE_URL from environment, raising clearly if missing."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set.\n"
            "Set it in your .env file to the Supabase connection string.\n"
            "Use the Session pooler string from Supabase Dashboard → Settings → Database\n"
            "if the direct host is unreachable:\n"
            "  postgresql://postgres.<ref>:<pass>@aws-0-<region>.pooler.supabase.com:5432/postgres"
        )
    return url


def get_conn(
    url: Optional[str] = None,
    schema: str = "public",
    autocommit: bool = False,
) -> "psycopg.Connection":  # type: ignore[name-defined]
    """Open a psycopg 3 connection to Postgres.

    Args:
        url:         Connection string (default: DATABASE_URL from env).
        schema:      Postgres search_path to set (default: public).
        autocommit:  Whether to run in autocommit mode.

    Raises:
        RuntimeError: DATABASE_URL not set, or psycopg not installed.
        psycopg.OperationalError: Connection failure (message is masked).
    """
    if not _PSYCOPG_AVAILABLE:
        raise RuntimeError(
            "psycopg (psycopg[binary]) is not installed.\n"
            "Run:  pip install 'psycopg[binary]>=3'"
        )

    url = url or get_database_url()
    masked = _mask_url(url)

    # Supabase requires SSL; localhost does not.
    sslmode = "prefer" if _is_local(url) else "require"

    # Build connection kwargs
    conn_kwargs: dict[str, Any] = {
        "prepare_threshold": None,   # disable prepared stmts for pooler compat
        "autocommit": autocommit,
    }

    # Add sslmode if not already in the URL
    if "sslmode=" not in url:
        conn_kwargs["sslmode"] = sslmode

    try:
        conn = psycopg.connect(url, **conn_kwargs)
    except Exception as exc:
        exc_str = str(exc)
        # Mask the password if it leaked into the exception message
        exc_masked = re.sub(r"(?<=:)[^:@]+(?=@)", "***", exc_str)
        raise type(exc)(
            f"Failed to connect to Postgres at {masked}: {exc_masked}\n"
            "If the direct host is unreachable, try the Supabase Session pooler "
            "connection string from Dashboard → Project Settings → Database."
        ) from None

    # Set search_path
    if schema != "public":
        conn.execute(f"SET search_path TO {schema}, public, extensions")
    else:
        conn.execute("SET search_path TO public, extensions")

    # Register pgvector types if available
    if _PGVECTOR_AVAILABLE:
        pgvector.psycopg.register_vector(conn)

    return conn


def is_postgres_available(url: Optional[str] = None) -> bool:
    """Return True if a Postgres connection can be established."""
    if not _PSYCOPG_AVAILABLE:
        return False
    try:
        url = url or os.environ.get("DATABASE_URL", "")
        if not url:
            return False
        conn = get_conn(url)
        conn.close()
        return True
    except Exception:
        return False
