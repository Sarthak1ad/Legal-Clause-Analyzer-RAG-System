"""Neo4j connection check CLI (app/graph/neo4j_check.py).

Usage:
    python -m app.graph.neo4j_check

Prints server version/edition, detects Aura vs local, masks credentials.
Fails with a clear message for:
  - wrong password (AuthError)
  - unreachable host (ServiceUnavailable)
  - wrong URI scheme
"""
from __future__ import annotations

import os
import re
import sys
from urllib.parse import urlparse

# Load .env file if present (idempotent; no error if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional at import time


def _mask_uri(uri: str) -> str:
    """Return URI with password redacted, showing only scheme and host."""
    try:
        p = urlparse(uri)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{host}{port}/***"
    except Exception:
        return "<masked>"


def _is_aura(uri: str) -> bool:
    """Detect AuraDB by the neo4j+s:// or neo4j+ssc:// scheme."""
    return uri.lower().startswith(("neo4j+s://", "neo4j+ssc://"))


def check_neo4j_connection() -> int:
    """Run the connection check; return 0 on success, 1 on failure."""
    # ------------------------------------------------------------------ #
    # 1. Read credentials (never print them)                               #
    # ------------------------------------------------------------------ #
    required_keys = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        print("[ERROR] The following environment variables are required but not set:")
        for k in missing:
            print(f"  • {k}")
        print("\nCreate a .env file from .env.example and fill in the values.")
        print("Required keys:")
        print("  NEO4J_URI        bolt://localhost:7687  or  neo4j+s://<id>.databases.neo4j.io")
        print("  NEO4J_USER       neo4j")
        print("  NEO4J_PASSWORD   (your password)")
        return 1

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]

    masked_uri = _mask_uri(uri)
    db_kind = "AuraDB (cloud)" if _is_aura(uri) else "Local / Desktop"

    print(f"Neo4j connection check")
    print(f"  URI  : {masked_uri}  [{db_kind}]")
    print(f"  User : {user}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Validate URI scheme before attempting connection                  #
    # ------------------------------------------------------------------ #
    valid_schemes = ("bolt://", "bolt+s://", "bolt+ssc://",
                     "neo4j://", "neo4j+s://", "neo4j+ssc://")
    if not any(uri.lower().startswith(s) for s in valid_schemes):
        print(f"[ERROR] Invalid URI scheme in NEO4J_URI: {masked_uri}")
        print("  Valid schemes: bolt://, bolt+s://, neo4j://, neo4j+s://")
        print("  Example (local):  bolt://localhost:7687")
        print("  Example (Aura):   neo4j+s://<id>.databases.neo4j.io")
        return 1

    # ------------------------------------------------------------------ #
    # 3. Attempt connection (lazy import so the package works without neo4j) #
    # ------------------------------------------------------------------ #
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        print("[ERROR] The 'neo4j' Python package is not installed.")
        print("  Install it with:  pip install neo4j>=5.0.0")
        return 1

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=5.0,
        )
        driver.verify_connectivity()
    except Exception as exc:
        exc_str = str(exc)
        exc_type = type(exc).__name__

        # Map common exceptions to actionable messages
        if "AuthError" in exc_type or "authentication" in exc_str.lower() or "unauthorized" in exc_str.lower():
            print(f"[ERROR] Authentication failed for user '{user}'.")
            print("  Check NEO4J_PASSWORD in your .env file.")
        elif "ServiceUnavailable" in exc_type or "Connection refused" in exc_str or "Failed to" in exc_str:
            print(f"[ERROR] Cannot reach Neo4j at {masked_uri}.")
            print("  Is the database running?")
            print("  For Desktop: start the DBMS in Neo4j Desktop.")
            print("  For Docker:  docker-compose up -d")
            print("  For Aura:    check your AuraDB instance is started.")
        elif "CertificateError" in exc_str or "SSL" in exc_str:
            print(f"[ERROR] TLS/SSL error connecting to {masked_uri}.")
            print("  For Aura, use neo4j+s:// not bolt://.")
        else:
            print(f"[ERROR] {exc_type}: {exc_str[:200]}")

        return 1

    # ------------------------------------------------------------------ #
    # 4. Query server info                                                 #
    # ------------------------------------------------------------------ #
    try:
        with driver.session() as session:
            info = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition").single()
            version = info["versions"][0] if info and info["versions"] else "?"
            edition = (info["edition"] or "").capitalize() if info else "?"
    except Exception:
        version = "unknown"
        edition = "unknown"
    finally:
        driver.close()

    print(f"[OK] Connected successfully!")
    print(f"  Server : Neo4j {version} ({edition} edition)")
    print(f"  Host   : {masked_uri}")
    print(f"  Type   : {db_kind}")

    if edition.lower() == "community":
        print()
        print("  [NOTE] Community edition has a single database.")
        print("         Use --namespace to keep test and production data separate.")

    return 0


def main() -> None:
    sys.exit(check_neo4j_connection())


if __name__ == "__main__":
    main()
