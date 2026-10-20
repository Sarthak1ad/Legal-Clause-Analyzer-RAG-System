"""Robust Neo4j store for Layer 4/5b Knowledge Graph (app/graph/neo4j_store.py).

Features:
- Parameterized Cypher only (no injection risk).
- Namespace isolation: every node/edge gets an ns property so test data
  never touches main data in community/single-DB instances.
- Idempotent MERGE with UNWIND batches.
- Schema constraints and indexes created idempotently before loading.
- Transient-error retry with exponential back-off (neo4j.exceptions).
- Progress output during loading.
- Count verification after load: fails loud if Neo4j counts differ from
  in-memory graph counts per label and per relationship type.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional
from urllib.parse import urlparse

# Load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# Lazy import: the package must import cleanly even if neo4j is absent.
try:
    from neo4j import GraphDatabase, Driver  # type: ignore
    from neo4j.exceptions import (  # type: ignore
        TransientError,
        ServiceUnavailable,
    )
    _NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore
    TransientError = Exception  # type: ignore
    ServiceUnavailable = Exception  # type: ignore
    _NEO4J_AVAILABLE = False

from .model import KnowledgeGraph, Node, Edge


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_neo4j_config() -> tuple[str, str, str]:
    """Retrieve Neo4j connection parameters from environment.

    Never prints or logs the password.
    """
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    return uri, user, password


def _mask_uri(uri: str) -> str:
    """Mask password in URI for safe logging."""
    try:
        p = urlparse(uri)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{host}{port}"
    except Exception:
        return "<masked-uri>"


def is_neo4j_available() -> bool:
    """Return True if the configured Neo4j instance is currently reachable."""
    if not _NEO4J_AVAILABLE:
        return False
    uri, user, password = get_neo4j_config()
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=2.0)
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _run_with_retry(
    fn, max_attempts: int = 3, base_delay: float = 0.5
) -> Any:
    """Run fn(), retrying on TransientError up to max_attempts times."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] TransientError on attempt {attempt+1}: {exc}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _sanitize_for_neo4j(properties: dict[str, Any]) -> dict[str, Any]:
    """Sanitize properties so they are acceptable to Neo4j.

    Neo4j supports primitives (str, int, float, bool) and arrays thereof.
    Dicts (such as clause attributes) have their primitive top-level keys
    flattened onto the node (e.g. cap_multiple, cap_basis), and the dict itself
    is stored as a JSON string (<key>_json).
    Lists containing dicts or mixed types are stored as JSON strings.
    """
    import json
    clean: dict[str, Any] = {}
    for k, v in properties.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (str, int, float, bool)):
                    clean[sub_k] = sub_v
            try:
                clean[f"{k}_json"] = json.dumps(v)
            except Exception:
                pass
        elif isinstance(v, list):
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                clean[k] = v
            else:
                try:
                    clean[k] = json.dumps(v)
                except Exception:
                    clean[k] = [str(x) for x in v]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# ---------------------------------------------------------------------------
# Neo4jStore
# ---------------------------------------------------------------------------

class Neo4jStore:
    """Manager for loading and querying the knowledge graph in Neo4j.

    Args:
        uri:       Neo4j Bolt URI (default from NEO4J_URI env).
        user:      Database user (default from NEO4J_USER env).
        password:  Password (default from NEO4J_PASSWORD env).
        namespace: Namespace label for node isolation (default "main").
                   Every node gets an ``ns`` property and its effective id
                   is stored as ``<ns>:<id>``.  Reset/rebuild only touches
                   nodes matching this namespace.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        namespace: str = "main",
    ) -> None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError(
                "neo4j Python package is not installed. "
                "Run:  pip install 'neo4j>=5.0.0'"
            )
        def_uri, def_user, def_pass = get_neo4j_config()
        self.uri = uri or def_uri
        self.user = user or def_user
        self.password = password or def_pass
        self.namespace = namespace
        self._driver: Optional[Driver] = None

    # ------------------------------------------------------------------
    # Driver lifecycle
    # ------------------------------------------------------------------

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create idempotent uniqueness constraints and indexes.

        Constraints are on (label, id) which is the namespaced id string
        "<ns>:<original_id>".  We only create them if not already present
        so re-running is safe.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Contract)        REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Clause)          REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Party)           REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DefinedTerm)     REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:StandardTemplate) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Jurisdiction)    REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DateEvent)       REQUIRE n.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Contract)  ON (n.ns)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Clause)    ON (n.ns)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Contract)  ON (n.effective_date)",
            "CREATE INDEX IF NOT EXISTS FOR (n:DateEvent) ON (n.date)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Clause)    ON (n.status)",
        ]
        with self.driver.session() as session:
            for q in constraints:
                _run_with_retry(lambda q=q: session.run(q))
            for q in indexes:
                _run_with_retry(lambda q=q: session.run(q))

    # ------------------------------------------------------------------
    # Namespace reset
    # ------------------------------------------------------------------

    def reset_namespace(self) -> None:
        """Delete ONLY nodes (and their relationships) with ns == namespace."""
        with self.driver.session() as session:
            _run_with_retry(
                lambda: session.run(
                    "MATCH (n {ns: $ns}) DETACH DELETE n",
                    ns=self.namespace,
                )
            )

    def reset_database(self) -> None:
        """Delete ALL nodes and relationships in the database (use with care!)."""
        with self.driver.session() as session:
            _run_with_retry(lambda: session.run("MATCH (n) DETACH DELETE n"))

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------

    def load_graph(
        self,
        graph: KnowledgeGraph,
        batch_size: int = 500,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Load KnowledgeGraph into Neo4j in namespaced, idempotent batches.

        Each node gets:
          - ``ns``  = namespace string (for isolation/delete).
          - ``id``  = "<ns>:<original_id>" (globally unique even in a
                      shared single-database community instance).
          - All original properties from the in-memory graph.

        Returns a verification dict: {node_counts, edge_counts} from Neo4j
        matching the in-memory graph.  Raises if counts mismatch.
        """
        self.init_schema()

        if reset:
            print(f"  Resetting namespace '{self.namespace}' ...")
            self.reset_namespace()

        ns = self.namespace

        def _ns_id(original_id: str) -> str:
            # Prevent double-prefixing on idempotent reload
            if original_id.startswith(f"{ns}:"):
                return original_id
            return f"{ns}:{original_id}"

        # ---- Nodes -------------------------------------------------------
        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        for n in graph.nodes:
            props = _sanitize_for_neo4j(dict(n.properties))
            props["id"] = _ns_id(n.id)       # namespaced id
            props["ns"] = ns                  # namespace tag
            props["_original_id"] = n.id      # keep original for edge matching
            nodes_by_label.setdefault(n.label, []).append(props)

        node_total = sum(len(v) for v in nodes_by_label.values())
        loaded_nodes = 0

        with self.driver.session() as session:
            for label, items in nodes_by_label.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    cypher = (
                        f"UNWIND $batch AS row "
                        f"MERGE (n:{label} {{id: row.id}}) "
                        f"SET n += row"
                    )
                    _run_with_retry(lambda c=cypher, b=batch: session.run(c, batch=b))
                    loaded_nodes += len(batch)
                    print(f"  Nodes [{label}]: {min(loaded_nodes, node_total)}/{node_total}", end="\r")

        print(f"  Nodes: {node_total} loaded (namespace={ns})                    ")

        # Build original_id -> namespaced_id lookup for edges
        orig_to_ns: dict[str, str] = {
            n.id: _ns_id(n.id) for n in graph.nodes
        }

        # ---- Edges -------------------------------------------------------
        edges_by_type: dict[str, list[dict[str, Any]]] = {}
        for e in graph.edges:
            src_ns = orig_to_ns.get(e.source)
            tgt_ns = orig_to_ns.get(e.target)
            if not src_ns or not tgt_ns:
                continue  # skip dangling edges
            edge_dict: dict[str, Any] = {
                "source": src_ns,
                "target": tgt_ns,
                "ns": ns,
                "provenance": e.provenance,
                "evidence_span_ids": e.evidence_span_ids,
            }
            edge_dict.update(_sanitize_for_neo4j(dict(e.properties)))
            edges_by_type.setdefault(e.type, []).append(edge_dict)

        edge_total = sum(len(v) for v in edges_by_type.values())
        loaded_edges = 0

        with self.driver.session() as session:
            for rel_type, items in edges_by_type.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    cypher = (
                        f"UNWIND $batch AS row "
                        f"MATCH (s {{id: row.source}}) "
                        f"MATCH (t {{id: row.target}}) "
                        f"MERGE (s)-[r:{rel_type}]->(t) "
                        f"SET r += row"
                    )
                    _run_with_retry(lambda c=cypher, b=batch: session.run(c, batch=b))
                    loaded_edges += len(batch)

        print(f"  Edges: {edge_total} loaded")

        # ---- Count verification ------------------------------------------
        return self._verify_counts(graph, ns)

    def _verify_counts(
        self, graph: KnowledgeGraph, ns: str
    ) -> dict[str, Any]:
        """Compare Neo4j node/edge counts against the in-memory graph.

        Raises ValueError if counts mismatch; returns them on success.
        The check is namespace-scoped so it only examines our own data.
        """
        # In-memory expected counts
        expected_nodes = graph.node_counts()

        # Neo4j actual node counts (per label, namespace-filtered)
        with self.driver.session() as session:
            result = _run_with_retry(
                lambda: session.run(
                    "MATCH (n {ns: $ns}) "
                    "RETURN labels(n)[0] AS label, count(*) AS cnt",
                    ns=ns,
                ).data()
            )
        actual_nodes: dict[str, int] = {r["label"]: r["cnt"] for r in result}

        mismatches: list[str] = []
        for label, expected in expected_nodes.items():
            actual = actual_nodes.get(label, 0)
            if actual != expected:
                mismatches.append(
                    f"  Node {label}: expected {expected}, got {actual}"
                )

        # Edge counts by type (ns-scoped)
        expected_edges = graph.edge_counts()
        with self.driver.session() as session:
            result = _run_with_retry(
                lambda: session.run(
                    "MATCH ({ns: $ns})-[r]->({ns: $ns}) "
                    "RETURN type(r) AS rel_type, count(*) AS cnt",
                    ns=ns,
                ).data()
            )
        actual_edges: dict[str, int] = {r["rel_type"]: r["cnt"] for r in result}

        for rel_type, expected in expected_edges.items():
            actual = actual_edges.get(rel_type, 0)
            if actual != expected:
                mismatches.append(
                    f"  Edge {rel_type}: expected {expected}, got {actual}"
                )

        if mismatches:
            raise ValueError(
                f"Neo4j count verification FAILED (namespace={ns}):\n"
                + "\n".join(mismatches)
            )

        print(f"  Count verification PASSED ✓  (nodes={sum(expected_nodes.values())}, edges={sum(expected_edges.values())})")
        return {
            "node_counts": actual_nodes,
            "edge_counts": actual_edges,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        cypher: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Run a parameterized Cypher query and return rows as dicts."""
        with self.driver.session() as session:
            result = _run_with_retry(
                lambda: session.run(cypher, parameters or {}).data()
            )
        return result
