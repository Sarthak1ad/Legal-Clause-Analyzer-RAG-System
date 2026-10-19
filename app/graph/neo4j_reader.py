"""Neo4j-backed GraphReader implementation (app/graph/neo4j_reader.py).

Implements the same GraphReader interface as InMemoryGraphReader but
executes Cypher queries against a live Neo4j database.

Design decisions:
- All Cypher queries are parameterized (no string interpolation of values).
- Namespace-scoped: only returns nodes whose ns matches self.namespace,
  so test namespaces cannot bleed into production reads.
- Results are sorted identically to InMemoryGraphReader to allow byte-level
  comparison in parity tests.
- Lazy import of neo4j so the module loads cleanly without the package.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from .reader import GraphReader
from .neo4j_store import Neo4jStore


class Neo4jGraphReader(GraphReader):
    """GraphReader backed by a live Neo4j database.

    All methods execute parameterized Cypher queries via Neo4jStore.query().
    The ``namespace`` parameter scopes all reads to the namespace used when
    loading (default ``"main"``).
    """

    def __init__(
        self,
        store: Optional[Neo4jStore] = None,
        namespace: str = "main",
    ) -> None:
        """
        Args:
            store:     An open Neo4jStore instance.  If None, a new one is
                       created from environment variables.
            namespace: Which namespace to read from (must match the one
                       used during load_graph()).
        """
        if store is None:
            store = Neo4jStore(namespace=namespace)
        self._store = store
        self._ns = namespace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _props_from_row(self, row: dict, node_key: str = "n") -> dict[str, Any]:
        """Extract node properties from a query row, removing internal keys."""
        props = dict(row.get(node_key) or {})
        # Remove internal namespacing props from returned properties
        props.pop("ns", None)
        props.pop("_original_id", None)
        # Restore the original (un-namespaced) id so callers see "doc-id" not "main:doc-id"
        if "id" in props:
            raw = props["id"]
            prefix = f"{self._ns}:"
            if raw.startswith(prefix):
                props["id"] = raw[len(prefix):]
        return props

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a Cypher query with namespace injected and return rows."""
        params["_ns"] = self._ns
        return self._store.query(cypher, params)

    def _strip_ns(self, ns_id: str) -> str:
        """Remove namespace prefix from a node id."""
        prefix = f"{self._ns}:"
        return ns_id[len(prefix):] if ns_id.startswith(prefix) else ns_id

    # ------------------------------------------------------------------
    # Contract queries
    # ------------------------------------------------------------------

    def contracts(self) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Contract {ns: $_ns}) "
            "RETURN c ORDER BY c.doc_id"
        )
        result = []
        for r in rows:
            p = self._props_from_row(r, "c")
            result.append(p)
        return result

    def contract(self, doc_id: str) -> Optional[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Contract {ns: $_ns, doc_id: $doc_id}) RETURN c",
            doc_id=doc_id,
        )
        return self._props_from_row(rows[0], "c") if rows else None

    # ------------------------------------------------------------------
    # Clause queries
    # ------------------------------------------------------------------

    def clauses_of(
        self,
        doc_id: str,
        as_of: Optional[str] = None,
        status: Optional[Literal["active", "superseded"]] = "active",
    ) -> list[dict[str, Any]]:
        # Build Cypher dynamically but with parameterized values only
        where_parts = ["c.doc_id = $doc_id", "c.ns = $_ns", "cl.ns = $_ns"]
        params: dict[str, Any] = {"doc_id": doc_id}

        if status is not None:
            where_parts.append("cl.status = $status")
            params["status"] = status

        if as_of is not None:
            # Exclude clauses whose effective_to is set and <= as_of
            where_parts.append(
                "(cl.effective_to IS NULL OR cl.effective_to > $as_of)"
            )
            params["as_of"] = as_of

        where_str = " AND ".join(where_parts)
        cypher = (
            f"MATCH (c:Contract)-[:CONTAINS]->(cl:Clause) "
            f"WHERE {where_str} "
            f"RETURN cl ORDER BY cl.clause_id"
        )
        rows = self._run(cypher, **params)
        return [self._props_from_row(r, "cl") for r in rows]

    def clause(self, clause_id: str) -> Optional[dict[str, Any]]:
        rows = self._run(
            "MATCH (cl:Clause {ns: $_ns, clause_id: $clause_id}) RETURN cl",
            clause_id=clause_id,
        )
        return self._props_from_row(rows[0], "cl") if rows else None

    def template_of(self, clause_id: str) -> Optional[dict[str, Any]]:
        rows = self._run(
            "MATCH (cl:Clause {ns: $_ns, clause_id: $clause_id})"
            "-[:INSTANCE_OF]->(t:StandardTemplate {ns: $_ns}) "
            "RETURN t",
            clause_id=clause_id,
        )
        return self._props_from_row(rows[0], "t") if rows else None

    # ------------------------------------------------------------------
    # Document hierarchy
    # ------------------------------------------------------------------

    def parent_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        # PARENT_OF goes parent -> child
        rows = self._run(
            "MATCH (parent:Contract {ns: $_ns})-[:PARENT_OF]->"
            "(child:Contract {ns: $_ns, doc_id: $doc_id}) "
            "RETURN parent",
            doc_id=doc_id,
        )
        return self._props_from_row(rows[0], "parent") if rows else None

    def children_of(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (parent:Contract {ns: $_ns, doc_id: $doc_id})"
            "-[:PARENT_OF]->(child:Contract {ns: $_ns}) "
            "RETURN child ORDER BY child.doc_id",
            doc_id=doc_id,
        )
        return [self._props_from_row(r, "child") for r in rows]

    # ------------------------------------------------------------------
    # Parties
    # ------------------------------------------------------------------

    def parties_of(self, doc_id: str) -> list[dict[str, Any]]:
        # PARTY_TO goes party -> contract
        rows = self._run(
            "MATCH (p:Party {ns: $_ns})-[:PARTY_TO]->"
            "(c:Contract {ns: $_ns, doc_id: $doc_id}) "
            "RETURN p ORDER BY p.party_id",
            doc_id=doc_id,
        )
        return [self._props_from_row(r, "p") for r in rows]

    # ------------------------------------------------------------------
    # Jurisdiction & Dates
    # ------------------------------------------------------------------

    def jurisdiction_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Contract {ns: $_ns, doc_id: $doc_id})"
            "-[:GOVERNED_BY]->(j:Jurisdiction {ns: $_ns}) "
            "RETURN j",
            doc_id=doc_id,
        )
        return self._props_from_row(rows[0], "j") if rows else None

    def dates_of(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Contract {ns: $_ns, doc_id: $doc_id})"
            "-[:HAS_DATE]->(d:DateEvent {ns: $_ns}) "
            "RETURN d ORDER BY d.kind",
            doc_id=doc_id,
        )
        return [self._props_from_row(r, "d") for r in rows]

    # ------------------------------------------------------------------
    # Portfolio / date-range queries
    # ------------------------------------------------------------------

    def contracts_expiring_between(
        self,
        start: str,
        end: str,
        which: Literal["current_term_end", "initial_term_end"] = "current_term_end",
    ) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Contract {ns: $_ns})-[:HAS_DATE]->"
            "(d:DateEvent {ns: $_ns, kind: $which}) "
            "WHERE d.date >= $start AND d.date <= $end "
            "RETURN c ORDER BY c.doc_id",
            which=which,
            start=start,
            end=end,
        )
        return [self._props_from_row(r, "c") for r in rows]

    # ------------------------------------------------------------------
    # Generic graph traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: Literal["out", "in", "both"] = "out",
    ) -> list[dict[str, Any]]:
        # Build directional pattern
        if direction == "out":
            pattern = "(src {ns: $_ns, _original_id: $node_id})-[r]->(nbr {ns: $_ns})"
        elif direction == "in":
            pattern = "(nbr {ns: $_ns})-[r]->(src {ns: $_ns, _original_id: $node_id})"
        else:
            pattern = "(src {ns: $_ns, _original_id: $node_id})-[r]-(nbr {ns: $_ns})"

        if edge_type:
            cypher = (
                f"MATCH {pattern} "
                f"WHERE type(r) = $edge_type "
                f"RETURN DISTINCT nbr"
            )
            rows = self._run(cypher, node_id=node_id, edge_type=edge_type)
        else:
            cypher = f"MATCH {pattern} RETURN DISTINCT nbr"
            rows = self._run(cypher, node_id=node_id)

        return [self._props_from_row(r, "nbr") for r in rows]
