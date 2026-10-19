"""Neo4j store for Layer 4 Knowledge Graph (app/graph/neo4j_store.py).

Features:
- Parameterized Cypher only (prevents injection).
- Batch loading via UNWIND.
- MERGE on unique ID constraints for full idempotency.
- Schema constraints & indexes created before loading.
- Safe connection checking so system works offline without Neo4j running.
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore

from .model import KnowledgeGraph, Node, Edge


def get_neo4j_config() -> tuple[str, str, str]:
    """Retrieve Neo4j connection parameters from environment or local defaults."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    return uri, user, password


def is_neo4j_available() -> bool:
    """Test whether the Neo4j instance is reachable."""
    if GraphDatabase is None:
        return False
    uri, user, password = get_neo4j_config()
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=1.0)
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


class Neo4jStore:
    """Manager for loading and querying the knowledge graph in Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if GraphDatabase is None:
            raise RuntimeError("neo4j python package is not installed.")
        def_uri, def_user, def_pass = get_neo4j_config()
        self.uri = uri or def_uri
        self.user = user or def_user
        self.password = password or def_pass
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def reset_database(self):
        """Delete all nodes, relationships, and constraints in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def init_schema(self):
        """Create unique ID constraints and indexes for all node types."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Party) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:DefinedTerm) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:StandardTemplate) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DateEvent) REQUIRE d.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Contract) ON (c.effective_date)",
            "CREATE INDEX IF NOT EXISTS FOR (d:DateEvent) ON (d.date)",
        ]
        with self.driver.session() as session:
            for q in constraints:
                session.run(q)
            for q in indexes:
                session.run(q)

    def load_graph(self, graph: KnowledgeGraph, batch_size: int = 500):
        """Load KnowledgeGraph nodes and edges into Neo4j in idempotent UNWIND batches."""
        self.init_schema()

        # 1. Load Nodes grouped by label
        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        for n in graph.nodes:
            # Flatten or format properties safely
            props = dict(n.properties)
            props["id"] = n.id
            nodes_by_label.setdefault(n.label, []).append(props)

        with self.driver.session() as session:
            for label, items in nodes_by_label.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    cypher = (
                        f"UNWIND $batch AS row "
                        f"MERGE (n:{label} {{id: row.id}}) "
                        f"SET n += row"
                    )
                    session.run(cypher, batch=batch)

            # 2. Load Edges grouped by type
            edges_by_type: dict[str, list[dict[str, Any]]] = {}
            for e in graph.edges:
                edge_dict = {
                    "source": e.source,
                    "target": e.target,
                    "provenance": e.provenance,
                    "evidence_span_ids": e.evidence_span_ids,
                }
                edge_dict.update(e.properties)
                edges_by_type.setdefault(e.type, []).append(edge_dict)

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
                    session.run(cypher, batch=batch)

    def query(self, cypher: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Run a parameterized Cypher query and return rows as dictionaries."""
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]
