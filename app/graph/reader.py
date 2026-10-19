"""Offline GraphReader interface and in-memory implementation (Layer 4 Part 0d).

Provides a clean, stable API for Layer 5 (indexes, search, communities) and
any future layers that need to query the contract knowledge graph without
touching the raw graph JSON or Neo4j directly.

Design decisions:
- GraphReader is abstract; callers are decoupled from storage.
- InMemoryGraphReader wraps a KnowledgeGraph built by build_graph().
- All methods are deterministic and offline-first (no network calls).
- as_of filtering applies only to Clause nodes with effective_from/effective_to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Literal, Optional

from .dates import parse_iso_date
from .model import (
    ClauseNode,
    KnowledgeGraph,
    Node,
)


class GraphReader(ABC):
    """Abstract interface for reading the contract knowledge graph.

    All methods return plain dicts or typed model objects so that callers
    do not need to know the underlying storage backend.
    """

    # ------------------------------------------------------------------
    # Contract queries
    # ------------------------------------------------------------------

    @abstractmethod
    def contracts(self) -> list[dict[str, Any]]:
        """Return all contract node property dicts, sorted by doc_id."""

    @abstractmethod
    def contract(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Return properties of a single contract node, or None."""

    # ------------------------------------------------------------------
    # Clause queries
    # ------------------------------------------------------------------

    @abstractmethod
    def clauses_of(
        self,
        doc_id: str,
        as_of: Optional[str] = None,
        status: Optional[Literal["active", "superseded"]] = "active",
    ) -> list[dict[str, Any]]:
        """Return clause property dicts for a contract.

        Args:
            doc_id:  Target contract.
            as_of:   ISO date string; if supplied, exclude clauses whose
                     effective_to is set and is <= as_of (i.e. superseded
                     before this date).  None means return all.
            status:  Filter by status field ("active", "superseded", or None
                     for no filter).
        """

    @abstractmethod
    def clause(self, clause_id: str) -> Optional[dict[str, Any]]:
        """Return properties of a single clause node, or None."""

    @abstractmethod
    def template_of(self, clause_id: str) -> Optional[dict[str, Any]]:
        """Return properties of the StandardTemplate the clause is an instance of, or None."""

    # ------------------------------------------------------------------
    # Document hierarchy
    # ------------------------------------------------------------------

    @abstractmethod
    def parent_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Return properties of the parent contract (via PARENT_OF edge), or None."""

    @abstractmethod
    def children_of(self, doc_id: str) -> list[dict[str, Any]]:
        """Return property dicts for all child contracts (SOWs, amendments)."""

    # ------------------------------------------------------------------
    # Parties
    # ------------------------------------------------------------------

    @abstractmethod
    def parties_of(self, doc_id: str) -> list[dict[str, Any]]:
        """Return party property dicts linked to the contract via PARTY_TO edges."""

    # ------------------------------------------------------------------
    # Jurisdiction & Dates
    # ------------------------------------------------------------------

    @abstractmethod
    def jurisdiction_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Return Jurisdiction node properties for the contract, or None."""

    @abstractmethod
    def dates_of(self, doc_id: str) -> list[dict[str, Any]]:
        """Return DateEvent property dicts linked to the contract via HAS_DATE edges."""

    # ------------------------------------------------------------------
    # Portfolio / date-range queries
    # ------------------------------------------------------------------

    @abstractmethod
    def contracts_expiring_between(
        self,
        start: str,
        end: str,
        which: Literal["current_term_end", "initial_term_end"] = "current_term_end",
    ) -> list[dict[str, Any]]:
        """Return contract property dicts whose chosen date event falls in [start, end].

        Args:
            start: ISO date string (inclusive).
            end:   ISO date string (inclusive).
            which: Which date kind to filter on.
        """

    # ------------------------------------------------------------------
    # Generic graph traversal
    # ------------------------------------------------------------------

    @abstractmethod
    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: Literal["out", "in", "both"] = "out",
    ) -> list[dict[str, Any]]:
        """Return property dicts of neighboring nodes.

        Args:
            node_id:    Source node id.
            edge_type:  Optional edge type filter (e.g. "REFERENCES").
            direction:  "out" (source==node_id), "in" (target==node_id),
                        or "both".
        """


class InMemoryGraphReader(GraphReader):
    """GraphReader backed by a KnowledgeGraph object built by build_graph().

    Builds lookup indexes on construction so every query is O(1) or O(k)
    where k is the result set size.
    """

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> InMemoryGraphReader:
        """Load a KnowledgeGraph from a file path or JSON string and return a reader."""
        from pathlib import Path
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            graph = KnowledgeGraph.model_validate_json(content)
        else:
            graph = KnowledgeGraph.model_validate_json(str(path_or_str))
        return cls(graph)

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

        # Index: id -> Node
        self._nodes: dict[str, Node] = {n.id: n for n in graph.nodes}

        # Index: label -> list[Node]
        self._by_label: dict[str, list[Node]] = {}
        for n in graph.nodes:
            self._by_label.setdefault(n.label, []).append(n)

        # Index: edge type -> list[Edge]
        from .model import Edge  # local import to avoid circular at module level
        self._by_edge_type: dict[str, list] = {}
        for e in graph.edges:
            self._by_edge_type.setdefault(e.type, []).append(e)

        # Reverse index: (source, edge_type) -> list[target_id]
        self._out_edges: dict[tuple[str, str], list[str]] = {}
        # Reverse index: (target, edge_type) -> list[source_id]
        self._in_edges: dict[tuple[str, str], list[str]] = {}
        for e in graph.edges:
            self._out_edges.setdefault((e.source, e.type), []).append(e.target)
            self._in_edges.setdefault((e.target, e.type), []).append(e.source)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _props(self, node_id: str) -> Optional[dict[str, Any]]:
        n = self._nodes.get(node_id)
        return dict(n.properties) if n else None

    def _props_list(self, ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for nid in ids:
            p = self._props(nid)
            if p is not None:
                out.append(p)
        return out

    def _out_neighbors(self, node_id: str, edge_type: str) -> list[str]:
        return list(self._out_edges.get((node_id, edge_type), []))

    def _in_neighbors(self, node_id: str, edge_type: str) -> list[str]:
        return list(self._in_edges.get((node_id, edge_type), []))

    # ------------------------------------------------------------------
    # Contract queries
    # ------------------------------------------------------------------

    def contracts(self) -> list[dict[str, Any]]:
        nodes = sorted(self._by_label.get("Contract", []), key=lambda n: n.id)
        return [dict(n.properties) for n in nodes]

    def contract(self, doc_id: str) -> Optional[dict[str, Any]]:
        return self._props(doc_id)

    # ------------------------------------------------------------------
    # Clause queries
    # ------------------------------------------------------------------

    def clauses_of(
        self,
        doc_id: str,
        as_of: Optional[str] = None,
        status: Optional[Literal["active", "superseded"]] = "active",
    ) -> list[dict[str, Any]]:
        clause_ids = self._out_neighbors(doc_id, "CONTAINS")
        results: list[dict[str, Any]] = []
        as_of_date = parse_iso_date(as_of) if as_of else None

        for cid in sorted(clause_ids):
            props = self._props(cid)
            if props is None:
                continue
            if props.get("label") == "Contract":
                # CONTAINS can link to sub-contracts in some models; skip
                continue

            # Status filter
            if status is not None and props.get("status") != status:
                continue

            # Time-aware filter: skip clauses superseded before as_of
            if as_of_date is not None:
                eff_to = parse_iso_date(props.get("effective_to"))
                if eff_to is not None and eff_to <= as_of_date:
                    continue

            results.append(props)

        return results

    def clause(self, clause_id: str) -> Optional[dict[str, Any]]:
        return self._props(clause_id)

    def template_of(self, clause_id: str) -> Optional[dict[str, Any]]:
        tpl_ids = self._out_neighbors(clause_id, "INSTANCE_OF")
        if not tpl_ids:
            return None
        return self._props(tpl_ids[0])

    # ------------------------------------------------------------------
    # Document hierarchy
    # ------------------------------------------------------------------

    def parent_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        # PARENT_OF edge goes from parent -> child
        parent_ids = self._in_neighbors(doc_id, "PARENT_OF")
        if not parent_ids:
            return None
        return self._props(parent_ids[0])

    def children_of(self, doc_id: str) -> list[dict[str, Any]]:
        child_ids = self._out_neighbors(doc_id, "PARENT_OF")
        return self._props_list(sorted(child_ids))

    # ------------------------------------------------------------------
    # Parties
    # ------------------------------------------------------------------

    def parties_of(self, doc_id: str) -> list[dict[str, Any]]:
        # PARTY_TO goes from party -> contract
        party_ids = self._in_neighbors(doc_id, "PARTY_TO")
        return self._props_list(sorted(party_ids))

    # ------------------------------------------------------------------
    # Jurisdiction & Dates
    # ------------------------------------------------------------------

    def jurisdiction_of(self, doc_id: str) -> Optional[dict[str, Any]]:
        jur_ids = self._out_neighbors(doc_id, "GOVERNED_BY")
        if not jur_ids:
            return None
        return self._props(jur_ids[0])

    def dates_of(self, doc_id: str) -> list[dict[str, Any]]:
        date_ids = self._out_neighbors(doc_id, "HAS_DATE")
        return self._props_list(sorted(date_ids))

    # ------------------------------------------------------------------
    # Portfolio / date-range queries
    # ------------------------------------------------------------------

    def contracts_expiring_between(
        self,
        start: str,
        end: str,
        which: Literal["current_term_end", "initial_term_end"] = "current_term_end",
    ) -> list[dict[str, Any]]:
        start_d = parse_iso_date(start)
        end_d = parse_iso_date(end)
        if not start_d or not end_d:
            return []

        results: list[dict[str, Any]] = []
        for c_node in self._by_label.get("Contract", []):
            doc_id = c_node.id
            for date_props in self.dates_of(doc_id):
                if date_props.get("kind") != which:
                    continue
                d = parse_iso_date(date_props.get("date"))
                if d and start_d <= d <= end_d:
                    results.append(dict(c_node.properties))
                    break

        return sorted(results, key=lambda p: p.get("doc_id", ""))

    # ------------------------------------------------------------------
    # Generic graph traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: Literal["out", "in", "both"] = "out",
    ) -> list[dict[str, Any]]:
        neighbor_ids: list[str] = []

        edge_types = [edge_type] if edge_type else list(set(
            [e.type for e in self._graph.edges]
        ))

        for et in edge_types:
            if direction in ("out", "both"):
                neighbor_ids.extend(self._out_neighbors(node_id, et))
            if direction in ("in", "both"):
                neighbor_ids.extend(self._in_neighbors(node_id, et))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for nid in neighbor_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)

        return self._props_list(unique_ids)
