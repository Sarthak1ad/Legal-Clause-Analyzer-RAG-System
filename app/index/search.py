"""Hybrid vector + keyword search over contract clauses and templates (app/index/search.py).

Features:
- Modes: 'hybrid' (RRF fusion), 'vector' (cosine), 'keyword' (BM25 via FTS5).
- Reciprocal Rank Fusion (RRF) with k=60.
- Strict pre-filtering inside retrieval (filters applied before ranking):
  doc_type, clause_type, contract_id(s), party_id, status (default active), kind, acl_groups.
- ACL permission gating via user_groups (default: ['all']).
- Time-aware effective-state filtering via as_of date.
- Safe query sanitization supporting quoted exact phrases (e.g. '"gross negligence"').
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from .embedder import Embedder
from .store import IndexStore


@dataclass
class SearchHit:
    """A single search result with rank/score breakdown."""
    clause_id: str
    contract_id: Optional[str]
    doc_type: Optional[str]
    clause_type: Optional[str]
    section_path: Optional[str]
    heading: Optional[str]
    span_id: Optional[str]
    text: str
    snippet: str
    status: str
    kind: str
    doc_key: str
    attributes: dict[str, Any] = field(default_factory=dict)
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: float = 0.0


def sanitize_fts5_query(query: str) -> str:
    """Safely sanitize user input for SQLite FTS5.

    Preserves quoted phrases like '"gross negligence"' for exact matching,
    while escaping special characters in bare words.
    """
    if not query or not query.strip():
        return ""

    tokens: list[str] = []
    # Match "quoted phrases" or individual words
    for match in re.finditer(r'"([^"]+)"|([A-Za-z0-9_\-.]+)', query):
        quoted, bare = match.groups()
        if quoted:
            # Clean inner quotes
            clean = quoted.replace('"', '""').strip()
            if clean:
                tokens.append(f'"{clean}"')
        elif bare:
            # Strip non-alphanumeric
            clean = re.sub(r'[^A-Za-z0-9_]', '', bare).strip()
            if clean:
                tokens.append(f'"{clean}"')

    return " ".join(tokens)


def _make_snippet(text: str, query: str, max_len: int = 160) -> str:
    """Generate a readable snippet containing query keywords if possible."""
    clean_text = " ".join(text.split())
    if len(clean_text) <= max_len:
        return clean_text

    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
    lower = clean_text.lower()
    first_pos = -1
    for t in tokens:
        p = lower.find(t)
        if p != -1 and (first_pos == -1 or p < first_pos):
            first_pos = p

    if first_pos == -1 or first_pos < 40:
        return clean_text[:max_len] + "..."

    start = max(0, first_pos - 40)
    end = min(len(clean_text), start + max_len)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return prefix + clean_text[start:end] + suffix


class HybridSearcher:
    """Executes keyword, vector, and hybrid search over IndexStore."""

    def __init__(self, store: IndexStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self.store.verify_embedder(embedder)

    def _get_candidate_rowids(
        self,
        filters: Optional[dict[str, Any]] = None,
        as_of: Optional[str] = None,
        user_groups: Optional[list[str]] = None,
    ) -> set[int]:
        """Pre-filtering: retrieve matching rowids satisfying all filter criteria."""
        clauses: list[str] = []
        params: list[Any] = []

        flt = dict(filters or {})

        # Kind filter (default: clauses and templates)
        if "kind" in flt and flt["kind"]:
            kind_val = flt["kind"]
            if isinstance(kind_val, (list, tuple, set)):
                placeholders = ",".join(["?"] * len(kind_val))
                clauses.append(f"kind IN ({placeholders})")
                params.extend(list(kind_val))
            else:
                clauses.append("kind = ?")
                params.append(kind_val)

        # Status filter (default to 'active' if not explicitly overridden)
        status_val = flt.get("status", "active")
        if status_val and status_val != "all":
            clauses.append("status = ?")
            params.append(status_val)

        # Doc type filter
        if "doc_type" in flt and flt["doc_type"]:
            dt = flt["doc_type"]
            if isinstance(dt, (list, tuple, set)):
                placeholders = ",".join(["?"] * len(dt))
                clauses.append(f"doc_type IN ({placeholders})")
                params.extend(list(dt))
            else:
                clauses.append("doc_type = ?")
                params.append(dt)

        # Clause type filter
        if "clause_type" in flt and flt["clause_type"]:
            ct = flt["clause_type"]
            if isinstance(ct, (list, tuple, set)):
                placeholders = ",".join(["?"] * len(ct))
                clauses.append(f"clause_type IN ({placeholders})")
                params.extend(list(ct))
            else:
                clauses.append("clause_type = ?")
                params.append(ct)

        # Contract ID filter (single or multiple)
        cid = flt.get("contract_id") or flt.get("contract_ids")
        if cid:
            if isinstance(cid, (list, tuple, set)):
                placeholders = ",".join(["?"] * len(cid))
                clauses.append(f"contract_id IN ({placeholders})")
                params.extend(list(cid))
            else:
                clauses.append("contract_id = ?")
                params.append(cid)

        # Effective as_of filter (time-aware)
        if as_of:
            clauses.append("(effective_from IS NULL OR effective_from <= ?)")
            params.append(as_of)
            clauses.append("(effective_to IS NULL OR ? < effective_to)")
            params.append(as_of)

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT rowid, party_ids, acl_groups FROM docs {where_sql}"

        cur = self.store._conn.execute(query, params)
        rows = cur.fetchall()

        # In-Python filtering for JSON arrays (party_id, acl_groups)
        valid_rowids: set[int] = set()
        req_party = flt.get("party_id")
        allowed_groups = set(user_groups or ["all"])

        for r in rows:
            rid = r["rowid"]

            # ACL Check: document must share at least one group with user_groups
            try:
                doc_groups = set(json.loads(r["acl_groups"]))
            except Exception:
                doc_groups = {"all"}

            if not ("admin" in allowed_groups or bool(doc_groups & allowed_groups)):
                continue

            # Party filter
            if req_party:
                try:
                    pids = json.loads(r["party_ids"])
                    if req_party not in pids:
                        continue
                except Exception:
                    continue

            valid_rowids.add(rid)

        return valid_rowids

    def search(
        self,
        query: str,
        k: int = 10,
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        filters: Optional[dict[str, Any]] = None,
        as_of: Optional[str] = None,
        user_groups: Optional[list[str]] = None,
    ) -> list[SearchHit]:
        """Search clauses, templates, and summaries with pre-filtering and score breakdown."""
        # Step 1: Pre-filtering
        candidate_rowids = self._get_candidate_rowids(
            filters=filters, as_of=as_of, user_groups=user_groups
        )
        if not candidate_rowids:
            return []

        keyword_ranks: dict[int, int] = {}
        keyword_scores: dict[int, float] = {}
        vector_ranks: dict[int, int] = {}
        vector_scores: dict[int, float] = {}

        # Step 2: Keyword search
        if mode in ("hybrid", "keyword"):
            clean_q = sanitize_fts5_query(query)
            kw_hits: list[tuple[int, float]] = []

            if clean_q and self.store.has_fts5:
                try:
                    cur = self.store._conn.cursor()
                    fts_sql = """
                        SELECT rowid, bm25(docs_fts) as score
                        FROM docs_fts
                        WHERE docs_fts MATCH ?
                        ORDER BY score ASC
                    """
                    for r in cur.execute(fts_sql, (clean_q,)):
                        rid = r[0]
                        # bm25 in SQLite FTS5 returns negative score (lower is better)
                        score = float(-r[1])
                        if rid in candidate_rowids:
                            kw_hits.append((rid, score))
                            if len(kw_hits) >= 100:
                                break
                except Exception:
                    pass

            if not kw_hits and self.store.bm25_fallback is not None:
                kw_hits = self.store.bm25_fallback.search(query, rowid_filter=candidate_rowids, k=100)

            for rank, (rid, score) in enumerate(kw_hits, start=1):
                keyword_ranks[rid] = rank
                keyword_scores[rid] = score

        # Step 3: Vector search
        if mode in ("hybrid", "vector"):
            q_emb = self.embedder.embed([query])[0]
            v_hits = self.store.vectors.search(q_emb, rowid_filter=candidate_rowids, k=100)
            for rank, (rid, score) in enumerate(v_hits, start=1):
                vector_ranks[rid] = rank
                vector_scores[rid] = score

        # Step 4: Fusion / Ranking
        all_candidate_rids = set(keyword_ranks.keys()) | set(vector_ranks.keys())
        if not all_candidate_rids:
            return []

        rrf_scores: dict[int, float] = {}
        k_rrf = 60

        for rid in all_candidate_rids:
            if mode == "hybrid":
                score = 0.0
                if rid in vector_ranks:
                    score += 1.0 / (k_rrf + vector_ranks[rid])
                if rid in keyword_ranks:
                    score += 1.0 / (k_rrf + keyword_ranks[rid])
                rrf_scores[rid] = score
            elif mode == "vector":
                rrf_scores[rid] = vector_scores.get(rid, 0.0)
            else:  # keyword
                rrf_scores[rid] = keyword_scores.get(rid, 0.0)

        # Sort descending by final score
        sorted_rids = sorted(rrf_scores.keys(), key=lambda r: rrf_scores[r], reverse=True)[:k]

        # Fetch doc metadata
        hits: list[SearchHit] = []
        if sorted_rids:
            placeholders = ",".join(["?"] * len(sorted_rids))
            cur = self.store._conn.execute(
                f"SELECT rowid, doc_key, kind, ref_id, contract_id, doc_type, "
                f"clause_type, section_path, heading, text, span_id, status, "
                f"attributes_json FROM docs WHERE rowid IN ({placeholders})",
                sorted_rids,
            )
            doc_map = {r["rowid"]: r for r in cur.fetchall()}

            for rid in sorted_rids:
                row = doc_map.get(rid)
                if not row:
                    continue
                attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
                hits.append(SearchHit(
                    clause_id=row["ref_id"],
                    contract_id=row["contract_id"],
                    doc_type=row["doc_type"],
                    clause_type=row["clause_type"],
                    section_path=row["section_path"],
                    heading=row["heading"],
                    span_id=row["span_id"],
                    text=row["text"],
                    snippet=_make_snippet(row["text"], query),
                    status=row["status"],
                    kind=row["kind"],
                    doc_key=row["doc_key"],
                    attributes=attrs,
                    vector_rank=vector_ranks.get(rid),
                    keyword_rank=keyword_ranks.get(rid),
                    vector_score=vector_scores.get(rid),
                    keyword_score=keyword_scores.get(rid),
                    rrf_score=rrf_scores[rid],
                ))

        return hits
