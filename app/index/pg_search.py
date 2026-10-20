"""Postgres-backed HybridSearcher (app/index/pg_search.py).

Implements the same search() signature as HybridSearcher but uses
PgKeywordSearch (websearch_to_tsquery + ts_rank_cd) and PgVectorIndex
(pgvector cosine distance).

RRF fusion happens in Python (identical logic to the local backend).
All filters are applied as SQL pre-filters before ranking.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

import numpy as np

from .embedder import Embedder
from .search import SearchHit, _make_snippet
from .pg_backend import (
    PgKeywordSearch,
    build_pg_filter,
    get_candidate_rowids_pg,
)


class PgHybridSearcher:
    """HybridSearcher backed by Postgres FTS + pgvector.

    Has the same .search() signature as HybridSearcher so callers can
    swap between backends transparently.
    """

    def __init__(
        self,
        conn: Any,
        embedder: Embedder,
        schema: str = "public",
    ) -> None:
        """
        Args:
            conn:     An open psycopg connection (get_conn() from app.db.conn).
            embedder: Embedder instance; must match what was used during sync.
            schema:   Postgres schema to search in.
        """
        self._conn = conn
        self._embedder = embedder
        self._schema = schema
        self._kw = PgKeywordSearch(conn, schema)
        self._verify_embedder()

    def _verify_embedder(self) -> None:
        """Raise if the embedder doesn't match what's in Postgres index_meta."""
        try:
            cur = self._conn.execute(
                "SELECT key, value FROM index_meta WHERE key IN ('embedder_name','embedder_dim')"
            )
            meta = {r[0]: r[1] for r in cur.fetchall()}
            stored_name = meta.get("embedder_name")
            stored_dim = meta.get("embedder_dim")
            if stored_name and (
                stored_name != self._embedder.name
                or int(stored_dim or "0") != self._embedder.dim
            ):
                raise ValueError(
                    f"Embedder mismatch: Postgres index built with '{stored_name}' "
                    f"(dim={stored_dim}), but caller provided '{self._embedder.name}' "
                    f"(dim={self._embedder.dim}).  Re-run: python -m app.db.sync"
                )
        except ValueError:
            raise
        except Exception:
            pass  # table might not exist yet; allow gracefully

    def search(
        self,
        query: str,
        k: int = 10,
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        filters: Optional[dict[str, Any]] = None,
        as_of: Optional[str] = None,
        user_groups: Optional[list[str]] = None,
    ) -> list[SearchHit]:
        """Search using Postgres FTS and/or pgvector, with RRF fusion in Python.

        Args match HybridSearcher.search() exactly for drop-in substitution.
        """
        flt = dict(filters or {})
        groups = list(user_groups or ["all"])

        # ---- Pre-filter: get candidate rowids from SQL ----
        candidate_rowids = get_candidate_rowids_pg(self._conn, flt, as_of, groups)
        if not candidate_rowids:
            return []

        keyword_ranks: dict[int, int] = {}
        keyword_scores: dict[int, float] = {}
        vector_ranks: dict[int, int] = {}
        vector_scores: dict[int, float] = {}

        # ---- Keyword search (Postgres FTS) ----
        if mode in ("hybrid", "keyword"):
            kw_hits = self._kw.search(query, candidate_rowids=candidate_rowids, k=100)
            for rank, (rid, score) in enumerate(kw_hits, start=1):
                keyword_ranks[rid] = rank
                keyword_scores[rid] = score

        # ---- Vector search (pgvector) ----
        if mode in ("hybrid", "vector"):
            q_emb = self._embedder.embed([query])[0]
            q = np.asarray(q_emb, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(q))
            if norm > 1e-12:
                q = q / norm

            vec_str = "[" + ",".join(str(x) for x in q.tolist()) + "]"

            # Use rowid filter + cosine distance
            if candidate_rowids:
                placeholders = ",".join(["%s"] * len(candidate_rowids))
                sql = (
                    f"SELECT rowid, 1 - (embedding <=> %s::extensions.vector) AS score "
                    f"FROM index_docs "
                    f"WHERE rowid IN ({placeholders}) AND embedding IS NOT NULL "
                    f"ORDER BY embedding <=> %s::extensions.vector LIMIT 100"
                )
                params: list[Any] = [vec_str] + list(candidate_rowids) + [vec_str]
            else:
                sql = (
                    "SELECT rowid, 1 - (embedding <=> %s::extensions.vector) AS score "
                    "FROM index_docs WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s::extensions.vector LIMIT 100"
                )
                params = [vec_str, vec_str]

            try:
                cur = self._conn.execute(sql, params)
                v_hits = [(int(r[0]), float(r[1])) for r in cur.fetchall()]
            except Exception:
                v_hits = []

            for rank, (rid, score) in enumerate(v_hits, start=1):
                vector_ranks[rid] = rank
                vector_scores[rid] = score

        # ---- RRF Fusion (identical to local HybridSearcher) ----
        all_rids = set(keyword_ranks.keys()) | set(vector_ranks.keys())
        if not all_rids:
            return []

        k_rrf = 60
        rrf_scores: dict[int, float] = {}
        for rid in all_rids:
            if mode == "hybrid":
                score = 0.0
                if rid in vector_ranks:
                    score += 1.0 / (k_rrf + vector_ranks[rid])
                if rid in keyword_ranks:
                    score += 1.0 / (k_rrf + keyword_ranks[rid])
                rrf_scores[rid] = score
            elif mode == "vector":
                rrf_scores[rid] = vector_scores.get(rid, 0.0)
            else:
                rrf_scores[rid] = keyword_scores.get(rid, 0.0)

        sorted_rids = sorted(rrf_scores, key=lambda r: rrf_scores[r], reverse=True)[:k]

        # ---- Fetch doc metadata ----
        hits: list[SearchHit] = []
        if sorted_rids:
            placeholders = ",".join(["%s"] * len(sorted_rids))
            cur = self._conn.execute(
                f"SELECT rowid, doc_key, kind, ref_id, contract_id, doc_type, "
                f"clause_type, section_path, heading, text, span_id, status "
                f"FROM index_docs WHERE rowid IN ({placeholders})",
                sorted_rids,
            )
            doc_map = {int(r[0]): r for r in cur.fetchall()}

            for rid in sorted_rids:
                row = doc_map.get(rid)
                if not row:
                    continue
                hits.append(SearchHit(
                    clause_id=row[3],
                    contract_id=row[4],
                    doc_type=row[5],
                    clause_type=row[6],
                    section_path=row[7],
                    heading=row[8],
                    span_id=row[10],
                    text=row[9] or "",
                    snippet=_make_snippet(row[9] or "", query),
                    status=row[11] or "active",
                    kind=row[2],
                    doc_key=row[1],
                    attributes={},
                    vector_rank=vector_ranks.get(rid),
                    keyword_rank=keyword_ranks.get(rid),
                    vector_score=vector_scores.get(rid),
                    keyword_score=keyword_scores.get(rid),
                    rrf_score=rrf_scores[rid],
                ))

        return hits
