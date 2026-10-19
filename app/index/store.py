"""Index store: SQLite (metadata + docs + FTS5) + Numpy VectorIndex (app/index/store.py).

Provides:
- VectorIndex (ABC) and NumpyVectorIndex for swappable vector storage.
- KeywordIndex (ABC) with SQLite FTS5 implementation and pure-Python BM25 fallback.
- IndexStore: manages indexing of Clause nodes, StandardTemplate nodes, and Community summaries.
- Incremental update support via text_hash and full --rebuild.
- Strict embedder validation against stored metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..graph.reader import GraphReader
from .embedder import Embedder

DEFAULT_INDEX_DIR = Path("data/index")


def compute_text_hash(text: str, extra: str = "") -> str:
    """MD5 hash of combined text and attributes text for incremental updates."""
    content = f"{text}\n---\n{extra}".encode("utf-8")
    return hashlib.md5(content).hexdigest()


def render_attributes_text(clause_type: Optional[str], attributes: dict[str, Any]) -> str:
    """Render extracted attributes into plain search-friendly text.

    e.g. 'indemnity cap 2.0x annual fees; one-way; covers gross negligence'
    """
    if not attributes:
        return ""

    parts: list[str] = []

    cap = attributes.get("cap_multiple")
    if cap is not None:
        basis = attributes.get("cap_basis", "annual fees")
        parts.append(f"indemnity liability cap {cap}x {basis}")

    direction = attributes.get("direction")
    if direction:
        parts.append(direction.replace("_", "-"))
    if attributes.get("mutual") is True:
        parts.append("mutual obligations")

    if attributes.get("covers_gross_negligence") is True:
        parts.append("covers gross negligence and willful misconduct")
    elif attributes.get("covers_gross_negligence") is False:
        parts.append("excludes gross negligence")

    if attributes.get("excludes_indirect_damages") is True:
        parts.append("excludes indirect or consequential damages")

    if attributes.get("auto_renew") is True:
        parts.append("renews automatically")
    elif attributes.get("auto_renew") is False:
        parts.append("does not renew automatically")

    renew_yrs = attributes.get("renewal_term_years")
    if renew_yrs:
        parts.append(f"successive renewal term {renew_yrs} years")

    notice_days = attributes.get("non_renewal_notice_days")
    if notice_days:
        parts.append(f"non-renewal notice {notice_days} days")

    pay_days = attributes.get("payment_days")
    if pay_days:
        parts.append(f"payment within {pay_days} days of invoice")

    surv_yrs = attributes.get("survival_years")
    if surv_yrs:
        parts.append(f"obligation survives for {surv_yrs} years after termination")

    term_yrs = attributes.get("term_years")
    if term_yrs:
        parts.append(f"initial term {term_yrs} years")

    # Include other string attributes
    for k, v in attributes.items():
        if k not in (
            "cap_multiple", "cap_basis", "direction", "mutual",
            "covers_gross_negligence", "excludes_indirect_damages",
            "auto_renew", "renewal_term_years", "non_renewal_notice_days",
            "payment_days", "survival_years", "term_years", "spans"
        ) and isinstance(v, (str, int, float, bool)):
            parts.append(f"{k} {v}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Vector Index Interface & Numpy Implementation
# ---------------------------------------------------------------------------

class VectorIndex(ABC):
    """Abstract interface for vector retrieval (can be replaced by pgvector in Layer 5b)."""

    @abstractmethod
    def add(self, rowids: list[int], vectors: np.ndarray) -> None:
        """Add or update vectors for specified rowids."""

    @abstractmethod
    def search(
        self, query_vec: np.ndarray, rowid_filter: Optional[set[int]] = None, k: int = 10
    ) -> list[tuple[int, float]]:
        """Return list of (rowid, cosine_score) sorted descending."""

    @abstractmethod
    def delete(self, rowids: list[int]) -> None:
        """Remove vectors for rowids."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist vector index to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load vector index from disk."""


class NumpyVectorIndex(VectorIndex):
    """File-backed brute-force cosine vector index using numpy arrays."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._rowids: list[int] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._rowid_to_idx: dict[int, int] = {}

    def add(self, rowids: list[int], vectors: np.ndarray) -> None:
        if len(rowids) == 0:
            return
        if vectors.shape[0] != len(rowids):
            raise ValueError("Rowids length must match vectors count")

        for rid, vec in zip(rowids, vectors):
            if rid in self._rowid_to_idx:
                idx = self._rowid_to_idx[rid]
                self._vectors[idx] = vec
            else:
                idx = len(self._rowids)
                self._rowids.append(rid)
                self._rowid_to_idx[rid] = idx
                if self._vectors.shape[0] == 0:
                    self._vectors = np.expand_dims(vec, axis=0)
                else:
                    self._vectors = np.vstack([self._vectors, np.expand_dims(vec, axis=0)])

    def delete(self, rowids: list[int]) -> None:
        to_remove = set(rowids)
        keep_indices = [i for i, r in enumerate(self._rowids) if r not in to_remove]
        self._rowids = [self._rowids[i] for i in keep_indices]
        self._vectors = self._vectors[keep_indices] if keep_indices else np.zeros((0, self.dim), dtype=np.float32)
        self._rowid_to_idx = {r: i for i, r in enumerate(self._rowids)}

    def search(
        self, query_vec: np.ndarray, rowid_filter: Optional[set[int]] = None, k: int = 10
    ) -> list[tuple[int, float]]:
        if len(self._rowids) == 0:
            return []

        # Cosine similarity (vectors are L2-normalized)
        q = np.asarray(query_vec, dtype=np.float32).flatten()
        q_norm = float(np.linalg.norm(q))
        if q_norm > 1e-12:
            q = q / q_norm

        scores = self._vectors @ q

        results: list[tuple[int, float]] = []
        for idx, rowid in enumerate(self._rowids):
            if rowid_filter is not None and rowid not in rowid_filter:
                continue
            results.append((rowid, float(scores[idx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "vectors.npy", self._vectors)
        np.save(dir_path / "vector_rowids.npy", np.array(self._rowids, dtype=np.int64))

    def load(self, dir_path: Path) -> None:
        vec_path = dir_path / "vectors.npy"
        rid_path = dir_path / "vector_rowids.npy"
        if vec_path.exists() and rid_path.exists():
            self._vectors = np.load(vec_path).astype(np.float32)
            self._rowids = np.load(rid_path).tolist()
            self._rowid_to_idx = {r: i for i, r in enumerate(self._rowids)}
            if self._vectors.shape[0] > 0:
                self.dim = int(self._vectors.shape[1])


# ---------------------------------------------------------------------------
# Pure Python BM25 Fallback (used only if SQLite FTS5 is not compiled)
# ---------------------------------------------------------------------------

class PythonBM25Index:
    """Pure-Python BM25 fallback when SQLite FTS5 is unavailable."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lens: dict[int, int] = {}
        self.avg_len = 0.0
        self.inverted: dict[str, dict[int, int]] = {}
        self.num_docs = 0

    def index(self, docs: list[tuple[int, str]]) -> None:
        self.num_docs = len(docs)
        total_len = 0
        self.inverted.clear()
        self.doc_lens.clear()

        for rowid, text in docs:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            length = len(tokens)
            self.doc_lens[rowid] = length
            total_len += length
            for t in tokens:
                if t not in self.inverted:
                    self.inverted[t] = {}
                self.inverted[t][rowid] = self.inverted[t].get(rowid, 0) + 1

        self.avg_len = total_len / max(self.num_docs, 1)

    def search(
        self, query: str, rowid_filter: Optional[set[int]] = None, k: int = 10
    ) -> list[tuple[int, float]]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        scores: dict[int, float] = {}

        for t in tokens:
            postings = self.inverted.get(t)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            for rid, freq in postings.items():
                if rowid_filter is not None and rid not in rowid_filter:
                    continue
                dlen = self.doc_lens.get(rid, 1)
                tf_norm = freq * (self.k1 + 1.0) / (freq + self.k1 * (1.0 - self.b + self.b * (dlen / max(self.avg_len, 1e-6))))
                scores[rid] = scores.get(rid, 0.0) + idf * tf_norm

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------

class IndexStore:
    """SQLite + VectorIndex store managing contract documents, keywords, and vectors."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "index.db"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self.has_fts5 = self._check_fts5()
        self.bm25_fallback: Optional[PythonBM25Index] = None
        self._init_db()
        self.vectors = NumpyVectorIndex()
        self.vectors.load(self.index_dir)

    def _check_fts5(self) -> bool:
        try:
            cur = self._conn.cursor()
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_check USING fts5(x)")
            cur.execute("DROP TABLE _fts_check")
            return True
        except Exception:
            return False

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_key TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL,
                ref_id TEXT NOT NULL,
                contract_id TEXT,
                doc_type TEXT,
                clause_type TEXT,
                section_path TEXT,
                heading TEXT,
                text TEXT NOT NULL,
                attributes_text TEXT NOT NULL,
                span_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                effective_from TEXT,
                effective_to TEXT,
                party_ids TEXT NOT NULL DEFAULT '[]',
                acl_groups TEXT NOT NULL DEFAULT '["all"]',
                attributes_json TEXT NOT NULL DEFAULT '{}',
                text_hash TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_kind ON docs(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_contract ON docs(contract_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_clause_type ON docs(clause_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON docs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_text_hash ON docs(text_hash)")

        if self.has_fts5:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                    heading,
                    text,
                    attributes_text,
                    content='docs',
                    content_rowid='rowid'
                )
            """)
            # Triggers to keep FTS in sync with docs
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                    INSERT INTO docs_fts(rowid, heading, text, attributes_text)
                    VALUES (new.rowid, new.heading, new.text, new.attributes_text);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, heading, text, attributes_text)
                    VALUES ('delete', old.rowid, old.heading, old.text, old.attributes_text);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, heading, text, attributes_text)
                    VALUES ('delete', old.rowid, old.heading, old.text, old.attributes_text);
                    INSERT INTO docs_fts(rowid, heading, text, attributes_text)
                    VALUES (new.rowid, new.heading, new.text, new.attributes_text);
                END;
            """)
        else:
            print("[INFO] SQLite FTS5 not available. Initializing pure-Python BM25 fallback.")
            self.bm25_fallback = PythonBM25Index()
            # Populate fallback if docs exist
            rows = self._conn.execute("SELECT rowid, heading, text, attributes_text FROM docs").fetchall()
            corpus = [(r["rowid"], f"{r['heading'] or ''} {r['text']} {r['attributes_text']}") for r in rows]
            self.bm25_fallback.index(corpus)

        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        cur = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))
        self._conn.commit()

    def verify_embedder(self, embedder: Embedder) -> None:
        """Refuse to query or update if embedder doesn't match the one used at build time."""
        stored_name = self.get_meta("embedder_name")
        stored_dim = self.get_meta("embedder_dim")
        if stored_name and (stored_name != embedder.name or int(stored_dim or "0") != embedder.dim):
            raise ValueError(
                f"Embedder mismatch! Index was built with embedder '{stored_name}' (dim={stored_dim}), "
                f"but caller provided '{embedder.name}' (dim={embedder.dim}). "
                f"Please rebuild the index using: python -m app.index.run build --rebuild"
            )

    def build_from_graph(
        self,
        reader: GraphReader,
        embedder: Embedder,
        acl_map: Optional[dict[str, list[str]]] = None,
        rebuild: bool = False,
        communities_file: Optional[Path] = None,
    ) -> dict[str, int]:
        """Build or incrementally update the index from GraphReader.

        Returns stats dict: {total_docs, added, updated, unchanged}.
        """
        if rebuild:
            self._conn.execute("DELETE FROM docs")
            if self.has_fts5:
                self._conn.execute("DELETE FROM docs_fts")
            self.vectors = NumpyVectorIndex(dim=embedder.dim)
            self._conn.commit()
        else:
            self.verify_embedder(embedder)

        # Load ACL map
        acl = acl_map or {}
        acl_file = Path("data/acl.json")
        if not acl and acl_file.exists():
            try:
                acl = json.loads(acl_file.read_text(encoding="utf-8"))
            except Exception:
                acl = {}

        # Collect candidate items to index
        items: list[dict[str, Any]] = []

        # Load parsed docs for clause text
        data_dir = Path("data")
        from ..graph.build import load_parsed_docs
        pdocs = load_parsed_docs(data_dir)
        span_to_text = {cl.span.span_id: cl.text for pdoc in pdocs.values() for cl in pdoc.clauses}

        # 1. All Clauses from all contracts
        contracts = reader.contracts()
        for c in contracts:
            doc_id = c["doc_id"]
            doc_acl = acl.get(doc_id, ["all"])
            party_ids = [p["id"] for p in reader.parties_of(doc_id)]
            clauses = reader.clauses_of(doc_id, as_of=None, status=None)
            for cl in clauses:
                cid = cl.get("id", "")
                span_id = cl.get("span_id", "")
                text = cl.get("text") or span_to_text.get(span_id, "")
                heading = cl.get("heading") or ""
                ctype = cl.get("clause_type") or "other"
                attrs = cl.get("attributes") or {}
                attrs_text = render_attributes_text(ctype, attrs)
                items.append({
                    "doc_key": f"clause:{cid}",
                    "kind": "clause",
                    "ref_id": cid,
                    "contract_id": doc_id,
                    "doc_type": c.get("doc_type"),
                    "clause_type": ctype,
                    "section_path": cl.get("section_path") or "",
                    "heading": heading,
                    "text": text,
                    "attributes_text": attrs_text,
                    "span_id": cl.get("span_id"),
                    "status": cl.get("status", "active"),
                    "effective_from": cl.get("effective_from"),
                    "effective_to": cl.get("effective_to"),
                    "party_ids": party_ids,
                    "acl_groups": doc_acl,
                    "attributes_json": attrs,
                    "text_hash": compute_text_hash(text, attrs_text),
                })

        # 2. Standard Templates
        from ..playbook.loader import load_playbook
        try:
            pb = load_playbook()
            for t_obj in pb.templates:
                t_name = t_obj.clause_type
                t_text = t_obj.standard_text or ""
                items.append({
                    "doc_key": f"template:{t_name}",
                    "kind": "template",
                    "ref_id": t_name,
                    "contract_id": None,
                    "doc_type": None,
                    "clause_type": t_name,
                    "section_path": "",
                    "heading": f"Standard Playbook Template: {t_name}",
                    "text": t_text,
                    "attributes_text": "",
                    "span_id": None,
                    "status": "active",
                    "effective_from": None,
                    "effective_to": None,
                    "party_ids": [],
                    "acl_groups": ["all"],
                    "attributes_json": {},
                    "text_hash": compute_text_hash(t_text),
                })
        except Exception:
            pass

        # 3. Community summaries (if communities_file given or exists)
        c_path = communities_file or (self.index_dir / "communities.json")
        if c_path.exists():
            try:
                comm_data = json.loads(c_path.read_text(encoding="utf-8"))
                for comm in comm_data.get("communities", []):
                    cid = comm["id"]
                    text = comm.get("narrative") or comm.get("description") or ""
                    stats_text = "; ".join(f"{k}: {v}" for k, v in comm.get("stats", {}).items())
                    combined = f"{text}\n\nStats:\n{stats_text}"
                    items.append({
                        "doc_key": f"summary:{cid}",
                        "kind": "summary",
                        "ref_id": cid,
                        "contract_id": None,
                        "doc_type": None,
                        "clause_type": comm.get("kind"),
                        "section_path": "",
                        "heading": comm.get("name") or f"Community Summary: {cid}",
                        "text": combined,
                        "attributes_text": stats_text,
                        "span_id": None,
                        "status": "active",
                        "effective_from": None,
                        "effective_to": None,
                        "party_ids": [],
                        "acl_groups": comm.get("permission_groups") or ["all"],
                        "attributes_json": comm.get("stats") or {},
                        "text_hash": compute_text_hash(combined, stats_text),
                    })
            except Exception:
                pass

        # Process items: detect new / changed / unchanged
        existing_rows = {
            r["doc_key"]: (r["rowid"], r["text_hash"])
            for r in self._conn.execute("SELECT rowid, doc_key, text_hash FROM docs").fetchall()
        }

        to_add: list[dict[str, Any]] = []
        to_update: list[tuple[int, dict[str, Any]]] = []
        unchanged = 0

        for item in items:
            key = item["doc_key"]
            h = item["text_hash"]
            if key in existing_rows:
                rid, old_h = existing_rows[key]
                if old_h != h:
                    to_update.append((rid, item))
                else:
                    unchanged += 1
            else:
                to_add.append(item)

        # Batch embed texts for added + updated
        embed_items = to_add + [it for _, it in to_update]
        if embed_items:
            texts_to_embed = [
                f"{it['heading']}\n{it['text']}\n{it['attributes_text']}".strip()
                for it in embed_items
            ]
            new_vectors = embedder.embed(texts_to_embed)
        else:
            new_vectors = np.zeros((0, embedder.dim), dtype=np.float32)

        cur = self._conn.cursor()

        # Handle updates
        vec_idx = 0
        update_rids: list[int] = []
        update_vecs: list[np.ndarray] = []
        for rid, item in to_update:
            cur.execute("""
                UPDATE docs SET
                    kind = ?, ref_id = ?, contract_id = ?, doc_type = ?,
                    clause_type = ?, section_path = ?, heading = ?, text = ?,
                    attributes_text = ?, span_id = ?, status = ?,
                    effective_from = ?, effective_to = ?, party_ids = ?,
                    acl_groups = ?, attributes_json = ?, text_hash = ?
                WHERE rowid = ?
            """, (
                item["kind"], item["ref_id"], item["contract_id"], item["doc_type"],
                item["clause_type"], item["section_path"], item["heading"], item["text"],
                item["attributes_text"], item["span_id"], item["status"],
                item["effective_from"], item["effective_to"],
                json.dumps(item["party_ids"]), json.dumps(item["acl_groups"]),
                json.dumps(item["attributes_json"]), item["text_hash"],
                rid
            ))
            update_rids.append(rid)
            update_vecs.append(new_vectors[vec_idx])
            vec_idx += 1

        if update_rids:
            self.vectors.add(update_rids, np.array(update_vecs, dtype=np.float32))

        # Handle inserts
        add_rids: list[int] = []
        add_vecs: list[np.ndarray] = []
        for item in to_add:
            cur.execute("""
                INSERT INTO docs (
                    doc_key, kind, ref_id, contract_id, doc_type,
                    clause_type, section_path, heading, text,
                    attributes_text, span_id, status, effective_from,
                    effective_to, party_ids, acl_groups, attributes_json,
                    text_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["doc_key"], item["kind"], item["ref_id"], item["contract_id"],
                item["doc_type"], item["clause_type"], item["section_path"],
                item["heading"], item["text"], item["attributes_text"],
                item["span_id"], item["status"], item["effective_from"],
                item["effective_to"], json.dumps(item["party_ids"]),
                json.dumps(item["acl_groups"]), json.dumps(item["attributes_json"]),
                item["text_hash"]
            ))
            rid = cur.lastrowid
            add_rids.append(rid)
            add_vecs.append(new_vectors[vec_idx])
            vec_idx += 1

        if add_rids:
            self.vectors.add(add_rids, np.array(add_vecs, dtype=np.float32))

        self._conn.commit()

        # Update pure-Python BM25 if active
        if self.bm25_fallback is not None:
            rows = self._conn.execute("SELECT rowid, heading, text, attributes_text FROM docs").fetchall()
            corpus = [(r["rowid"], f"{r['heading'] or ''} {r['text']} {r['attributes_text']}") for r in rows]
            self.bm25_fallback.index(corpus)

        # Save vectors and metadata
        self.vectors.save(self.index_dir)
        self.set_meta("embedder_name", embedder.name)
        self.set_meta("embedder_dim", str(embedder.dim))
        self.set_meta("total_docs", str(len(items)))

        return {
            "total_docs": len(items),
            "added": len(to_add),
            "updated": len(to_update),
            "unchanged": unchanged,
        }
