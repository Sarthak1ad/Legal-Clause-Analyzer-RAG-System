-- 001_init.sql
-- Initial schema for contract-graph Supabase backend.
--
-- This file is idempotent: every DDL statement uses IF NOT EXISTS or
-- OR REPLACE so running it multiple times is safe.
--
-- Assumes pgvector extension is already enabled in the 'extensions' schema
-- (Supabase enables it by default there).  The migration runner enables it
-- before applying this file.

-- ---------------------------------------------------------------------------
-- schema_migrations: tracks which migration files have been applied.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- documents: one row per (doc_id, version) pair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT        NOT NULL,
    version         INTEGER     NOT NULL DEFAULT 1,
    sha256          TEXT,
    doc_type        TEXT,
    title           TEXT,
    effective_date  DATE,
    status          TEXT        NOT NULL DEFAULT 'active',
    source_file     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (doc_id, version)
);

-- ---------------------------------------------------------------------------
-- clauses: extracted clause records with effective-state fields.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clauses (
    clause_id       TEXT        PRIMARY KEY,
    doc_id          TEXT        NOT NULL,
    version         INTEGER     NOT NULL DEFAULT 1,
    section_path    TEXT,
    heading         TEXT,
    clause_type     TEXT,
    span_id         TEXT,
    text            TEXT,
    text_hash       TEXT,
    status          TEXT        NOT NULL DEFAULT 'active',
    effective_from  DATE,
    effective_to    DATE,
    needs_review    BOOLEAN     NOT NULL DEFAULT FALSE,
    attributes      JSONB       NOT NULL DEFAULT '{}'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clauses_doc     ON clauses(doc_id, version);
CREATE INDEX IF NOT EXISTS idx_clauses_type    ON clauses(clause_type);
CREATE INDEX IF NOT EXISTS idx_clauses_status  ON clauses(status);
CREATE INDEX IF NOT EXISTS idx_clauses_review  ON clauses(needs_review) WHERE needs_review = TRUE;

-- ---------------------------------------------------------------------------
-- index_meta: stores which embedder was used to build the index.
-- Refusing to insert/search with a mismatched embedder prevents silent
-- garbage results when someone switches models.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS index_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- index_docs: the main full-text + vector search table.
-- <DIM> is a placeholder; the migration runner substitutes the real value
-- (384 for bge-small, 512 for hashing) based on the active embedder.
-- ---------------------------------------------------------------------------
-- NOTE: The vector column dimension is set by the runner; this file uses
-- a sentinel value that the runner replaces before executing.
CREATE TABLE IF NOT EXISTS index_docs (
    doc_key         TEXT        PRIMARY KEY,
    kind            TEXT        NOT NULL CHECK (kind IN ('clause','template','summary')),
    ref_id          TEXT        NOT NULL,
    contract_id     TEXT,
    doc_type        TEXT,
    clause_type     TEXT,
    section_path    TEXT,
    heading         TEXT,
    text            TEXT        NOT NULL DEFAULT '',
    span_id         TEXT,
    status          TEXT        NOT NULL DEFAULT 'active',
    effective_from  DATE,
    effective_to    DATE,
    party_ids       JSONB       NOT NULL DEFAULT '[]'::JSONB,
    acl_groups      TEXT[]      NOT NULL DEFAULT ARRAY['all'],
    attributes_text TEXT        NOT NULL DEFAULT '',
    text_hash       TEXT        NOT NULL DEFAULT '',
    embedder        TEXT,
    -- embedding column added dynamically in 002 once dim is known
    fts             TSVECTOR    GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(heading, '') || ' ' ||
            coalesce(text, '') || ' ' ||
            coalesce(attributes_text, '')
        )
    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_idocs_fts          ON index_docs USING GIN(fts);
CREATE INDEX IF NOT EXISTS idx_idocs_contract     ON index_docs(contract_id);
CREATE INDEX IF NOT EXISTS idx_idocs_clause_type  ON index_docs(clause_type);
CREATE INDEX IF NOT EXISTS idx_idocs_status       ON index_docs(status);
CREATE INDEX IF NOT EXISTS idx_idocs_kind         ON index_docs(kind);

-- ---------------------------------------------------------------------------
-- document_acl: per-document group permissions.
-- Default group 'all' means publicly readable within the tenant.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_acl (
    doc_id      TEXT    NOT NULL,
    group_name  TEXT    NOT NULL,
    PRIMARY KEY (doc_id, group_name)
);

-- ---------------------------------------------------------------------------
-- communities: contract-family and clause-pattern community summaries.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS communities (
    community_id      TEXT        PRIMARY KEY,
    kind              TEXT        NOT NULL,
    permission_group  TEXT        NOT NULL DEFAULT 'all',
    summary           JSONB       NOT NULL DEFAULT '{}'::JSONB,
    source_hash       TEXT,
    stale             BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- review_queue: flags from extraction/resolution/graph for human review.
-- Unique constraint prevents duplicate open items for the same
-- (source_layer, doc_id, clause_id, reason) combination.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_queue (
    id              BIGSERIAL   PRIMARY KEY,
    source_layer    TEXT        NOT NULL,
    doc_id          TEXT,
    clause_id       TEXT,
    severity        TEXT        NOT NULL DEFAULT 'warning'
                                CHECK (severity IN ('info','warning','error')),
    reason          TEXT        NOT NULL,
    details         JSONB       NOT NULL DEFAULT '{}'::JSONB,
    status          TEXT        NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open','resolved','dismissed')),
    resolved_by     TEXT,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prevent duplicate open items for the same (source, doc, clause, reason).
-- COALESCE turns NULLs into sentinel strings so they participate in equality.
CREATE UNIQUE INDEX IF NOT EXISTS uq_review_open
    ON review_queue (source_layer, COALESCE(doc_id,''), COALESCE(clause_id,''), reason)
    WHERE status = 'open';

-- ---------------------------------------------------------------------------
-- review_overrides: versioned human corrections applied by later layers.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_overrides (
    id          BIGSERIAL   PRIMARY KEY,
    target_type TEXT        NOT NULL,   -- 'clause', 'document', 'party', etc.
    target_id   TEXT        NOT NULL,
    field       TEXT        NOT NULL,
    old_value   JSONB,
    new_value   JSONB       NOT NULL,
    reviewer    TEXT        NOT NULL,
    reason      TEXT,
    applied     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- audit_log: append-only record of every query/action.
-- A trigger rejects UPDATE and DELETE to enforce immutability.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id                  BIGSERIAL   PRIMARY KEY,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id             TEXT,
    action              TEXT        NOT NULL,
    query               TEXT,
    plan                JSONB,
    retrieved_span_ids  TEXT[]      NOT NULL DEFAULT '{}',
    output_hash         TEXT,
    model_versions      JSONB
);

-- Trigger function: reject any attempt to UPDATE or DELETE audit rows.
CREATE OR REPLACE FUNCTION audit_log_immutable()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_log rows are immutable: UPDATE and DELETE are not permitted.';
END;
$$;

-- DROP + CREATE triggers because CREATE TRIGGER IF NOT EXISTS requires PG 15+
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_update'
    ) THEN
        CREATE TRIGGER trg_audit_no_update
            BEFORE UPDATE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_no_delete'
    ) THEN
        CREATE TRIGGER trg_audit_no_delete
            BEFORE DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
    END IF;
END;
$$;
