-- 002_add_embedding_column.sql
-- Adds the embedding vector column to index_docs with the dimension set by
-- the configured embedder.  This is separated from 001 so we can set the
-- dimension dynamically (384 for bge-small, 512 for hashing embedder).
--
-- The migration runner replaces %%DIM%% with the actual integer before
-- executing this file.  If the column already exists (idempotent re-run),
-- the DO block silently skips.

DO $$
BEGIN
    -- Enable pgvector if not already present.
    -- Supabase exposes it in the 'extensions' schema; qualify the type below.
    CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

    -- Add the column only if it doesn't exist yet.
    IF NOT EXISTS (
        SELECT 1
        FROM   information_schema.columns
        WHERE  table_name = 'index_docs'
          AND  column_name = 'embedding'
    ) THEN
        EXECUTE format(
            'ALTER TABLE index_docs ADD COLUMN embedding extensions.vector(%s)',
            %%DIM%%
        );
    END IF;
END;
$$;

-- HNSW index for fast approximate nearest-neighbour search (cosine distance).
-- IF NOT EXISTS requires PG15+; use DO block for compatibility with PG14.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE  indexname = 'idx_idocs_embedding_hnsw'
    ) THEN
        EXECUTE 'CREATE INDEX idx_idocs_embedding_hnsw
                 ON index_docs USING hnsw (embedding extensions.vector_cosine_ops)
                 WITH (m = 16, ef_construction = 64)';
    END IF;
END;
$$;
