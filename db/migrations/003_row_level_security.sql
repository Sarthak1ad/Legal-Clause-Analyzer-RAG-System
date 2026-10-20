-- 003_row_level_security.sql
-- Row-level security (RLS) for defence-in-depth on confidential contract data.
--
-- Design:
-- - A NOLOGIN role 'app_reader' is created to represent application queries.
-- - RLS is enabled on index_docs, clauses, documents, and communities.
-- - Policy: a row passes if its acl_groups (text[]) overlaps the set of groups
--   provided by the application in the local setting app.user_groups.
-- - 'admin' group bypasses RLS entirely.
-- - The primary protection is the explicit WHERE acl_groups && $groups filter
--   in every query; RLS is the second wall.

-- Create the role if absent (idempotent).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_reader') THEN
        CREATE ROLE app_reader NOLOGIN;
    END IF;
END;
$$;

-- Grant SELECT on the four protected tables to app_reader.
GRANT SELECT ON index_docs   TO app_reader;
GRANT SELECT ON clauses      TO app_reader;
GRANT SELECT ON documents    TO app_reader;
GRANT SELECT ON communities  TO app_reader;

-- Enable RLS (idempotent).
ALTER TABLE index_docs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE clauses      ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents    ENABLE ROW LEVEL SECURITY;
ALTER TABLE communities  ENABLE ROW LEVEL SECURITY;

-- Force RLS even for the table owner (ensures tests work correctly).
ALTER TABLE index_docs   FORCE ROW LEVEL SECURITY;
ALTER TABLE clauses      FORCE ROW LEVEL SECURITY;
ALTER TABLE documents    FORCE ROW LEVEL SECURITY;
ALTER TABLE communities  FORCE ROW LEVEL SECURITY;

-- ---- index_docs policy ----
DROP POLICY IF EXISTS rls_index_docs ON index_docs;
CREATE POLICY rls_index_docs ON index_docs
    FOR SELECT
    USING (
        -- 'admin' group bypasses all restrictions
        'admin' = ANY(string_to_array(current_setting('app.user_groups', TRUE), ','))
        OR
        -- acl_groups overlaps with the session's user groups
        acl_groups && string_to_array(current_setting('app.user_groups', TRUE), ',')
    );

-- ---- clauses policy ----
-- clauses table uses doc_id for ACL; join to document_acl for the group list.
-- For simplicity, clauses inherit document-level ACL: a clause is visible if
-- its doc_id has at least one group that overlaps the user's groups.
DROP POLICY IF EXISTS rls_clauses ON clauses;
CREATE POLICY rls_clauses ON clauses
    FOR SELECT
    USING (
        'admin' = ANY(string_to_array(current_setting('app.user_groups', TRUE), ','))
        OR
        EXISTS (
            SELECT 1 FROM document_acl da
            WHERE da.doc_id = clauses.doc_id
              AND da.group_name = ANY(
                  string_to_array(current_setting('app.user_groups', TRUE), ',')
              )
        )
        OR
        NOT EXISTS (
            SELECT 1 FROM document_acl da WHERE da.doc_id = clauses.doc_id
        )  -- no ACL row => public
    );

-- ---- documents policy ----
DROP POLICY IF EXISTS rls_documents ON documents;
CREATE POLICY rls_documents ON documents
    FOR SELECT
    USING (
        'admin' = ANY(string_to_array(current_setting('app.user_groups', TRUE), ','))
        OR
        EXISTS (
            SELECT 1 FROM document_acl da
            WHERE da.doc_id = documents.doc_id
              AND da.group_name = ANY(
                  string_to_array(current_setting('app.user_groups', TRUE), ',')
              )
        )
        OR
        NOT EXISTS (
            SELECT 1 FROM document_acl da WHERE da.doc_id = documents.doc_id
        )
    );

-- ---- communities policy ----
DROP POLICY IF EXISTS rls_communities ON communities;
CREATE POLICY rls_communities ON communities
    FOR SELECT
    USING (
        'admin' = ANY(string_to_array(current_setting('app.user_groups', TRUE), ','))
        OR
        permission_group = 'all'
        OR
        permission_group = ANY(
            string_to_array(current_setting('app.user_groups', TRUE), ',')
        )
    );
