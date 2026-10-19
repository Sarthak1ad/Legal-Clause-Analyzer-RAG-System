// 5. Clauses superseded by an amendment
MATCH (amend_cl:Clause)-[:SUPERSEDES]->(orig_cl:Clause)
MATCH (amend:Contract)-[:CONTAINS]->(amend_cl)
MATCH (orig:Contract)-[:CONTAINS]->(orig_cl)
RETURN orig.doc_id AS original_contract, orig_cl.section_path AS section, orig_cl.status AS original_status, amend.doc_id AS amending_doc, amend_cl.clause_id AS replacement_clause
ORDER BY orig.doc_id, orig_cl.section_path;
