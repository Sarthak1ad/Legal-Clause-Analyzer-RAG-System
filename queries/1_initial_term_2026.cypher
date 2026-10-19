// 1. Contracts whose initial_term_end falls in 2026
MATCH (c:Contract)-[:HAS_DATE]->(d:DateEvent {kind: 'initial_term_end'})
WHERE d.date STARTS WITH '2026'
RETURN c.doc_id AS contract_id, c.doc_type AS doc_type, d.date AS initial_term_end, d.derivation AS derivation
ORDER BY c.doc_id;
