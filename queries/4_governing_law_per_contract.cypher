// 4. Governing law per contract (to eyeball the NY vs CA mismatch)
MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
RETURN c.doc_id AS contract_id, c.doc_type AS doc_type, j.name AS governing_law
ORDER BY c.doc_id;
