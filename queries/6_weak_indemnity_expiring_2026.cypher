// 6. Contracts whose ACTIVE indemnification clause has a cap_multiple below
//    the playbook standard (2.0x) AND whose current term ends in 2026.
//    Status filter ensures superseded clauses are excluded.
MATCH (c:Contract)-[:HAS_DATE]->(d:DateEvent {kind: 'current_term_end'})
WHERE d.date STARTS WITH '2026'
MATCH (c)-[:CONTAINS]->(cl:Clause {clause_type: 'indemnification', status: 'active'})
WHERE cl.cap_multiple IS NOT NULL
  AND toFloat(cl.cap_multiple) < 2.0
RETURN
  c.doc_id            AS contract_id,
  c.doc_type          AS doc_type,
  d.date              AS current_term_end,
  toFloat(cl.cap_multiple) AS cap_multiple,
  cl.clause_id        AS clause_id
ORDER BY c.doc_id;
