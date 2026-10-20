// 2. Every ACTIVE clause with its standard template (INSTANCE_OF)
MATCH (c:Contract)-[:CONTAINS]->(cl:Clause)-[:INSTANCE_OF]->(t:StandardTemplate)
WHERE cl.status IS NULL OR cl.status <> 'superseded'
RETURN c.doc_id AS contract_id, cl.section_path AS section, cl.clause_type AS clause_type, t.template_id AS template_id, coalesce(cl.status, 'active') AS status
ORDER BY c.doc_id, cl.section_path;

