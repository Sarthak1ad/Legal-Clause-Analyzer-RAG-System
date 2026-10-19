// 2. Every clause with its template (INSTANCE_OF)
MATCH (c:Contract)-[:CONTAINS]->(cl:Clause)-[:INSTANCE_OF]->(t:StandardTemplate)
RETURN c.doc_id AS contract_id, cl.section_path AS section, cl.clause_type AS clause_type, t.template_id AS template_id
ORDER BY c.doc_id, cl.section_path;
