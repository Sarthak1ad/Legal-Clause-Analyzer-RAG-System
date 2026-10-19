// 3. SOWs and their parent MSA (PARENT_OF)
MATCH (parent:Contract)-[:PARENT_OF]->(child:Contract)
WHERE child.doc_type = 'SOW'
RETURN child.doc_id AS sow_id, parent.doc_id AS parent_msa_id, parent.effective_date AS parent_date
ORDER BY child.doc_id;
