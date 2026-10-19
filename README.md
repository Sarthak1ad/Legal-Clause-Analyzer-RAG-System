# Legal Clause Analyzer: Graph-Based Contract Reasoning

Backend-only project, built layer by layer.

## Status
| Layer | What it does | Status |
|---|---|---|
| 1. Ingestion | Parse DOCX/PDF/TXT, structure tree, span IDs, clause segmentation, metadata, versioning | done |
| 2. Extraction | Clause type + attributes + references + defined terms, with evidence spans (rules + local LLM) | done |
| 3. Entity resolution + template matching | Party extraction, canonical registry, playbook library, clause-to-template matching | done |
| 4. Knowledge graph + effective state | Graph model, derived dates, amendment effective resolution, Neo4j storage | done |
| 5+ | Retrieval, planner, deviation engine, consistency checker, verifier | planned |

## Setup
```
pip install -r requirements.txt
ollama pull qwen2.5:3b        # only needed for LLM mode

# Optional Neo4j Database:
docker-compose up -d          # starts Neo4j on ports 7474 (browser) and 7687 (bolt)
```

## Run
```
python scripts/make_samples.py                                   # creates samples/
python -m app.ingestion.pipeline samples/msa_acme.docx --tree    # Layer 1
python -m app.extraction.run data/parsed/msa-acme.v1.json --provider ollama   # Layer 2
python -m app.extraction.run data/parsed/msa-acme.v1.json --provider rules    # no LLM
python -m app.resolution.run data/parsed/msa-acme.v1.json data/extracted/msa-acme.v1.json  # Layer 3
python -m app.graph.run --data-dir data --as-of 2026-10-19       # Layer 4 (build + export JSON)
python -m app.graph.run --data-dir data --as-of 2026-10-19 --neo4j --rebuild  # Layer 4 (load Neo4j)
python -m app.graph.query all                                    # Layer 4 (sanity queries)
pytest
```

## Layout
```
app/models.py              shared models (Span, Section, ClauseCandidate, ParsedDocument)
app/ingestion/             Layer 1: parsers, numbering, structure, segmenter, metadata, registry, pipeline
app/llm/client.py          LLM interface: Ollama (qwen2.5:3b), rules-only, fake (tests)
app/extraction/            Layer 2: schemas, rules, extractor, CLI
app/playbook/              Layer 3: playbook schema, loader, templates validator
app/resolution/            Layer 3: party extraction, cross-doc registry, template matcher, pipeline, CLI
app/graph/                 Layer 4: knowledge graph model, derived dates, effective state, Neo4j store, CLI
docker-compose.yml         Layer 4: local Neo4j community container definition
playbook/                  Playbook configuration, templates, and historical deal records
queries/                   Layer 4: 5 ready-made Cypher sanity queries
scripts/make_samples.py    sample contracts (DOCX + PDF)
tests/                     pytest suites for layers 1, 2, 3, and 4
```
