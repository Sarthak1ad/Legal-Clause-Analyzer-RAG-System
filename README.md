# Legal Clause Analyzer: Graph-Based Contract Reasoning

Backend-only project, built layer by layer.

## Status
| Layer | What it does | Status |
|---|---|---|
| 1. Ingestion | Parse DOCX/PDF/TXT, structure tree, span IDs, clause segmentation, metadata, versioning | done |
| 2. Extraction | Clause type + attributes + references + defined terms, with evidence spans (rules + local LLM) | done |
| 3. Entity resolution + template matching | Party extraction, canonical registry, playbook library, clause-to-template matching | done |
| 4+ | Graph, retrieval, planner, deviation engine, consistency checker, verifier | planned |

## Setup
```
pip install -r requirements.txt
ollama pull qwen2.5:3b        # only needed for LLM mode
```

## Run
```
python scripts/make_samples.py                                   # creates samples/
python -m app.ingestion.pipeline samples/msa_acme.docx --tree    # Layer 1
python -m app.extraction.run data/parsed/msa-acme.v1.json --provider ollama   # Layer 2
python -m app.extraction.run data/parsed/msa-acme.v1.json --provider rules    # no LLM
python -m app.resolution.run data/parsed/msa-acme.v1.json data/extracted/msa-acme.v1.json  # Layer 3
pytest tests
```

## Layout
```
app/models.py              shared models (Span, Section, ClauseCandidate, ParsedDocument)
app/ingestion/             Layer 1: parsers, numbering, structure, segmenter, metadata, registry, pipeline
app/llm/client.py          LLM interface: Ollama (qwen2.5:3b), rules-only, fake (tests)
app/extraction/            Layer 2: schemas, rules, extractor, CLI
app/playbook/              Layer 3: playbook schema, loader, templates validator
app/resolution/            Layer 3: party extraction, cross-doc registry, template matcher, pipeline, CLI
playbook/                  Playbook configuration, templates, and historical deal records
scripts/make_samples.py    sample contracts (DOCX + PDF)
tests/                     pytest suites for layers 1, 2, and 3
```
