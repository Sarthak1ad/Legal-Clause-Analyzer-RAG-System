"""Layer 2 CLI.

    python -m app.extraction.run data/parsed/msa-acme.v1.json --provider ollama
    python -m app.extraction.run data/parsed/msa-acme.v1.json --provider rules   # no LLM needed

Ollama setup (once):  ollama pull qwen2.5:3b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..llm.client import get_client
from ..models import ParsedDocument
from .extractor import extract_document


def main():
    ap = argparse.ArgumentParser(description="Layer 2: extract clause types and attributes")
    ap.add_argument("parsed_json")
    ap.add_argument("--provider", default=None, help="ollama | rules (default: $LLM_PROVIDER or ollama)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-llm-fill", action="store_true")
    a = ap.parse_args()

    doc = ParsedDocument.model_validate_json(Path(a.parsed_json).read_text())
    llm = get_client(a.provider)
    ex = extract_document(doc, llm, use_llm_fill=not a.no_llm_fill)

    out = Path(a.data_dir) / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ex.doc_id}.v{ex.version}.json").write_text(ex.model_dump_json(indent=2))

    print(f"{ex.doc_id} v{ex.version}  type={ex.doc_type}  model={ex.llm_model}")
    for c in ex.clauses:
        attrs = ", ".join(f"{x.name}={x.value}" for x in c.attributes if x.source != "absence")
        flag = "  [REVIEW]" if c.needs_review else ""
        print(f"  {c.section_path:<7} {c.clause_type.value:<24} {c.confidence:.2f}{flag}")
        if attrs:
            print(f"          {attrs}")
        for r in c.review_reasons:
            print(f"          ! {r}")
    n_rev = sum(c.needs_review for c in ex.clauses)
    print(f"{len(ex.clauses)} clauses, {n_rev} need review")


if __name__ == "__main__":
    main()
