"""Layer 1 entry point: file in -> ParsedDocument (JSON) out.

    python -m app.ingestion.pipeline samples/msa_acme.docx
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from ..models import ClauseCandidate, ParsedDocument, Section, make_span
from .metadata import extract_metadata
from .parsers import assemble_full_text, parse_file
from .registry import register
from .segmenter import segment
from .structure import build_sections


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def validate(doc: ParsedDocument) -> list[str]:
    """Self-check: every span must point inside full_text and match its clause."""
    problems = []
    n = len(doc.full_text)
    for c in doc.clauses:
        s = c.span
        if not (0 <= s.start < s.end <= n):
            problems.append(f"{c.clause_id}: span out of range")
        elif doc.text_of(s) != c.text:
            problems.append(f"{c.clause_id}: span text mismatch")
    return problems


def ingest(path: str | Path, data_dir: str | Path = "data",
           doc_id: str | None = None, save: bool = True) -> ParsedDocument:
    path, data_dir = Path(path), Path(data_dir)
    doc_id = doc_id or _slug(path.stem)

    blocks, num_pages, warnings = parse_file(path)
    full_text = assemble_full_text(blocks)
    sha = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    version = register(data_dir, doc_id, sha) if save else 1

    sections: list[Section] = build_sections(doc_id, version, blocks, len(full_text))
    clauses: list[ClauseCandidate] = segment(doc_id, version, full_text, sections)
    if not clauses and full_text:
        warnings.append("no numbered structure found; whole document is one clause")
        clauses = [ClauseCandidate(
            clause_id=f"{doc_id}:v{version}:doc", section_path="doc", text=full_text,
            span=make_span(doc_id, version, 1, "doc", 0, len(full_text)))]

    title, doc_type, eff, parent = extract_metadata([b.text for b in blocks], full_text)
    doc = ParsedDocument(
        doc_id=doc_id, version=version, sha256=sha, source_file=path.name,
        doc_type=doc_type, title=title, effective_date=eff, parent_ref=parent,
        num_pages=num_pages, full_text=full_text, sections=sections,
        clauses=clauses, warnings=warnings,
    )
    doc.warnings += validate(doc)
    if save:
        out = data_dir / "parsed"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{doc_id}.v{version}.json").write_text(doc.model_dump_json(indent=2))
    return doc


def _print_tree(secs: list[Section], indent: int = 0):
    for s in secs:
        print("  " * indent + f"{s.path:<12} {s.title or ''}   [{s.span.start}-{s.span.end}]")
        _print_tree(s.children, indent + 1)


def main():
    ap = argparse.ArgumentParser(description="Layer 1: ingest a contract")
    ap.add_argument("file")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tree", action="store_true", help="print the section tree")
    a = ap.parse_args()
    d = ingest(a.file, a.data_dir)
    print(f"doc_id={d.doc_id}  version={d.version}  type={d.doc_type}  pages={d.num_pages}")
    print(f"title={d.title!r}  effective={d.effective_date}  parent={d.parent_ref.model_dump() if d.parent_ref else None}")
    print(f"sections={sum(1 for _ in _walk(d.sections))}  clauses={len(d.clauses)}")
    for c in d.clauses:
        print(f"  {c.section_path:<8} {(c.heading or '-'):<28} {c.span.span_id}")
    if a.tree:
        _print_tree(d.sections)
    for w in d.warnings:
        print("WARNING:", w)


def _walk(secs):
    for s in secs:
        yield s
        yield from _walk(s.children)


if __name__ == "__main__":
    main()
