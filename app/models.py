"""Data models shared by every layer.

The most important idea in this file is the Span: a pointer into the
document's immutable `full_text`. Every fact the system extracts later
(a cap amount, a date, a party) must carry a Span, so that the final
citation verifier can check `full_text[start:end]` against the quoted text.
"""
from __future__ import annotations

from datetime import date as Date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Block(BaseModel):
    """One paragraph (or table row) as read from the source file."""
    text: str
    page: int = 1
    kind: Literal["paragraph", "table"] = "paragraph"
    start: int = 0  # offsets into ParsedDocument.full_text (filled in later)
    end: int = 0


class Span(BaseModel):
    """Immutable pointer to a character range of one document version."""
    span_id: str  # e.g. msa-acme:v1:p1:9.2:4210-4688
    doc_id: str
    version: int
    page: int
    section_path: str
    start: int
    end: int


def make_span(doc_id: str, version: int, page: int, section_path: str,
              start: int, end: int) -> Span:
    sid = f"{doc_id}:v{version}:p{page}:{section_path}:{start}-{end}"
    return Span(span_id=sid, doc_id=doc_id, version=version, page=page,
                section_path=section_path, start=start, end=end)


class Section(BaseModel):
    """A node of the structure tree: Article > Section > Subsection > ..."""
    label: str  # "9.2", "(a)", "Article IV"
    path: str   # "9.2(a)"
    kind: Literal["article", "decimal", "letter", "roman"]
    level: int
    title: Optional[str] = None
    span: Span
    children: list["Section"] = Field(default_factory=list)


class ClauseCandidate(BaseModel):
    """A unit of text the extraction layer will classify (Layer 2)."""
    clause_id: str
    section_path: str
    heading: Optional[str] = None
    parent_heading: Optional[str] = None
    text: str
    span: Span
    sub_spans: list[Span] = Field(default_factory=list)  # (a), (b)... for precise citations


class ParentRef(BaseModel):
    """'This SOW is issued under the Master Services Agreement dated ...'"""
    doc_type: str
    date: Optional[Date] = None
    raw: str


class ParsedDocument(BaseModel):
    doc_id: str
    version: int
    sha256: str
    source_file: str
    doc_type: str = "OTHER"
    title: Optional[str] = None
    effective_date: Optional[Date] = None
    parent_ref: Optional[ParentRef] = None
    num_pages: int = 1
    full_text: str
    sections: list[Section] = Field(default_factory=list)
    clauses: list[ClauseCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def text_of(self, span: Span) -> str:
        return self.full_text[span.start:span.end]


Section.model_rebuild()
