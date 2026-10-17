"""Step 3: decide which pieces of the tree are 'clauses'.

A clause is the unit the extraction layer (Layer 2) will classify, e.g.
"9.2 Indemnification". Rules:
  * Article: a container, its numbered children are the clauses.
  * Level-1 section WITH numbered sub-sections (9. > 9.1, 9.2): each sub-section
    is a clause; the parent's own intro text becomes a clause only if substantial.
  * Everything else is one clause, INCLUDING its (a), (b), (i) children, so a
    clause is never cut in the middle of its own list.
Each clause keeps spans of its sub-paragraphs for precise citations later.
"""
from __future__ import annotations

from typing import Iterator, Optional

from ..models import ClauseCandidate, Section, Span, make_span

_MIN_INTRO_CHARS = 80


def _descendants(sec: Section) -> Iterator[Section]:
    for c in sec.children:
        yield c
        yield from _descendants(c)


def segment(doc_id: str, version: int, full_text: str,
            sections: list[Section]) -> list[ClauseCandidate]:
    out: list[ClauseCandidate] = []
    seen: dict[str, int] = {}

    def emit(sec: Section, parent_heading: Optional[str], end: Optional[int] = None):
        start = sec.span.start
        end = sec.span.end if end is None else end
        subs: list[Span] = [d.span for d in _descendants(sec) if d.span.start < end]
        cid = f"{doc_id}:v{version}:{sec.path}"
        seen[cid] = seen.get(cid, 0) + 1
        if seen[cid] > 1:  # repeated numbering in a messy document
            cid += f"#{seen[cid]}"
        out.append(ClauseCandidate(
            clause_id=cid, section_path=sec.path, heading=sec.title,
            parent_heading=parent_heading, text=full_text[start:end],
            span=make_span(doc_id, version, sec.span.page, sec.path, start, end),
            sub_spans=subs,
        ))

    def walk(sec: Section, parent: Optional[Section]):
        if sec.kind == "article":
            if sec.children:
                for c in sec.children:
                    walk(c, sec)
            else:
                emit(sec, None)
            return
        dec_children = [c for c in sec.children if c.kind == "decimal"]
        if sec.level == 1 and dec_children:
            intro_end = dec_children[0].span.start - 1
            intro = full_text[sec.span.start:intro_end]
            if len(intro) - len(sec.title or "") > _MIN_INTRO_CHARS:
                emit(sec, parent.title if parent else None, end=intro_end)
            for c in dec_children:
                walk(c, sec)
        else:
            emit(sec, parent.title if parent else None)

    for s in sections:
        walk(s, None)
    return out
