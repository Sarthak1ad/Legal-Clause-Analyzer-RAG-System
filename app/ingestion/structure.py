"""Step 2: rebuild the document hierarchy (Article > Section > Subsection > Paragraph).

Idea: walk the paragraphs in order. Whenever a paragraph starts with a legal
number, it opens a new node. A stack of "currently open" nodes tells us the
parent. When a node closes (the next sibling or a shallower node begins), its
end offset is the start of that next node, so a parent's range always covers
all of its children.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .numbering import Marker, classify
from ..models import Block, Section, make_span


@dataclass
class _Node:
    marker: Marker
    path: str
    level: int
    start: int
    page: int
    end: int = -1
    children: list["_Node"] = field(default_factory=list)


def _article_path(label: str) -> str:
    return "Art." + label.split()[-1]


def build_sections(doc_id: str, version: int, blocks: list[Block],
                   full_len: int) -> list[Section]:
    roots: list[_Node] = []
    stack: list[_Node] = []
    last_letter: Optional[str] = None
    i = 0
    while i < len(blocks):
        b = blocks[i]
        mk = classify(b.text, last_letter)
        i += 1
        if mk is None:
            continue

        # "ARTICLE IV" on one line, "INDEMNIFICATION" (all caps) on the next
        if mk.kind == "article" and mk.title is None and i < len(blocks):
            nxt = blocks[i].text
            if nxt.isupper() and len(nxt) < 80 and classify(nxt, None) is None:
                mk.title = nxt.title()
                i += 1

        def close(node: _Node):
            node.end = b.start - 1  # exclude the '\n' separator

        if mk.kind == "article":
            while stack:
                close(stack.pop())
            level, path = 0, _article_path(mk.label)
            last_letter = None
        elif mk.kind == "decimal":
            level = mk.level
            while stack and stack[-1].level >= level:
                close(stack.pop())
            path = mk.label
            last_letter = None
        elif mk.kind == "letter":
            while stack and stack[-1].marker.kind in ("letter", "roman"):
                close(stack.pop())
            if not stack:  # a lettered list with no numbered parent: body text
                continue
            level = stack[-1].level + 1
            path = stack[-1].path + mk.label
            last_letter = mk.label.strip("()").lower()
        else:  # roman
            while stack and stack[-1].marker.kind == "roman":
                close(stack.pop())
            if not stack:
                continue
            level = stack[-1].level + 1
            path = stack[-1].path + mk.label

        node = _Node(marker=mk, path=path, level=level, start=b.start, page=b.page)
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    while stack:
        stack.pop().end = full_len

    def convert(n: _Node) -> Section:
        return Section(
            label=n.marker.label, path=n.path, kind=n.marker.kind, level=n.level,
            title=n.marker.title,
            span=make_span(doc_id, version, n.page, n.path, n.start, n.end),
            children=[convert(c) for c in n.children],
        )

    return [convert(r) for r in roots]
