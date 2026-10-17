"""Step 1: turn a file into a list of text Blocks (paragraphs) with page numbers.

Why blocks? Every later step (structure, clauses, spans) works on paragraphs,
not on raw bytes. Parsers differ per format; everything after this is shared.

Security note: contract text is UNTRUSTED. We normalise it here (strip hidden
zero-width characters, odd whitespace) so that hidden text tricks cannot smuggle
instructions past us, and so that offsets stay stable.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .numbering import is_numbered_line
from ..models import Block

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿⁠­"), None)


def normalize(text: str) -> str:
    """Normalise one paragraph. MUST run before offsets are computed."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace(" ", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_txt(path: Path) -> tuple[list[Block], int, list[str]]:
    blocks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = normalize(line)
        if t:
            blocks.append(Block(text=t, page=1))
    return blocks, 1, []


def parse_docx(path: Path) -> tuple[list[Block], int, list[str]]:
    """Read paragraphs and tables from a Word file.

    Limitation (documented, fine for the demo): Word *auto-numbering* is not
    part of paragraph.text, so we assume section numbers are typed in the text.
    DOCX also has no fixed pagination, so every block is page 1.
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks: list[Block] = []
    # iterate body in document order so tables stay where they appear
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = normalize(Paragraph(child, doc).text)
            if t:
                blocks.append(Block(text=t, page=1))
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [normalize(c.text) for c in row.cells]
                t = " | ".join(c for c in cells if c)
                if t:
                    blocks.append(Block(text=t, page=1, kind="table"))
    return blocks, 1, ["DOCX has no fixed pages; all blocks assigned to page 1"]


def parse_pdf(path: Path) -> tuple[list[Block], int, list[str]]:
    """Read a digital PDF with PyMuPDF, page by page.

    PDF text comes as visual lines. We rebuild paragraphs: a new paragraph
    starts at a PyMuPDF block boundary or at a line that begins with a clause
    number (so '9.2 Indemnity...' never merges into the line above it).
    Pages with almost no text are flagged as needing OCR (not run in the demo).
    """
    import pymupdf

    blocks: list[Block] = []
    warnings: list[str] = []
    pdf = pymupdf.open(str(path))
    for pno, page in enumerate(pdf, start=1):
        raw = page.get_text("blocks", sort=True)
        text_blocks = [b for b in raw if b[6] == 0]
        total_chars = sum(len(b[4].strip()) for b in text_blocks)
        if total_chars < 20:
            warnings.append(f"page {pno}: no extractable text, needs OCR (low confidence)")
            continue
        for b in text_blocks:
            acc: list[str] = []
            for line in b[4].splitlines():
                line = normalize(line)
                if not line:
                    continue
                if acc and is_numbered_line(line):
                    blocks.append(Block(text=normalize(" ".join(acc)), page=pno))
                    acc = []
                acc.append(line)
            if acc:
                blocks.append(Block(text=normalize(" ".join(acc)), page=pno))
    return blocks, len(pdf), warnings


def parse_file(path: Path) -> tuple[list[Block], int, list[str]]:
    ext = path.suffix.lower()
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in (".txt", ".md"):
        return parse_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def assemble_full_text(blocks: list[Block]) -> str:
    """Join blocks with '\\n' and record each block's [start, end) offsets.

    `full_text` is the single source of truth. Spans are offsets into it, so
    `full_text[span.start:span.end]` always returns the cited text exactly.
    """
    parts, pos = [], 0
    for b in blocks:
        b.start = pos
        b.end = pos + len(b.text)
        parts.append(b.text)
        pos = b.end + 1  # +1 for the '\n' separator
    return "\n".join(parts)
