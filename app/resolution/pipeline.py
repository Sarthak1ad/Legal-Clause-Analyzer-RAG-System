"""Layer 3 pipeline: ParsedDocument + ExtractedDocument → ResolvedDocument.

Combines:
  A. Party extraction + entity resolution  (app/resolution/parties.py + registry.py)
  B. Role mapping from role labels
  C. Clause-to-template matching           (app/resolution/matcher.py)

Saves output to data/resolved/{doc_id}.v{version}.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..extraction.schemas import ExtractedDocument
from ..llm.client import LLMClient, NullLLM
from ..models import ParsedDocument
from ..playbook.loader import load_playbook
from ..playbook.schemas import Playbook
from .matcher import match_document
from .registry import resolve_parties
from .schemas import ResolvedDocument, RoleMapping


def _build_role_mapping(parties) -> RoleMapping:
    """Build the customer/supplier role mapping from party role labels.

    We look for a party whose role_label (case-insensitive) is 'customer'
    or 'supplier'.  If no role labels were captured (e.g. an NDA that just
    says 'between X and Y'), both fields remain None.
    """
    customer_id: Optional[str] = None
    supplier_id: Optional[str] = None
    for p in parties:
        if p.role_label and p.party_id:
            label = p.role_label.lower()
            if "customer" in label or "buyer" in label or "client" in label:
                customer_id = p.party_id
            elif "supplier" in label or "vendor" in label or "provider" in label:
                supplier_id = p.party_id
    return RoleMapping(customer_party_id=customer_id, supplier_party_id=supplier_id)


def resolve(
    parsed: ParsedDocument,
    extracted: ExtractedDocument,
    data_dir: Path,
    llm: Optional[LLMClient] = None,
    playbook: Optional[Playbook] = None,
    save: bool = True,
) -> ResolvedDocument:
    """Run the full Layer 3 pipeline for one document.

    Args:
        parsed:     Layer 1 output (contains full_text and preamble)
        extracted:  Layer 2 output (contains clause types and attributes)
        data_dir:   root data directory (data/parties.json lives here)
        llm:        LLM client for borderline name merges (NullLLM if None)
        playbook:   loaded Playbook; loaded from default path if None
        save:       write ResolvedDocument to data/resolved/

    Returns:
        ResolvedDocument
    """
    if llm is None:
        llm = NullLLM()
    if playbook is None:
        playbook = load_playbook()

    # A. Party extraction + registry resolution
    parties = resolve_parties(parsed, data_dir, llm)

    # B. Role mapping
    role_mapping = _build_role_mapping(parties)

    # C. Template matching
    matches = match_document(extracted.clauses, playbook)

    doc = ResolvedDocument(
        doc_id=parsed.doc_id,
        version=parsed.version,
        parties=parties,
        role_mapping=role_mapping,
        template_matches=matches,
    )

    if save:
        out = data_dir / "resolved"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{parsed.doc_id}.v{parsed.version}.json").write_text(
            doc.model_dump_json(indent=2)
        )

    return doc
