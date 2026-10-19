"""Clause-to-template matcher (Part C).

Primary rule (deterministic, score=1.0):
    clause_type  → the single approved template for that type.

This 1-to-1 mapping is intentional.  Note that indemnification has TWO clauses
(9.1 scope, 9.2 cap) of the same type in the sample contracts; both link to the
same TPL-INDEM-v1 template.  That is correct behaviour.
TODO (Layer 4): the deviation engine will combine attributes across clauses of
the same type per contract before comparing against the template standards.

Feature flag USE_EMBEDDINGS (off by default):
    If set, we also compute sentence-transformers cosine similarity between
    the clause text and the template's standard_text.  This score is stored
    for information only and does NOT change the primary type_match result.
    This is off by default so the pipeline remains fully offline/deterministic.
"""
from __future__ import annotations

import os
from typing import Optional

from ..extraction.schemas import ExtractedClause
from ..playbook.schemas import Playbook, StandardTemplate
from .schemas import TemplateMatch

# Feature flag: set USE_EMBEDDINGS=1 in environment to enable sentence-transformers.
_USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "0") == "1"


def _embedding_sim(clause_text: str, standard_text: str) -> Optional[float]:
    """Compute cosine similarity via sentence-transformers (feature-flagged).

    Returns None if the flag is off, or if sentence-transformers is not installed.
    We import lazily so the module can always be imported even without the library.
    """
    if not _USE_EMBEDDINGS:
        return None
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = _model.encode([clause_text, standard_text], convert_to_tensor=True)
        sim = float(util.cos_sim(embs[0], embs[1]))
        return round(sim, 4)
    except Exception:
        return None


def match_clause(
    clause: ExtractedClause,
    playbook: Playbook,
) -> TemplateMatch:
    """Match one extracted clause to a standard template.

    Args:
        clause:   extracted clause with type, confidence, and review flags
        playbook: the loaded playbook

    Returns:
        TemplateMatch – always one per clause; template_id is None when there
        is no standard template for this clause type.
    """
    template: Optional[StandardTemplate] = playbook.approved_template(
        clause.clause_type.value
    )

    # Carry forward Layer 2 review flags regardless of whether we find a template
    needs_review = clause.needs_review
    review_reasons = list(clause.review_reasons)

    # Also flag low-confidence clauses from Layer 2
    if clause.confidence < 0.8 and not clause.needs_review:
        needs_review = True
        review_reasons.append(
            f"Layer 2 confidence {clause.confidence:.2f} < 0.8; type uncertain"
        )

    if template is None:
        # No standard template for this clause type (e.g. notices, definitions).
        # This is NOT an error—just record it.
        return TemplateMatch(
            clause_id=clause.clause_id,
            section_path=clause.section_path,
            clause_type=clause.clause_type.value,
            template_id=None,
            template_version=None,
            playbook_version=None,
            method="none",
            score=0.0,
            needs_review=needs_review,
            review_reasons=review_reasons + ["no standard template for this clause type"],
        )

    # Primary: type_match is deterministic and always scores 1.0
    emb_sim = _embedding_sim(
        clause.span.section_path,  # use section_path as proxy; full text not stored here
        template.standard_text,
    )

    return TemplateMatch(
        clause_id=clause.clause_id,
        section_path=clause.section_path,
        clause_type=clause.clause_type.value,
        template_id=template.template_id,
        template_version=template.version,
        playbook_version=playbook.config.playbook_version,
        method="type_match",
        score=1.0,
        embedding_sim=emb_sim,
        needs_review=needs_review,
        review_reasons=review_reasons,
    )


def match_document(
    clauses: list[ExtractedClause],
    playbook: Playbook,
) -> list[TemplateMatch]:
    """Match all clauses in a document to templates."""
    return [match_clause(c, playbook) for c in clauses]
