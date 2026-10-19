"""Amendment parsing and contract effective-state resolution (Layer 4 Part C).

Parses amendment operations ("Section X is deleted and replaced with..."),
runs replacement text through Layer 2 rules to extract synthetic clauses,
and resolves the effective state of each contract as of a given date.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional, Union

from ..extraction.rules import (
    EXTRACTORS,
    classify_rules,
    find_defined_terms,
    find_references,
)
from ..extraction.schemas import ClauseType, ExtractedClause
from ..models import ClauseCandidate, ParsedDocument, make_span
from .dates import parse_iso_date
from .model import (
    ClauseNode,
    EffectiveClauseEntry,
    EffectiveView,
    ReviewItem,
    compute_text_hash,
)

# Regex matching amendment replacement operations (requires matching double/smart quotes so internal apostrophes are preserved)
_AMEND_OP_RE = re.compile(
    r'Section\s+([\d.]+)\s+is\s+(?:deleted\s+and\s+replaced\s+with|amended\s+to\s+read)[:\s]*'
    r'["\u201c](.*?)["\u201d]',
    re.IGNORECASE | re.DOTALL,
)


class AmendmentOperation:
    """A parsed modification operation from an amendment contract."""

    def __init__(
        self,
        target_path: str,
        replacement_text: str,
        heading: Optional[str],
        extracted_clause: ExtractedClause,
        span_id: str,
        source_doc_id: str,
        effective_date: Optional[date],
    ):
        self.target_path = target_path
        self.replacement_text = replacement_text
        self.heading = heading
        self.extracted_clause = extracted_clause
        self.span_id = span_id
        self.source_doc_id = source_doc_id
        self.effective_date = effective_date


def parse_amendment_operations(
    amend_parsed: ParsedDocument,
) -> tuple[list[AmendmentOperation], list[ReviewItem]]:
    """Parse replacement operations from an amendment ParsedDocument.

    Returns:
        tuple of (list of AmendmentOperation, list of ReviewItem)
    """
    operations: list[AmendmentOperation] = []
    reviews: list[ReviewItem] = []
    full_text = amend_parsed.full_text

    # Search each clause/paragraph for operations
    for clause in amend_parsed.clauses:
        text = clause.text
        matches = list(_AMEND_OP_RE.finditer(text))
        if not matches:
            # Check if this clause looks like an amendment operation that failed parsing
            if re.search(r"\b(?:deleted\s+and\s+replaced|amended\s+to\s+read)\b", text, re.IGNORECASE):
                reviews.append(
                    ReviewItem(
                        item_id=f"rev:amend_parse:{clause.clause_id}",
                        doc_id=amend_parsed.doc_id,
                        kind="unparseable_amendment_operation",
                        description=(
                            f"Clause {clause.section_path} in {amend_parsed.doc_id} contains "
                            f"amendment phrasing but could not be parsed into target and replacement text."
                        ),
                        span_id=clause.span.span_id,
                    )
                )
            continue

        for m in matches:
            target_path = m.group(1).strip()
            replacement_raw = m.group(2).strip()

            # Locate the replacement text in full_text to get exact char offsets
            idx = full_text.find(replacement_raw, clause.span.start)
            if idx < 0:
                idx = full_text.find(replacement_raw)
            if idx >= 0:
                abs_start = idx
                abs_end = idx + len(replacement_raw)
            else:
                abs_start = clause.span.start
                abs_end = clause.span.end

            span = make_span(
                amend_parsed.doc_id,
                amend_parsed.version,
                clause.span.page,
                target_path,
                abs_start,
                abs_end,
            )

            # Detect heading from replacement text: e.g. "9.2 Cap on Indemnity. Supplier's liability..."
            heading = None
            hm = re.match(r"^[\d.]+\s+([A-Za-z\s]+?)\.\s*(.*)$", replacement_raw, re.DOTALL)
            if hm:
                heading = hm.group(1).strip()

            candidate = ClauseCandidate(
                clause_id=f"{amend_parsed.doc_id}:{target_path}",
                section_path=target_path,
                heading=heading,
                span=span,
                text=replacement_raw,
                parent_heading=None,
            )

            # Run through Layer 2 deterministic rules
            rule_type, rule_conf = classify_rules(candidate)
            attrs = EXTRACTORS[rule_type](candidate) if rule_type in EXTRACTORS else []
            refs = find_references(candidate)
            terms = find_defined_terms(candidate)

            extracted = ExtractedClause(
                clause_id=candidate.clause_id,
                section_path=target_path,
                heading=heading,
                parent_heading=None,
                span=span,
                clause_type=rule_type,
                type_by_rule=rule_type,
                type_by_llm=None,
                confidence=rule_conf,
                needs_review=(rule_conf < 0.6),
                review_reasons=[] if rule_conf >= 0.6 else ["Low extraction confidence"],
                attributes=attrs,
                references=refs,
                defined_terms=terms,
            )

            operations.append(
                AmendmentOperation(
                    target_path=target_path,
                    replacement_text=replacement_raw,
                    heading=heading,
                    extracted_clause=extracted,
                    span_id=span.span_id,
                    source_doc_id=amend_parsed.doc_id,
                    effective_date=amend_parsed.effective_date,
                )
            )

    return operations, reviews


def resolve_effective_clauses(
    parent_doc_id: str,
    original_clauses: list[ClauseNode],
    amendment_operations: list[AmendmentOperation],
    as_of: Union[str, date],
) -> tuple[EffectiveView, dict[str, ClauseNode], list[ReviewItem]]:
    """Determine the effective clauses for a contract as of an as_of date.

    Args:
        parent_doc_id: target contract doc_id (e.g. 'msa-acme')
        original_clauses: list of ClauseNodes belonging to the original contract
        amendment_operations: list of all AmendmentOperations amending this contract
        as_of: target evaluation date

    Returns:
        tuple of (EffectiveView, dict[section_path -> effective_ClauseNode], list of ReviewItem)
    """
    as_of_date = parse_iso_date(as_of) or date.today()
    reviews: list[ReviewItem] = []

    # Map original clauses by section_path
    orig_by_path: dict[str, ClauseNode] = {c.section_path: c for c in original_clauses}

    # Group valid operations by target section path
    ops_by_path: dict[str, list[AmendmentOperation]] = {}
    for op in amendment_operations:
        # Check target existence in parent
        if op.target_path not in orig_by_path:
            reviews.append(
                ReviewItem(
                    item_id=f"rev:target_not_found:{op.source_doc_id}:{op.target_path}",
                    doc_id=op.source_doc_id,
                    kind="amendment_target_not_found",
                    description=(
                        f"Amendment {op.source_doc_id} targets section {op.target_path!r} "
                        f"which does not exist in parent contract {parent_doc_id}."
                    ),
                    span_id=op.span_id,
                )
            )
            continue

        # Check effective date presence
        if op.effective_date is None:
            reviews.append(
                ReviewItem(
                    item_id=f"rev:missing_eff_date:{op.source_doc_id}",
                    doc_id=op.source_doc_id,
                    kind="amendment_missing_effective_date",
                    description=f"Amendment {op.source_doc_id} has no effective date; cannot apply.",
                    span_id=op.span_id,
                )
            )
            continue

        ops_by_path.setdefault(op.target_path, []).append(op)

    effective_clauses: dict[str, ClauseNode] = {}
    effective_entries: list[EffectiveClauseEntry] = []

    for path, orig_clause in sorted(orig_by_path.items()):
        ops = ops_by_path.get(path, [])
        # Filter operations in effect as of as_of_date
        active_ops = [op for op in ops if op.effective_date and op.effective_date <= as_of_date]

        if not active_ops:
            # Original clause remains effective
            effective_clauses[path] = orig_clause
            effective_entries.append(
                EffectiveClauseEntry(
                    section_path=path,
                    effective_clause_id=orig_clause.clause_id,
                    source_doc=parent_doc_id,
                    effective_from=orig_clause.attributes.get("effective_date", "original"),
                    superseded_clause_ids=[],
                )
            )
        else:
            # Check for ambiguity: multiple operations with the exact same effective date
            by_date: dict[date, list[AmendmentOperation]] = {}
            for op in active_ops:
                assert op.effective_date is not None
                by_date.setdefault(op.effective_date, []).append(op)

            # Sort by date descending to find the latest
            sorted_dates = sorted(by_date.keys(), reverse=True)
            latest_date = sorted_dates[0]
            latest_ops = by_date[latest_date]

            if len(latest_ops) > 1:
                # Ambiguity conflict! Never guess; flag review and fall back to original
                reviews.append(
                    ReviewItem(
                        item_id=f"rev:ambig_amend:{parent_doc_id}:{path}",
                        doc_id=parent_doc_id,
                        kind="ambiguous_amendment",
                        description=(
                            f"Multiple amendments target section {path} with the same effective "
                            f"date ({latest_date.isoformat()}): {[op.source_doc_id for op in latest_ops]}."
                        ),
                    )
                )
                effective_clauses[path] = orig_clause
                effective_entries.append(
                    EffectiveClauseEntry(
                        section_path=path,
                        effective_clause_id=orig_clause.clause_id,
                        source_doc=parent_doc_id,
                        effective_from="original",
                        superseded_clause_ids=[],
                    )
                )
            else:
                chosen_op = latest_ops[0]
                # Convert extracted clause to ClauseNode, inheriting clause_type from superseded clause
                attr_dict = {a.name: a.value for a in chosen_op.extracted_clause.attributes}
                new_clause_node = ClauseNode(
                    id=f"{chosen_op.source_doc_id}:{path}",
                    clause_id=f"{chosen_op.source_doc_id}:{path}",
                    section_path=path,
                    heading=chosen_op.heading or orig_clause.heading,
                    clause_type=orig_clause.clause_type,
                    span_id=chosen_op.span_id,
                    text_hash=compute_text_hash(chosen_op.replacement_text),
                    attributes=attr_dict,
                    needs_review=chosen_op.extracted_clause.needs_review,
                    status="active",
                    effective_from=chosen_op.effective_date.isoformat(),
                    effective_to=None,
                )
                orig_clause.effective_to = chosen_op.effective_date.isoformat()
                effective_clauses[path] = new_clause_node
                effective_entries.append(
                    EffectiveClauseEntry(
                        section_path=path,
                        effective_clause_id=new_clause_node.clause_id,
                        source_doc=chosen_op.source_doc_id,
                        effective_from=chosen_op.effective_date.isoformat(),
                        superseded_clause_ids=[orig_clause.clause_id],
                    )
                )

    view = EffectiveView(
        doc_id=parent_doc_id,
        as_of=as_of_date.isoformat(),
        clauses=effective_entries,
    )
    return view, effective_clauses, reviews
