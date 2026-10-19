"""Layer 4 Knowledge Graph builder (app/graph/build.py).

Builds a deterministic in-memory KnowledgeGraph from:
- data/parsed/*.json
- data/extracted/*.json
- data/resolved/*.json
- data/parties.json

Produces:
- data/graph/graph.json (complete export)
- data/graph/effective/{doc_id}.json (materialized effective views)
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union

from ..extraction.schemas import ExtractedDocument
from ..models import ParsedDocument
from ..resolution.schemas import ResolvedDocument
from .dates import derive_dates_for_contract, parse_iso_date
from .effective import (
    AmendmentOperation,
    parse_amendment_operations,
    resolve_effective_clauses,
)
from .jurisdiction import make_jurisdiction_node, normalize_jurisdiction
from .model import (
    ClauseNode,
    ContractNode,
    DefinedTermNode,
    Edge,
    KnowledgeGraph,
    Node,
    PartyNode,
    ReviewItem,
    StandardTemplateNode,
    compute_text_hash,
)


def load_parsed_docs(data_dir: Path) -> dict[str, ParsedDocument]:
    """Load all ParsedDocument files from data/parsed/."""
    parsed_dir = data_dir / "parsed"
    docs: dict[str, ParsedDocument] = {}
    if parsed_dir.exists():
        for p in sorted(parsed_dir.glob("*.json")):
            doc = ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))
            docs[doc.doc_id] = doc
    return docs


def load_extracted_docs(data_dir: Path) -> dict[str, ExtractedDocument]:
    """Load all ExtractedDocument files from data/extracted/."""
    ext_dir = data_dir / "extracted"
    docs: dict[str, ExtractedDocument] = {}
    if ext_dir.exists():
        for p in sorted(ext_dir.glob("*.json")):
            doc = ExtractedDocument.model_validate_json(p.read_text(encoding="utf-8"))
            docs[doc.doc_id] = doc
    return docs


def load_resolved_docs(data_dir: Path) -> dict[str, ResolvedDocument]:
    """Load all ResolvedDocument files from data/resolved/."""
    res_dir = data_dir / "resolved"
    docs: dict[str, ResolvedDocument] = {}
    if res_dir.exists():
        for p in sorted(res_dir.glob("*.json")):
            doc = ResolvedDocument.model_validate_json(p.read_text(encoding="utf-8"))
            docs[doc.doc_id] = doc
    return docs


def load_party_registry(data_dir: Path) -> list[dict[str, Any]]:
    """Load raw party entries from data/parties.json."""
    parties_path = data_dir / "parties.json"
    if parties_path.exists():
        return json.loads(parties_path.read_text(encoding="utf-8"))
    return []


def _resolve_reference_target(target_raw: str, clauses_by_path: dict[str, ClauseNode]) -> Optional[ClauseNode]:
    """Resolve a reference string like '12.3' or '9.1(b)' to a clause in the same document."""
    # 1. Exact match
    if target_raw in clauses_by_path:
        return clauses_by_path[target_raw]

    # 2. Section prefix match: '9.1(b)' -> '9.1'
    m = re.match(r"^([\d.]+)", target_raw)
    if m:
        base_path = m.group(1).rstrip(".")
        if base_path in clauses_by_path:
            return clauses_by_path[base_path]

    return None


def build_graph(
    data_dir: Union[str, Path],
    as_of: Union[str, date] = "2026-10-19",
) -> KnowledgeGraph:
    """Build the complete Knowledge Graph from parsed, extracted, and resolved data.

    Deterministic: running twice with identical input yields identical output.
    """
    d_dir = Path(data_dir)
    as_of_str = str(as_of)[:10]

    parsed_docs = load_parsed_docs(d_dir)
    extracted_docs = load_extracted_docs(d_dir)
    resolved_docs = load_resolved_docs(d_dir)
    party_registry = load_party_registry(d_dir)

    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}
    review_items: list[ReviewItem] = []

    # ------------------------------------------------------------------
    # 1. Party Nodes from Registry
    # ------------------------------------------------------------------
    for p_entry in party_registry:
        pid = p_entry["party_id"]
        cname = p_entry["canonical_name"]
        aliases = p_entry.get("aliases", [])
        p_node = PartyNode(id=pid, party_id=pid, canonical_name=cname, aliases=aliases)
        nodes[pid] = Node(id=pid, label="Party", properties=p_node.model_dump())

    # Map doc_id -> {role_label -> party_id}
    doc_roles: dict[str, dict[str, str]] = {}
    doc_parties: dict[str, list[tuple[str, Optional[str], Optional[str]]]] = {}
    # doc_id -> list of (party_id, role_label, span_id)

    for doc_id, res in resolved_docs.items():
        doc_roles[doc_id] = {}
        if res.role_mapping.customer_party_id:
            doc_roles[doc_id]["Customer"] = res.role_mapping.customer_party_id
        if res.role_mapping.supplier_party_id:
            doc_roles[doc_id]["Supplier"] = res.role_mapping.supplier_party_id

        doc_parties[doc_id] = []
        for p_mention in res.parties:
            doc_parties[doc_id].append(
                (p_mention.party_id, p_mention.role_label, p_mention.evidence.span.span_id)
            )

    # ------------------------------------------------------------------
    # 2. Contract and Clause Nodes (Initial Pass)
    # ------------------------------------------------------------------
    clauses_by_doc: dict[str, dict[str, ClauseNode]] = {}
    clause_objects_by_doc: dict[str, list[ClauseNode]] = {}

    for doc_id, p_doc in parsed_docs.items():
        ext_doc = extracted_docs.get(doc_id)
        res_doc = resolved_docs.get(doc_id)

        eff_date_str = p_doc.effective_date.isoformat() if p_doc.effective_date else None
        c_node = ContractNode(
            id=doc_id,
            doc_id=doc_id,
            version=p_doc.version,
            doc_type=p_doc.doc_type,
            title=p_doc.title,
            effective_date=eff_date_str,
            status="active",
        )
        nodes[doc_id] = Node(id=doc_id, label="Contract", properties=c_node.model_dump())

        # Build map of extracted clause info
        ext_by_path = {c.section_path: c for c in ext_doc.clauses} if ext_doc else {}
        match_by_path = {m.section_path: m for m in res_doc.template_matches} if res_doc else {}

        clauses_by_doc[doc_id] = {}
        clause_objects_by_doc[doc_id] = []

        for p_clause in p_doc.clauses:
            path = p_clause.section_path
            ext_c = ext_by_path.get(path)
            tpl_match = match_by_path.get(path)

            c_type = ext_c.clause_type.value if ext_c else "other"
            needs_rev = ext_c.needs_review if ext_c else False
            attr_dict = {a.name: a.value for a in ext_c.attributes} if ext_c else {}

            # Detect amendment operation clauses ("Section X is deleted and replaced...")
            # These must not match standard templates and must get clause_type="other"
            if p_doc.doc_type == "AMENDMENT":
                m_op = re.search(
                    r'Section\s+([\d.]+)\s+is\s+(?:deleted\s+and\s+replaced\s+with|amended\s+to\s+read)',
                    p_clause.text,
                    re.IGNORECASE,
                )
                if m_op or (ext_c and re.search(r"\b(?:deleted\s+and\s+replaced|amended\s+to\s+read)\b", p_clause.text, re.IGNORECASE)):
                    c_type = "other"
                    target_sec = m_op.group(1).strip() if m_op else ""
                    attr_dict = {"operation": "replace", "target_section": target_sec}
                    tpl_match = None

            cid = f"{doc_id}:{path}"
            cl_node = ClauseNode(
                id=cid,
                clause_id=cid,
                section_path=path,
                heading=p_clause.heading,
                clause_type=c_type,
                span_id=p_clause.span.span_id,
                text_hash=compute_text_hash(p_clause.text),
                attributes=attr_dict,
                needs_review=needs_rev,
                status="active",
                effective_from=eff_date_str,
                effective_to=None,
            )
            clauses_by_doc[doc_id][path] = cl_node
            clause_objects_by_doc[doc_id].append(cl_node)
            nodes[cid] = Node(id=cid, label="Clause", properties=cl_node.model_dump())

            # Contract -CONTAINS-> Clause edge
            edge_id = f"{doc_id}->CONTAINS->{cid}"
            edges[edge_id] = Edge(
                id=edge_id,
                type="CONTAINS",
                source=doc_id,
                target=cid,
                provenance=p_clause.span.span_id,
                evidence_span_ids=[p_clause.span.span_id],
            )

            # Defined terms: Clause -DEFINES-> DefinedTerm
            if ext_c and ext_c.defined_terms:
                for dt in ext_c.defined_terms:
                    dt_id = f"{doc_id}:term:{dt.term.lower().replace(' ', '_')}"
                    dt_node = DefinedTermNode(
                        id=dt_id,
                        term=dt.term,
                        definition=dt.definition,
                        span_id=dt.evidence.span.span_id,
                    )
                    nodes[dt_id] = Node(
                        id=dt_id, label="DefinedTerm", properties=dt_node.model_dump()
                    )
                    d_edge_id = f"{cid}->DEFINES->{dt_id}"
                    edges[d_edge_id] = Edge(
                        id=d_edge_id,
                        type="DEFINES",
                        source=cid,
                        target=dt_id,
                        provenance=dt.evidence.span.span_id,
                        evidence_span_ids=[dt.evidence.span.span_id],
                    )

            # Template Match: Clause -INSTANCE_OF-> StandardTemplate
            if tpl_match and tpl_match.template_id:
                tid = tpl_match.template_id
                if tid not in nodes:
                    st_node = StandardTemplateNode(
                        id=tid,
                        template_id=tid,
                        clause_type=tpl_match.clause_type,
                        version="1",
                    )
                    nodes[tid] = Node(
                        id=tid, label="StandardTemplate", properties=st_node.model_dump()
                    )
                inst_edge_id = f"{cid}->INSTANCE_OF->{tid}"
                edges[inst_edge_id] = Edge(
                    id=inst_edge_id,
                    type="INSTANCE_OF",
                    source=cid,
                    target=tid,
                    provenance="matched",
                    properties={"score": tpl_match.score, "method": tpl_match.method},
                )

    # ------------------------------------------------------------------
    # 3. Document Linking (PARENT_OF) and Role Inheritance
    # ------------------------------------------------------------------
    parent_links: dict[str, str] = {}  # child_doc_id -> parent_doc_id

    for doc_id, p_doc in parsed_docs.items():
        if not p_doc.parent_ref:
            continue
        pref = p_doc.parent_ref
        pref_date_str = pref.date.isoformat() if pref.date else None

        # Child's party IDs
        child_pids = {pid for pid, _, _ in doc_parties.get(doc_id, [])}

        candidates: list[str] = []
        for cand_id, cand_doc in parsed_docs.items():
            if cand_id == doc_id:
                continue
            if cand_doc.doc_type != pref.doc_type:
                continue
            cand_date_str = cand_doc.effective_date.isoformat() if cand_doc.effective_date else None
            if cand_date_str != pref_date_str:
                continue
            # Must share at least one party
            cand_pids = {pid for pid, _, _ in doc_parties.get(cand_id, [])}
            if child_pids & cand_pids:
                candidates.append(cand_id)

        if len(candidates) == 1:
            parent_id = candidates[0]
            parent_links[doc_id] = parent_id
            edge_id = f"{parent_id}->PARENT_OF->{doc_id}"
            edges[edge_id] = Edge(
                id=edge_id,
                type="PARENT_OF",
                source=parent_id,
                target=doc_id,
                provenance="resolved",
                properties={"parent_ref_raw": pref.raw},
            )

            # Role inheritance: if child has no role labels, inherit from parent
            parent_role_map = doc_roles.get(parent_id, {})
            child_role_map = doc_roles.get(doc_id, {})
            if parent_role_map and not child_role_map:
                # Check party sets match
                parent_pids = {pid for pid, _, _ in doc_parties.get(parent_id, [])}
                if child_pids == parent_pids:
                    # Inherit role mapping
                    doc_roles[doc_id] = dict(parent_role_map)
                    # Update child's party list roles with provenance="inherited"
                    rev_parent = {v: k for k, v in parent_role_map.items()}
                    new_list = []
                    for pid, _, sp_id in doc_parties.get(doc_id, []):
                        inh_role = rev_parent.get(pid)
                        new_list.append((pid, inh_role, "inherited"))
                    doc_parties[doc_id] = new_list

        elif len(candidates) == 0:
            review_items.append(
                ReviewItem(
                    item_id=f"rev:parent_unlinked:{doc_id}",
                    doc_id=doc_id,
                    kind="unlinked_parent",
                    description=(
                        f"Could not link parent {pref.doc_type} dated {pref_date_str} for {doc_id}."
                    ),
                )
            )
        else:
            review_items.append(
                ReviewItem(
                    item_id=f"rev:parent_ambiguous:{doc_id}",
                    doc_id=doc_id,
                    kind="ambiguous_parent",
                    description=(
                        f"Multiple candidate parents for {doc_id} matching {pref.doc_type} "
                        f"dated {pref_date_str}: {candidates}."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # 4. Create PARTY_TO Edges (including inherited roles)
    # ------------------------------------------------------------------
    for doc_id, party_list in doc_parties.items():
        for pid, role, sp_or_inh in party_list:
            edge_id = f"{pid}->PARTY_TO->{doc_id}"
            prov = "inherited" if sp_or_inh == "inherited" else (sp_or_inh or "extracted")
            ev_spans = [] if sp_or_inh == "inherited" or not sp_or_inh else [sp_or_inh]
            edges[edge_id] = Edge(
                id=edge_id,
                type="PARTY_TO",
                source=pid,
                target=doc_id,
                properties={"role": role} if role else {},
                provenance=prov,
                evidence_span_ids=ev_spans,
            )

    # ------------------------------------------------------------------
    # 5. Amendment Parsing, Effective-State Resolution, and SUPERSEDES
    # ------------------------------------------------------------------
    amend_ops_by_parent: dict[str, list[AmendmentOperation]] = {}

    for doc_id, p_doc in parsed_docs.items():
        if p_doc.doc_type != "AMENDMENT":
            continue

        ops, a_reviews = parse_amendment_operations(p_doc)
        review_items.extend(a_reviews)

        parent_id = parent_links.get(doc_id)
        if parent_id:
            amend_ops_by_parent.setdefault(parent_id, []).extend(ops)
            # Contract -AMENDS-> Contract edge
            edge_id = f"{doc_id}->AMENDS->{parent_id}"
            edges[edge_id] = Edge(
                id=edge_id,
                type="AMENDS",
                source=doc_id,
                target=parent_id,
                provenance="resolved",
            )
        else:
            review_items.append(
                ReviewItem(
                    item_id=f"rev:amend_no_parent:{doc_id}",
                    doc_id=doc_id,
                    kind="amendment_without_parent",
                    description=f"Amendment {doc_id} could not be linked to any parent contract.",
                )
            )

    # Determine effective clauses per contract as of as_of
    effective_clauses_by_doc: dict[str, dict[str, ClauseNode]] = {}
    effective_views_dir = d_dir / "graph" / "effective"
    effective_views_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in parsed_docs:
        orig_clauses = clause_objects_by_doc.get(doc_id, [])
        amend_ops = amend_ops_by_parent.get(doc_id, [])

        eff_view, eff_map, eff_reviews = resolve_effective_clauses(
            doc_id, orig_clauses, amend_ops, as_of
        )
        review_items.extend(eff_reviews)
        effective_clauses_by_doc[doc_id] = eff_map

        # Save materialized effective view
        view_path = effective_views_dir / f"{doc_id}.json"
        view_path.write_text(eff_view.model_dump_json(indent=2), encoding="utf-8")

        # Mark superseded clauses and add SUPERSEDES edges
        for entry in eff_view.clauses:
            if entry.superseded_clause_ids:
                # The effective clause is from an amendment!
                repl_node = eff_map[entry.section_path]
                # Ensure the replacement clause node is in nodes
                nodes[repl_node.id] = Node(
                    id=repl_node.id, label="Clause", properties=repl_node.model_dump()
                )

                # Connect amendment doc -CONTAINS-> repl_node if not already present
                cont_edge = f"{entry.source_doc}->CONTAINS->{repl_node.id}"
                if cont_edge not in edges:
                    edges[cont_edge] = Edge(
                        id=cont_edge,
                        type="CONTAINS",
                        source=entry.source_doc,
                        target=repl_node.id,
                        provenance=repl_node.span_id,
                        evidence_span_ids=[repl_node.span_id],
                    )

                for orig_cid in entry.superseded_clause_ids:
                    # Update status of original clause to 'superseded'
                    if orig_cid in nodes:
                        orig_props = dict(nodes[orig_cid].properties)
                        orig_props["status"] = "superseded"
                        nodes[orig_cid] = Node(
                            id=orig_cid, label="Clause", properties=orig_props
                        )

                    # Add SUPERSEDES edge
                    sup_edge_id = f"{repl_node.id}->SUPERSEDES->{orig_cid}"
                    edges[sup_edge_id] = Edge(
                        id=sup_edge_id,
                        type="SUPERSEDES",
                        source=repl_node.id,
                        target=orig_cid,
                        provenance="amendment",
                        evidence_span_ids=[repl_node.span_id],
                    )

                    # Inherit INSTANCE_OF edge from the original clause to the same template
                    # so the replacement is linked to the same StandardTemplate node.
                    for eid, e in list(edges.items()):
                        if e.type == "INSTANCE_OF" and e.source == orig_cid:
                            new_inst_id = f"{repl_node.id}->INSTANCE_OF->{e.target}"
                            if new_inst_id not in edges:
                                edges[new_inst_id] = Edge(
                                    id=new_inst_id,
                                    type="INSTANCE_OF",
                                    source=repl_node.id,
                                    target=e.target,
                                    provenance="inherited_from_superseded",
                                    properties={**e.properties, "inherited": True},
                                )

    # ------------------------------------------------------------------
    # 6. Reference Resolution (Internal References)
    # ------------------------------------------------------------------
    for doc_id, ext_doc in extracted_docs.items():
        p_doc = parsed_docs.get(doc_id)
        doc_clauses = clauses_by_doc.get(doc_id, {})
        for ext_c in ext_doc.clauses:
            source_cid = f"{doc_id}:{ext_c.section_path}"
            for ref in ext_c.references:
                target_raw = ref.target.strip()

                # Rule 1: Exact self-reference ("this Section 9.2" inside 9.2)
                if target_raw == ext_c.section_path or target_raw.startswith(
                    f"{ext_c.section_path}("
                ):
                    continue  # silently ignore

                # Rule 2: Ancestor / self-scope reference (e.g. "this Section 9" written
                # inside clause 9.2).  The referring clause IS a descendant of the target,
                # so the target is an ancestor path: ignore it with no review item.
                if ext_c.section_path.startswith(f"{target_raw}."):
                    continue  # self-scope reference – silently ignore

                # Rule 3: Amendment operation clauses reference the parent-contract
                # sections they supersede.  Those relationships are already captured
                # by SUPERSEDES edges, so we do not add a REFERENCES edge or a review item.
                if p_doc and p_doc.doc_type == "AMENDMENT":
                    continue  # all cross-doc references in amendments are modelled by SUPERSEDES

                target_clause = _resolve_reference_target(target_raw, doc_clauses)
                if target_clause:
                    ref_edge_id = f"{source_cid}->REFERENCES->{target_clause.id}"
                    edges[ref_edge_id] = Edge(
                        id=ref_edge_id,
                        type="REFERENCES",
                        source=source_cid,
                        target=target_clause.id,
                        properties={"target_raw": target_raw, "quote": ref.evidence.quote},
                        provenance=ref.evidence.span.span_id,
                        evidence_span_ids=[ref.evidence.span.span_id],
                    )
                else:
                    # Rule 4: Schedule/exhibit references (e.g. "Schedule A") that have
                    # no clause node are low-severity – they are expected gaps.
                    is_schedule = (
                        ref.kind == "schedule"
                        or re.match(r'^[A-Z]$', target_raw)
                        or target_raw.lower().startswith("schedule")
                        or target_raw.lower().startswith("exhibit")
                    )
                    severity = "info" if is_schedule else "warning"
                    review_items.append(
                        ReviewItem(
                            item_id=f"rev:ref_unres:{source_cid}:{target_raw}",
                            doc_id=doc_id,
                            kind="unresolvable_reference",
                            description=(
                                f"Clause {ext_c.section_path} in {doc_id} references "
                                f"{target_raw!r} which cannot be resolved to any clause."
                            ),
                            span_id=ref.evidence.span.span_id,
                            severity=severity,
                        )
                    )

    # ------------------------------------------------------------------
    # 7. IMPOSES_ON and BENEFITS Edges (Indemnification Roles)
    # ------------------------------------------------------------------
    for doc_id, ext_doc in extracted_docs.items():
        role_map = doc_roles.get(doc_id, {})
        for ext_c in ext_doc.clauses:
            if ext_c.clause_type.value != "indemnification":
                continue
            cid = f"{doc_id}:{ext_c.section_path}"
            indemnitor_val = None
            indemnitee_val = None
            indemnitor_span = None
            indemnitee_span = None

            for attr in ext_c.attributes:
                if attr.name == "indemnitor":
                    indemnitor_val = str(attr.value)
                    if attr.evidence:
                        indemnitor_span = attr.evidence.span.span_id
                elif attr.name == "indemnitee":
                    indemnitee_val = str(attr.value)
                    if attr.evidence:
                        indemnitee_span = attr.evidence.span.span_id

            if indemnitor_val and indemnitor_val in role_map:
                target_pid = role_map[indemnitor_val]
                imp_edge_id = f"{cid}->IMPOSES_ON->{target_pid}"
                edges[imp_edge_id] = Edge(
                    id=imp_edge_id,
                    type="IMPOSES_ON",
                    source=cid,
                    target=target_pid,
                    provenance=indemnitor_span or "resolved",
                    evidence_span_ids=[indemnitor_span] if indemnitor_span else [],
                )

            if indemnitee_val and indemnitee_val in role_map:
                target_pid = role_map[indemnitee_val]
                ben_edge_id = f"{cid}->BENEFITS->{target_pid}"
                edges[ben_edge_id] = Edge(
                    id=ben_edge_id,
                    type="BENEFITS",
                    source=cid,
                    target=target_pid,
                    provenance=indemnitee_span or "resolved",
                    evidence_span_ids=[indemnitee_span] if indemnitee_span else [],
                )

    # ------------------------------------------------------------------
    # 8. Derived Dates & Jurisdiction from Effective State
    # ------------------------------------------------------------------
    for doc_id, p_doc in parsed_docs.items():
        eff_map = effective_clauses_by_doc.get(doc_id, {})
        # Flatten all attributes across effective clauses
        eff_attrs: dict[str, Any] = {}
        attr_spans: dict[str, str] = {}

        gov_law_val: Optional[str] = None
        gov_law_span: Optional[str] = None

        # Gather attributes from effective clauses
        for cnode in eff_map.values():
            for k, v in cnode.attributes.items():
                if k not in eff_attrs or v is not None:
                    eff_attrs[k] = v
                    attr_spans[k] = cnode.span_id
                if k == "governing_law" and v:
                    gov_law_val = str(v)
                    gov_law_span = cnode.span_id

        # Derived dates; amendments only get an effective DateEvent (doc_type guards this)
        date_events = derive_dates_for_contract(
            doc_id=doc_id,
            effective_date_str=p_doc.effective_date.isoformat() if p_doc.effective_date else None,
            attributes=eff_attrs,
            attribute_spans=attr_spans,
            as_of=as_of_str,
            doc_type=p_doc.doc_type,
        )

        for dev in date_events:
            nodes[dev.id] = Node(
                id=dev.id, label="DateEvent", properties=dev.model_dump()
            )
            d_edge_id = f"{doc_id}->HAS_DATE->{dev.id}"
            edges[d_edge_id] = Edge(
                id=d_edge_id,
                type="HAS_DATE",
                source=doc_id,
                target=dev.id,
                provenance="derived",
                evidence_span_ids=dev.input_span_ids,
            )

        # Jurisdiction
        if gov_law_val:
            j_node = make_jurisdiction_node(gov_law_val)
            if j_node:
                if j_node.id not in nodes:
                    nodes[j_node.id] = Node(
                        id=j_node.id, label="Jurisdiction", properties=j_node.model_dump()
                    )
                j_edge_id = f"{doc_id}->GOVERNED_BY->{j_node.id}"
                edges[j_edge_id] = Edge(
                    id=j_edge_id,
                    type="GOVERNED_BY",
                    source=doc_id,
                    target=j_node.id,
                    provenance=gov_law_span or "extracted",
                    evidence_span_ids=[gov_law_span] if gov_law_span else [],
                )

    # ------------------------------------------------------------------
    # 9. Assemble Graph with Deterministic Ordering & Export
    # ------------------------------------------------------------------
    sorted_nodes = sorted(nodes.values(), key=lambda n: (n.label, n.id))
    sorted_edges = sorted(edges.values(), key=lambda e: (e.type, e.source, e.target))
    sorted_reviews = sorted(review_items, key=lambda r: (r.doc_id, r.kind, r.item_id))

    graph = KnowledgeGraph(
        as_of=as_of_str,
        nodes=sorted_nodes,
        edges=sorted_edges,
        review_items=sorted_reviews,
    )

    # Export to data/graph/graph.json
    out_dir = d_dir / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "graph.json"
    out_file.write_text(graph.model_dump_json(indent=2), encoding="utf-8")

    return graph
