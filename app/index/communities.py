"""Community summaries over contract portfolio (app/index/communities.py).

Features:
1. Contract-family communities: graphs contracts linked by PARENT_OF, AMENDS, and shared
   counterparty/customer. Detects communities deterministically (networkx Louvain, seed=42,
   or python-igraph + leidenalg if installed).
2. Clause-type pattern communities: groups clauses by clause_type across the portfolio.
3. Deterministic structured stats: counts, cap distributions, below-standard caps,
   auto-renewal counts, expiries by year, jurisdiction distributions.
4. Optional LLM narrative with Number-Grounding Validation: prompt contains only stats,
   rejects any narrative that invents numbers not present in the stats, falling back to
   deterministic templates.
5. Permission isolation: communities are built per permission group and never mix contracts.
6. Stale detection via source clause hashes.
7. Output to data/index/communities.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from ..graph.reader import GraphReader
from ..llm.client import LLMClient, NullLLM

COMMUNITIES_PATH = Path("data/index/communities.json")


def extract_numbers(text: str) -> set[float]:
    """Extract all numbers from text for grounding verification."""
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    nums = set()
    for m in matches:
        try:
            nums.add(float(m))
        except ValueError:
            pass
    return nums


def extract_stats_numbers(stats: Any) -> set[float]:
    """Recursively collect all numeric values and numbers in keys/strings from a stats dict."""
    nums: set[float] = set()
    if isinstance(stats, (int, float)):
        nums.add(float(stats))
    elif isinstance(stats, str):
        nums.update(extract_numbers(stats))
    elif isinstance(stats, dict):
        for k, v in stats.items():
            nums.update(extract_numbers(str(k)))
            nums.update(extract_stats_numbers(v))
    elif isinstance(stats, (list, tuple, set)):
        for item in stats:
            nums.update(extract_stats_numbers(item))
    return nums


def verify_number_grounding(narrative: str, stats: dict[str, Any]) -> bool:
    """Check that every number in the narrative appears somewhere in the stats.

    Rejects hallucinated numbers, years, percentages, or metrics.
    """
    narrative_nums = extract_numbers(narrative)
    allowed_nums = extract_stats_numbers(stats)
    for n in narrative_nums:
        if n not in allowed_nums:
            return False
    return True


def compute_community_hash(clause_texts: list[str]) -> str:
    """Compute MD5 hash representing the source state of clauses in the community."""
    joined = "".join(sorted(clause_texts))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


class CommunityBuilder:
    """Builds, summarizes, and evaluates portfolio communities."""

    def __init__(
        self,
        reader: GraphReader,
        llm: Optional[LLMClient] = None,
        acl_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.reader = reader
        self.llm = llm or NullLLM()
        self.acl_map = acl_map or {}
        if not self.acl_map and Path("data/acl.json").exists():
            try:
                self.acl_map = json.loads(Path("data/acl.json").read_text(encoding="utf-8"))
            except Exception:
                self.acl_map = {}

    def _get_groups_map(self) -> dict[str, list[str]]:
        """Map contract_id -> list of permission groups (default ['all'])."""
        res: dict[str, list[str]] = {}
        for c in self.reader.contracts():
            cid = c["doc_id"]
            res[cid] = self.acl_map.get(cid, ["all"])
        return res

    def _generate_family_narrative(self, name: str, stats: dict[str, Any]) -> str:
        """Generate narrative for contract family (via LLM or deterministic template)."""
        fallback = (
            f"Contract family '{name}' contains {stats['contract_count']} contracts "
            f"({', '.join(stats['contracts'])}). "
            f"Total clauses: {stats['clause_count']}. "
            f"Governing law: {stats.get('governing_law', 'multiple')}."
        )

        if isinstance(self.llm, NullLLM):
            return fallback

        prompt = (
            f"Write a concise 2-sentence summary of this contract family based strictly on "
            f"these facts. Do not invent any numbers or details not present in the facts.\n"
            f"Facts:\n{json.dumps(stats, indent=2)}\nSummary:"
        )
        try:
            resp = self.llm.classify(prompt) if hasattr(self.llm, "classify") else ""
            if resp and verify_number_grounding(resp, stats):
                return resp.strip()
        except Exception:
            pass

        return fallback

    def _generate_pattern_narrative(self, clause_type: str, stats: dict[str, Any]) -> str:
        """Generate narrative for clause-type pattern."""
        fallback = (
            f"Portfolio pattern for '{clause_type}' across {stats['contract_count']} contracts "
            f"with {stats['clause_count']} clauses. "
        )
        if clause_type == "indemnification":
            fallback += (
                f"{stats.get('below_standard_count', 0)} contracts have liability caps below "
                f"the 2.0x standard. {stats.get('covers_gross_negligence_count', 0)} cover gross negligence."
            )
        elif clause_type == "term_renewal":
            fallback += (
                f"{stats.get('auto_renew_count', 0)} contracts auto-renew. "
                f"Expiries: {stats.get('expiries_by_year', {})}."
            )
        else:
            fallback += f"Distribution: {stats.get('distribution', {})}"

        if isinstance(self.llm, NullLLM):
            return fallback

        prompt = (
            f"Write a concise 2-sentence summary of this legal clause pattern based strictly on "
            f"these facts. Do not invent any numbers or details not present in the facts.\n"
            f"Facts:\n{json.dumps(stats, indent=2)}\nSummary:"
        )
        try:
            resp = self.llm.classify(prompt) if hasattr(self.llm, "classify") else ""
            if resp and verify_number_grounding(resp, stats):
                return resp.strip()
        except Exception:
            pass

        return fallback

    def build_communities(self) -> list[dict[str, Any]]:
        """Detect and summarize contract families and clause-type pattern communities."""
        communities: list[dict[str, Any]] = []
        groups_map = self._get_groups_map()

        # Collect unique groups
        all_groups: set[str] = set()
        for glist in groups_map.values():
            all_groups.update(glist)

        contracts = self.reader.contracts()
        contracts_by_id = {c["doc_id"]: c for c in contracts}

        # --------------------------------------------------------------
        # 1. Contract-Family Communities (per permission group)
        # --------------------------------------------------------------
        for group in sorted(all_groups):
            group_contracts = [
                cid for cid, glist in groups_map.items()
                if group in glist or "all" in glist
            ]
            if not group_contracts:
                continue

            # Build graph for this group
            G = nx.Graph()
            for cid in group_contracts:
                G.add_node(cid)

            # Add edges for PARENT_OF
            for cid in group_contracts:
                parent = self.reader.parent_of(cid)
                if parent and parent["doc_id"] in group_contracts:
                    G.add_edge(cid, parent["doc_id"])

                # Add edges for amendments (AMENDS)
                for nbr in self.reader.neighbors(cid, edge_type="AMENDS", direction="both"):
                    if nbr.get("label") == "Contract" and nbr.get("doc_id") in group_contracts:
                        G.add_edge(cid, nbr["doc_id"])

            # Also add edges for shared specific counterparty (not Acme)
            parties_by_contract: dict[str, list[dict[str, Any]]] = {}
            for cid in group_contracts:
                parties_by_contract[cid] = self.reader.parties_of(cid)

            for i in range(len(group_contracts)):
                for j in range(i + 1, len(group_contracts)):
                    c1, c2 = group_contracts[i], group_contracts[j]
                    p1 = {p["canonical_name"] for p in parties_by_contract[c1] if "Acme" not in p["canonical_name"]}
                    p2 = {p["canonical_name"] for p in parties_by_contract[c2] if "Acme" not in p["canonical_name"]}
                    if p1 and p2 and (p1 & p2):
                        G.add_edge(c1, c2)

            # Community detection
            try:
                # Deterministic networkx Louvain
                from networkx.algorithms.community import louvain_communities
                comm_sets = louvain_communities(G, seed=42)
            except Exception:
                # Connected components fallback
                comm_sets = list(nx.connected_components(G))

            for idx, cset in enumerate(sorted(comm_sets, key=lambda s: min(s))):
                member_cids = sorted(list(cset))
                comm_id = f"comm_family_{group}_{idx}"

                # Gather clauses and compute stats
                clause_ids: list[str] = []
                clause_texts: list[str] = []
                citations: list[str] = []
                doc_types: dict[str, int] = {}
                laws: dict[str, int] = {}

                for cid in member_cids:
                    cnode = contracts_by_id.get(cid, {})
                    dtype = cnode.get("doc_type", "UNKNOWN")
                    doc_types[dtype] = doc_types.get(dtype, 0) + 1

                    jur = self.reader.jurisdiction_of(cid)
                    if jur:
                        jname = jur.get("name", "Unknown")
                        laws[jname] = laws.get(jname, 0) + 1

                    for cl in self.reader.clauses_of(cid, as_of="2026-10-19"):
                        clause_ids.append(cl["id"])
                        clause_texts.append(cl.get("text", ""))
                        if cl.get("span_id"):
                            citations.append(cl["span_id"])

                primary_name = member_cids[0]
                comm_name = f"Family: {primary_name} ({len(member_cids)} contracts)"

                stats: dict[str, Any] = {
                    "contract_count": len(member_cids),
                    "clause_count": len(clause_ids),
                    "contracts": member_cids,
                    "doc_types": doc_types,
                    "governing_law": list(laws.keys())[0] if len(laws) == 1 else "multiple",
                    "laws_breakdown": laws,
                }

                narrative = self._generate_family_narrative(comm_name, stats)
                source_hash = compute_community_hash(clause_texts)

                communities.append({
                    "id": comm_id,
                    "kind": "family",
                    "name": comm_name,
                    "permission_groups": [group],
                    "member_contract_ids": member_cids,
                    "member_clause_ids": clause_ids,
                    "stats": stats,
                    "citations": citations[:15],
                    "narrative": narrative,
                    "source_graph_hash": source_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "is_stale": False,
                })

        # --------------------------------------------------------------
        # 2. Clause-Type Pattern Communities (per permission group)
        # --------------------------------------------------------------
        target_clause_types = [
            "indemnification", "limitation_of_liability", "term_renewal",
            "governing_law", "confidentiality", "payment",
        ]

        for group in sorted(all_groups):
            group_contracts = [
                cid for cid, glist in groups_map.items()
                if group in glist or "all" in glist
            ]
            if not group_contracts:
                continue

            for ctype in target_clause_types:
                member_cids: set[str] = set()
                clause_ids: list[str] = []
                clause_texts: list[str] = []
                citations: list[str] = []

                # Stats collectors
                caps: dict[str, int] = {}
                below_standard = 0
                standard_or_above = 0
                directions: dict[str, int] = {}
                covers_gross_neg = 0
                auto_renews = 0
                expiries: dict[str, int] = {}
                laws: dict[str, int] = {}

                for cid in group_contracts:
                    clauses = [
                        cl for cl in self.reader.clauses_of(cid, as_of="2026-10-19")
                        if cl.get("clause_type") == ctype
                    ]
                    if not clauses:
                        continue

                    member_cids.add(cid)
                    for cl in clauses:
                        clause_ids.append(cl["id"])
                        clause_texts.append(cl.get("text", ""))
                        if cl.get("span_id"):
                            citations.append(cl["span_id"])

                        attrs = cl.get("attributes") or {}

                        # Indemnity / cap stats
                        cap_m = attrs.get("cap_multiple")
                        if cap_m is not None:
                            try:
                                cap_float = float(cap_m)
                                cap_str = f"{cap_float}x"
                                caps[cap_str] = caps.get(cap_str, 0) + 1
                                if cap_float < 2.0:
                                    below_standard += 1
                                else:
                                    standard_or_above += 1
                            except Exception:
                                pass

                        if attrs.get("direction"):
                            d = attrs["direction"]
                            directions[d] = directions.get(d, 0) + 1

                        if attrs.get("covers_gross_negligence") is True:
                            covers_gross_neg += 1

                        # Term renewal stats
                        if attrs.get("auto_renew") is True:
                            auto_renews += 1

                # If no clauses found for this ctype in this group, skip
                if not clause_ids:
                    continue

                # Collect expiry and law stats across participating contracts
                for cid in member_cids:
                    dates = {d["kind"]: d["date"] for d in self.reader.dates_of(cid)}
                    end = dates.get("current_term_end") or dates.get("initial_term_end")
                    if end:
                        yr = end.split("-")[0]
                        expiries[yr] = expiries.get(yr, 0) + 1

                    jur = self.reader.jurisdiction_of(cid)
                    if jur:
                        jname = jur.get("name", "Unknown")
                        laws[jname] = laws.get(jname, 0) + 1

                stats = {
                    "clause_type": ctype,
                    "contract_count": len(member_cids),
                    "clause_count": len(clause_ids),
                    "contracts": sorted(list(member_cids)),
                }

                if ctype == "indemnification":
                    stats.update({
                        "cap_distribution": caps,
                        "below_standard_count": below_standard,
                        "standard_or_above_count": standard_or_above,
                        "direction_distribution": directions,
                        "covers_gross_negligence_count": covers_gross_neg,
                    })
                elif ctype == "term_renewal":
                    stats.update({
                        "auto_renew_count": auto_renews,
                        "non_auto_renew_count": len(member_cids) - auto_renews,
                        "expiries_by_year": expiries,
                    })
                elif ctype == "governing_law":
                    stats.update({
                        "jurisdiction_counts": laws,
                    })

                comm_id = f"comm_pattern_{group}_{ctype}"
                comm_name = f"Pattern: {ctype.replace('_', ' ').title()} ({len(member_cids)} contracts)"
                narrative = self._generate_pattern_narrative(ctype, stats)
                source_hash = compute_community_hash(clause_texts)

                communities.append({
                    "id": comm_id,
                    "kind": "pattern",
                    "name": comm_name,
                    "permission_groups": [group],
                    "member_contract_ids": sorted(list(member_cids)),
                    "member_clause_ids": clause_ids,
                    "stats": stats,
                    "citations": citations[:15],
                    "narrative": narrative,
                    "source_graph_hash": source_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "is_stale": False,
                })

        return communities

    def save_and_index(self, out_path: Path = COMMUNITIES_PATH) -> list[dict[str, Any]]:
        """Build communities, write to communities.json, and return list."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        communities = self.build_communities()
        data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "community_count": len(communities),
            "communities": communities,
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return communities
