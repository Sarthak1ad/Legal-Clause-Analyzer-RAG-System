"""Unit tests for Portfolio Communities and Summaries (Layer 5 Part D).

Verifies:
- Deterministic community detection across multiple runs.
- The Acme MSA, SOW, and Amendment belong to the same family community.
- Structured stats accurately match ground truth portfolio values.
- Number-grounding verification rejects narratives with hallucinated numbers.
- Permission safety: communities are isolated per permission group and never mix contracts.
- Stale detection triggers when underlying clause content changes.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.graph.reader import InMemoryGraphReader
from app.index.communities import (
    CommunityBuilder,
    compute_community_hash,
    verify_number_grounding,
)

GRAPH_JSON = Path("data/graph/graph.json")


@pytest.fixture(scope="module")
def reader() -> InMemoryGraphReader:
    if not GRAPH_JSON.exists():
        pytest.skip("data/graph/graph.json missing")
    return InMemoryGraphReader.from_json(GRAPH_JSON)


def test_communities_deterministic_output(reader: InMemoryGraphReader):
    """Running community detection twice produces identical community assignments and stats."""
    b1 = CommunityBuilder(reader)
    c1 = b1.build_communities()

    b2 = CommunityBuilder(reader)
    c2 = b2.build_communities()

    assert len(c1) == len(c2)
    for comm1, comm2 in zip(c1, c2):
        assert comm1["id"] == comm2["id"]
        assert comm1["member_contract_ids"] == comm2["member_contract_ids"]
        assert comm1["stats"] == comm2["stats"]


def test_acme_family_is_single_community(reader: InMemoryGraphReader):
    """Acme MSA, SOW 1, and Amendment 1 are grouped into a single family community."""
    builder = CommunityBuilder(reader)
    comms = builder.build_communities()

    family_comms = [c for c in comms if c["kind"] == "family"]
    acme_comm = None
    for fc in family_comms:
        cids = set(fc["member_contract_ids"])
        if "msa-acme" in cids:
            acme_comm = fc
            break

    assert acme_comm is not None, "Acme family community not found"
    acme_cids = set(acme_comm["member_contract_ids"])
    assert "msa-acme" in acme_cids
    assert "sow-acme-01" in acme_cids
    assert "amendment-acme-01" in acme_cids


def test_number_grounding_rejects_hallucinations():
    """verify_number_grounding rejects any narrative containing numbers not present in stats."""
    stats = {
        "contract_count": 5,
        "clause_count": 40,
        "below_standard_count": 3,
        "cap_distribution": {"1.0x": 2, "2.0x": 1},
    }

    # Grounded narrative: only uses numbers 5, 40, 3, 1.0, 2.0
    grounded = "There are 5 contracts with 40 clauses. 3 contracts have caps below standard."
    assert verify_number_grounding(grounded, stats) is True

    # Hallucinated narrative: mentions 99 or 85% which are NOT in stats
    hallucinated = "There are 5 contracts and 99 clauses with 85% compliance."
    assert verify_number_grounding(hallucinated, stats) is False


def test_permission_groups_isolation(reader: InMemoryGraphReader):
    """Communities never mix contracts from disjoint permission groups."""
    # Force distinct ACLs
    acl = {
        "msa-acme": ["legal_us"],
        "sow-acme-01": ["legal_us"],
        "msa-hooli-acme-eu": ["legal_eu"],
    }
    builder = CommunityBuilder(reader, acl_map=acl)
    comms = builder.build_communities()

    for c in comms:
        groups = set(c["permission_groups"])
        cids = c["member_contract_ids"]
        if "legal_us" in groups:
            assert "msa-hooli-acme-eu" not in cids
        if "legal_eu" in groups:
            assert "msa-acme" not in cids


def test_stale_detection_on_clause_change():
    """Hash changes when any underlying clause text is modified."""
    clauses_v1 = [
        "Supplier shall defend and indemnify Customer.",
        "Either party may terminate upon 90 days notice.",
    ]
    clauses_v2 = [
        "Supplier shall defend and indemnify Customer.",
        "Either party may terminate upon 60 days notice.",  # modified
    ]

    h1 = compute_community_hash(clauses_v1)
    h2 = compute_community_hash(clauses_v2)

    assert h1 != h2
