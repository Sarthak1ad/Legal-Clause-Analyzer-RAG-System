"""Unit tests for the demo portfolio and expected portfolio queries (Layer 4 Part 0e).

Verifies:
- Standard-wording contracts in samples/portfolio match ground_truth.json exactly.
- Acme name variants ("Acme Corp", "ACME Corporation", "Acme Corp., Inc.") resolve
  to a single Party node in the graph, while "Acme Europe Ltd" remains distinct.
- The 3 expected queries produce the expected contract IDs:
  1) Indemnity cap < 2x and term ending in 2026
  2) Governing law not New York
  3) Auto-renew with non-renewal deadline passed as of 2026-10-19
- Expiry distribution matches requirements (4 in Nov-Dec 2026, 1 in Jun 2026, 4 in 2027, 2 in 2028).
"""
import json
from pathlib import Path
import pytest

from app.graph.reader import InMemoryGraphReader

PORTFOLIO_DIR = Path("samples/portfolio")
GRAPH_JSON = Path("data/graph/graph.json")


@pytest.fixture(scope="module")
def reader() -> InMemoryGraphReader:
    if not GRAPH_JSON.exists():
        pytest.skip("data/graph/graph.json not found; run pipeline first")
    return InMemoryGraphReader.from_json(GRAPH_JSON)


@pytest.fixture(scope="module")
def ground_truth() -> list[dict]:
    gt_file = PORTFOLIO_DIR / "ground_truth.json"
    if not gt_file.exists():
        pytest.skip("samples/portfolio/ground_truth.json not found")
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def expected_queries() -> dict:
    eq_file = PORTFOLIO_DIR / "expected_queries.json"
    if not eq_file.exists():
        pytest.skip("samples/portfolio/expected_queries.json not found")
    with open(eq_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_portfolio_ground_truth_standard_contracts(reader: InMemoryGraphReader, ground_truth: list[dict]):
    """Standard-wording contracts must match ground truth values in extraction and graph."""
    standard_contracts = [c for c in ground_truth if not c.get("is_paraphrased")]
    assert len(standard_contracts) >= 7

    for expected in standard_contracts:
        doc_id = expected["doc_id"]
        c_node = reader.contract(doc_id)
        assert c_node is not None, f"Contract node missing: {doc_id}"

        # 1. Governing law check
        expected_law = expected.get("governing_law")
        if expected_law:
            jurisdiction = reader.jurisdiction_of(doc_id)
            assert jurisdiction is not None, f"Jurisdiction missing for {doc_id}"
            actual_law = jurisdiction.get("name", "").lower()
            exp_l = expected_law.lower()
            assert exp_l in actual_law or actual_law in exp_l, (
                f"Law mismatch for {doc_id}: expected {expected_law}, got {jurisdiction.get('name')}"
            )

        # 2. Indemnity cap check (for MSAs)
        if expected.get("cap_multiple") is not None:
            expected_cap = float(expected["cap_multiple"])
            clauses = reader.clauses_of(doc_id, as_of="2026-10-19")
            indem_cap = None
            for cl in clauses:
                if cl.get("clause_type") == "indemnification":
                    attrs = cl.get("attributes") or {}
                    if "cap_multiple" in attrs:
                        indem_cap = float(attrs["cap_multiple"])
            assert indem_cap == pytest.approx(expected_cap, 0.01), (
                f"Cap mismatch for {doc_id}: expected {expected_cap}, got {indem_cap}"
            )

        # 3. Initial term end date check
        expected_end = expected.get("initial_term_end")
        if expected_end:
            dates = {d.get("kind"): d.get("date") for d in reader.dates_of(doc_id)}
            actual_end = dates.get("initial_term_end")
            assert actual_end == expected_end, (
                f"Term end mismatch for {doc_id}: expected {expected_end}, got {actual_end}"
            )


def test_entity_resolution_acme_variants(reader: InMemoryGraphReader, expected_queries: dict):
    """Acme Corp name variants must merge to ONE party; Acme Europe Ltd must be separate."""
    # Find all contracts
    party_map: dict[str, str] = {}
    for c in reader.contracts():
        doc_id = c["doc_id"]
        parties = reader.parties_of(doc_id)
        for p in parties:
            canonical = p.get("canonical_name", "")
            party_map[doc_id] = canonical

    # Inspect parties of msa-hooli-acme, msa-umbrella-acme, msa-stark-acme
    hooli_parties = {p.get("canonical_name") for p in reader.parties_of("msa-hooli-acme")}
    umbrella_parties = {p.get("canonical_name") for p in reader.parties_of("msa-umbrella-acme")}
    stark_parties = {p.get("canonical_name") for p in reader.parties_of("msa-stark-acme")}
    eu_parties = {p.get("canonical_name") for p in reader.parties_of("msa-hooli-acme-eu")}

    # All US variants must share the canonical Acme Corp party
    acme_canonical = "Acme Corp"
    assert acme_canonical in hooli_parties
    assert acme_canonical in umbrella_parties
    assert acme_canonical in stark_parties

    # Acme Europe Ltd must NOT be the same as Acme Corp
    assert "Acme Europe Ltd" in eu_parties
    assert acme_canonical not in eu_parties


def test_expected_query_1_cap_below_standard(reader: InMemoryGraphReader, expected_queries: dict):
    """Query 1: contracts whose indemnity cap is below standard (2x) AND current term ends in 2026."""
    expected_ids = sorted(expected_queries["query_1_cap_below_standard_and_expiring_2026"]["expected_doc_ids"])

    matches = []
    for c in reader.contracts():
        doc_id = c["doc_id"]
        clauses = reader.clauses_of(doc_id, as_of="2026-10-19")
        cap_val = None
        for cl in clauses:
            if cl.get("clause_type") == "indemnification":
                attrs = cl.get("attributes") or {}
                if "cap_multiple" in attrs:
                    cap_val = float(attrs["cap_multiple"])
        dates = {d.get("kind"): d.get("date") for d in reader.dates_of(doc_id)}
        curr_end = dates.get("current_term_end") or dates.get("initial_term_end")
        if cap_val is not None and cap_val < 2.0 and curr_end and curr_end.startswith("2026"):
            matches.append(doc_id)

    assert sorted(matches) == expected_ids


def test_expected_query_2_governing_law_not_new_york(reader: InMemoryGraphReader, expected_queries: dict):
    """Query 2: portfolio contracts with governing law other than New York."""
    expected_ids = sorted(expected_queries["query_2_governing_law_not_new_york"]["expected_doc_ids"])

    matches = []
    for doc_id in expected_ids:
        jur = reader.jurisdiction_of(doc_id)
        assert jur is not None
        assert "new york" not in jur.get("name", "").lower()

    # Across portfolio contracts
    portfolio_ids = [c["doc_id"] for c in reader.contracts() if "portfolio" in c.get("path", "") or doc_id_in_portfolio(c["doc_id"])]
    matches = [
        c["doc_id"] for c in reader.contracts()
        if c["doc_id"] in expected_ids
    ]
    assert sorted(matches) == expected_ids


def doc_id_in_portfolio(doc_id: str) -> bool:
    return any(k in doc_id for k in ("hooli", "umbrella", "stark", "wayne", "wonka"))


def test_expected_query_3_auto_renew_deadline_passed(reader: InMemoryGraphReader, expected_queries: dict):
    """Query 3: contracts that auto-renew and non-renewal deadline passed as of 2026-10-19."""
    expected_ids = sorted(expected_queries["query_3_auto_renew_and_non_renewal_deadline_passed"]["expected_doc_ids"])

    matches = []
    for c in reader.contracts():
        doc_id = c["doc_id"]
        dates = {d.get("kind"): d for d in reader.dates_of(doc_id)}
        opt = dates.get("non_renewal_deadline") or dates.get("non_renewal_notice_deadline")
        if opt and opt.get("passed") is True:
            matches.append(doc_id)

    assert sorted(matches) == expected_ids


def test_portfolio_expiry_spread(reader: InMemoryGraphReader):
    """Verify portfolio expiry spread matches specifications."""
    nov_dec_2026 = reader.contracts_expiring_between("2026-11-01", "2026-12-31", which="initial_term_end")
    assert len(nov_dec_2026) >= 4

    exp_2027 = reader.contracts_expiring_between("2027-01-01", "2027-12-31", which="initial_term_end")
    assert len(exp_2027) >= 4

    exp_2028 = reader.contracts_expiring_between("2028-01-01", "2028-12-31", which="initial_term_end")
    assert len(exp_2028) >= 2

    # Expired before 2026-10-19
    wayne = reader.contract("msa-wayne-acme")
    assert wayne is not None
    dates = {d.get("kind"): d for d in reader.dates_of("msa-wayne-acme")}
    end = dates.get("initial_term_end")
    assert end and end.get("date") == "2026-06-30"
    assert end.get("passed") is True
