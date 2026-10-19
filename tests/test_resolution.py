"""Tests for Layer 3: entity resolution + playbook + template matching.

Tests 1-11 from the spec, plus test 12 (existing tests still pass, handled by
just running the full test suite).

The fixture approach mirrors test_ingestion.py and test_extraction.py:
we create temporary DOCX files in memory from the same make_samples constants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_samples  # noqa: E402
from app.extraction.extractor import extract_document  # noqa: E402
from app.extraction.schemas import ClauseType, ExtractedClause  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402
from app.llm.client import FakeLLM, LLMError, NullLLM  # noqa: E402
from app.models import ParsedDocument  # noqa: E402
from app.playbook.loader import load_playbook  # noqa: E402
from app.resolution.parties import (  # noqa: E402
    extract_parties, normalize_name, should_merge,
)
from app.resolution.pipeline import resolve  # noqa: E402
from app.resolution.registry import resolve_parties  # noqa: E402

PLAYBOOK_DIR = ROOT / "playbook"


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    """Write all sample DOCX/PDF files to a temp directory."""
    out = tmp_path_factory.mktemp("samples")
    make_samples.OUT = out
    make_samples.write_docx("msa_acme.docx", make_samples.MSA_ACME)
    make_samples.write_docx("sow_acme_01.docx", make_samples.SOW_ACME)
    make_samples.write_docx("nda_globex.docx", make_samples.NDA_GLOBEX)
    make_samples.write_pdf("msa_initech.pdf", make_samples.MSA_INITECH)
    return out


@pytest.fixture(scope="session")
def docs(sample_dir, tmp_path_factory):
    """Ingest all sample documents."""
    dd = tmp_path_factory.mktemp("data_ingested")
    return {
        "msa": ingest(sample_dir / "msa_acme.docx", dd),
        "sow": ingest(sample_dir / "sow_acme_01.docx", dd),
        "nda": ingest(sample_dir / "nda_globex.docx", dd),
        "initech": ingest(sample_dir / "msa_initech.pdf", dd),
    }


@pytest.fixture(scope="session")
def ex(docs):
    """Rules-only extraction of all sample documents."""
    return {k: extract_document(d, NullLLM()) for k, d in docs.items()}


@pytest.fixture(scope="session")
def playbook():
    return load_playbook(PLAYBOOK_DIR)


# ======================================================================
# Test 1: Party extraction from MSA (with role labels)
# ======================================================================

def test_msa_party_extraction_with_roles(docs):
    """Acme Corp = Customer, Northwind Software Ltd = Supplier; evidence round-trips."""
    parties = extract_parties(docs["msa"])
    assert len(parties) == 2

    by_role = {p.role_label: p for p in parties}
    assert "Customer" in by_role
    assert "Supplier" in by_role
    assert by_role["Customer"].raw_name == "Acme Corp"
    assert by_role["Supplier"].raw_name == "Northwind Software Ltd"

    # Evidence span round-trip: doc.full_text[span.start:span.end] == quote
    doc = docs["msa"]
    for p in parties:
        ev = p.evidence
        assert doc.full_text[ev.span.start:ev.span.end] == ev.quote, (
            f"Span mismatch for {p.raw_name!r}"
        )


# ======================================================================
# Test 2: NDA without role labels → both parties found, roles = None
# ======================================================================

def test_nda_party_extraction_without_role_labels(docs):
    """NDA preamble has no role labels; both parties are found with role_label=None."""
    parties = extract_parties(docs["nda"])
    assert len(parties) == 2
    for p in parties:
        assert p.role_label is None
    names = {p.raw_name for p in parties}
    assert "Globex Inc" in names
    assert "Initech LLC" in names

    # Evidence round-trip
    doc = docs["nda"]
    for p in parties:
        ev = p.evidence
        assert doc.full_text[ev.span.start:ev.span.end] == ev.quote


# ======================================================================
# Test 3: Same company across docs → exactly ONE party_id
# ======================================================================

def test_same_company_across_docs(docs, tmp_path):
    """MSA and SOW both mention 'Acme Corp' → exactly one party_id in registry."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    msa_parties = resolve_parties(docs["msa"], data_dir, NullLLM())
    sow_parties = resolve_parties(docs["sow"], data_dir, NullLLM())

    # Find Acme Corp in both docs
    msa_acme = next(p for p in msa_parties if "Acme" in p.raw_name)
    sow_acme = next(p for p in sow_parties if "Acme" in p.raw_name)

    # They must share the same party_id
    assert msa_acme.party_id == sow_acme.party_id

    # Verify registry has exactly one entry for Acme
    registry_path = data_dir / "parties.json"
    registry = json.loads(registry_path.read_text())
    acme_entries = [e for e in registry if "Acme" in e["canonical_name"]
                    or any("Acme" in a for a in e["aliases"])]
    assert len(acme_entries) == 1, (
        f"Expected 1 Acme entry in registry, got {len(acme_entries)}"
    )


# ======================================================================
# Test 4: Name variants merge / "Acme Europe" does NOT merge
# ======================================================================

def test_name_variants_merge(tmp_path):
    """Acme Corp, ACME Corporation, Acme Corp., Inc. → one party; Acme Europe → separate."""
    # Verify normalization
    assert normalize_name("Acme Corp") == normalize_name("ACME Corporation")
    assert normalize_name("Acme Corp., Inc.") == normalize_name("Acme Corp")

    # Auto-merge threshold: score >= 92
    merge, nr, _ = should_merge("Acme Corp", "ACME Corporation", NullLLM())
    assert merge is True and nr is False

    merge, nr, _ = should_merge("Acme Corp., Inc.", "Acme Corp", NullLLM())
    assert merge is True and nr is False

    # "Acme Europe" has a distinguishing token ("europe") vs "acme" alone
    # With NullLLM (borderline 80-91 range), this should NOT merge
    merge_eu, _, _ = should_merge("Acme Europe Ltd", "Acme Corp", NullLLM())
    assert merge_eu is False, (
        "Acme Europe Ltd should NOT auto-merge with Acme Corp"
    )


# ======================================================================
# Test 5: Borderline score + LLM unavailable → not merged, needs_review=True
# ======================================================================

def test_borderline_llm_unavailable_sets_review():
    """When score is 80-91 and LLM is unavailable, result: no merge, needs_review=True."""
    # "BlueStar Technologies Inc" vs "Bluestar Technology Inc" → score ~90 (borderline range)
    # NullLLM always raises LLMError → should set needs_review
    merge, needs_review, reason = should_merge(
        "BlueStar Technologies Inc", "Bluestar Technology Inc", NullLLM()
    )
    assert merge is False
    assert needs_review is True
    assert reason is not None and "LLM unavailable" in reason


# ======================================================================
# Test 6: Borderline + FakeLLM says same_company=True → merged
# ======================================================================

def test_borderline_llm_says_same_merges():
    """When LLM says same_company=true for a borderline pair, they merge."""
    fake = FakeLLM(lambda *_: {"same_company": True})
    merge, needs_review, reason = should_merge(
        "BlueStar Technologies Inc", "Bluestar Technology Inc", fake
    )
    assert merge is True
    assert needs_review is False
    assert len(fake.calls) == 1  # LLM was consulted exactly once


# ======================================================================
# Test 7: Idempotency – running resolution twice leaves parties.json unchanged
# ======================================================================

def test_idempotency(docs, tmp_path):
    """Running resolve_parties twice on the same doc produces identical registry."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    resolve_parties(docs["msa"], data_dir, NullLLM())
    content_after_first = (data_dir / "parties.json").read_text()

    resolve_parties(docs["msa"], data_dir, NullLLM())
    content_after_second = (data_dir / "parties.json").read_text()

    assert json.loads(content_after_first) == json.loads(content_after_second), (
        "Registry changed on second run — idempotency violated"
    )


# ======================================================================
# Test 8: Playbook loader — see tests/test_playbook.py
# ======================================================================
# (Tests 8a-8d are in test_playbook.py as required by the spec)


# ======================================================================
# Test 9: Template matching — 9.1 and 9.2 → TPL-INDEM; 12.2 → TPL-GOV; 12.1 → null
# ======================================================================

def test_template_matching_msa(ex, playbook, docs, tmp_path):
    """MSA clauses: 9.1 and 9.2 → indemnification template; 12.2 → governing_law; 12.1 → null."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = resolve(docs["msa"], ex["msa"], data_dir, NullLLM(), playbook)

    by_path = {m.section_path: m for m in result.template_matches}

    # Both indemnification clauses → same template
    assert by_path["9.1"].template_id == by_path["9.2"].template_id
    assert by_path["9.1"].template_id is not None
    assert "INDEM" in by_path["9.1"].template_id

    # Governing law clause → governing law template
    assert by_path["12.2"].template_id is not None
    assert "GOV" in by_path["12.2"].template_id

    # Notices clause → no template
    assert by_path["12.1"].template_id is None
    assert "no standard template" in " ".join(by_path["12.1"].review_reasons)


# ======================================================================
# Test 10: Review flags from Layer 2 are carried into the match
# ======================================================================

def test_review_flags_carried_from_layer2(docs, playbook, tmp_path):
    """Clauses with needs_review=True in Layer 2 keep that flag in the match."""
    from app.extraction.extractor import extract_clause
    from app.resolution.matcher import match_clause

    # Get a clause and force a review flag by using a fake LLM that disagrees
    c = next(x for x in docs["msa"].clauses if x.section_path == "9.2")

    # FakeLLM that disagrees with the rule → sets needs_review=True in extraction
    fake = FakeLLM(lambda *_: {"clause_type": "payment"})
    extracted = extract_clause(c, fake)
    assert extracted.needs_review, "Expected needs_review from LLM disagreement"

    match = match_clause(extracted, playbook)
    assert match.needs_review
    assert any("LLM says" in r for r in match.review_reasons)


# ======================================================================
# Test 11: Every Evidence span produced in Layer 3 round-trips
# ======================================================================

def test_layer3_evidence_spans_roundtrip(docs, ex, playbook, tmp_path):
    """All party mention Evidence spans must round-trip against full_text."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for key in ("msa", "sow", "nda", "initech"):
        result = resolve(docs[key], ex[key], data_dir, NullLLM(), playbook)
        doc = docs[key]
        for p in result.parties:
            ev = p.evidence
            actual = doc.full_text[ev.span.start:ev.span.end]
            assert actual == ev.quote, (
                f"[{key}] Party {p.raw_name!r} span mismatch: "
                f"{actual!r} != {ev.quote!r}"
            )


# ======================================================================
# Additional: Acme appears exactly once after running all 4 docs
# ======================================================================

def test_acme_appears_once_across_portfolio(docs, ex, playbook, tmp_path):
    """After resolving all 4 sample docs, 'Acme' has exactly one registry entry."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for key in ("msa", "sow", "nda", "initech"):
        resolve(docs[key], ex[key], data_dir, NullLLM(), playbook)

    registry = json.loads((data_dir / "parties.json").read_text())
    acme_entries = [
        e for e in registry
        if "acme" in e["canonical_name"].lower()
        or any("acme" in a.lower() for a in e["aliases"])
    ]
    assert len(acme_entries) == 1, (
        f"Expected 1 Acme entry, got {len(acme_entries)}: "
        f"{[e['party_id'] for e in acme_entries]}"
    )
    # The single Acme entry should have appearances in msa and sow (not nda or initech)
    entry = acme_entries[0]
    appearing_docs = {a["doc_id"] for a in entry["appearances"]}
    assert docs["msa"].doc_id in appearing_docs
    assert docs["sow"].doc_id in appearing_docs


# ======================================================================
# Additional: type_match score and method are correct
# ======================================================================

def test_type_match_score_and_method(ex, playbook, docs, tmp_path):
    """type_match clauses have method='type_match' and score=1.0."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = resolve(docs["msa"], ex["msa"], data_dir, NullLLM(), playbook)
    by_path = {m.section_path: m for m in result.template_matches}

    # Indemnification has a template → type_match
    assert by_path["9.2"].method == "type_match"
    assert by_path["9.2"].score == 1.0

    # Notices has no template → method='none', score=0.0
    assert by_path["12.1"].method == "none"
    assert by_path["12.1"].score == 0.0


# ======================================================================
# Additional: SOW preamble extracts parties without role labels
# ======================================================================

def test_sow_preamble_no_role_labels(docs):
    """SOW says 'between Acme Corp and Northwind Software Ltd' (no roles)."""
    parties = extract_parties(docs["sow"])
    assert len(parties) == 2
    for p in parties:
        assert p.role_label is None
    names = {p.raw_name for p in parties}
    assert "Acme Corp" in names or "Northwind Software Ltd" in names
