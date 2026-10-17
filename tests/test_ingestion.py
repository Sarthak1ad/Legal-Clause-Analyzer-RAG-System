import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_samples  # noqa: E402
from app.ingestion.numbering import classify  # noqa: E402
from app.ingestion.parsers import normalize  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    make_samples.OUT = out
    make_samples.write_docx("msa_acme.docx", make_samples.MSA_ACME)
    make_samples.write_docx("sow_acme_01.docx", make_samples.SOW_ACME)
    make_samples.write_docx("nda_globex.docx", make_samples.NDA_GLOBEX)
    make_samples.write_pdf("msa_initech.pdf", make_samples.MSA_INITECH)
    return out


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


# ---------- numbering ----------
def test_numbering_recognition():
    assert classify("9.2 Cap on Indemnity. Supplier's total...", None).kind == "decimal"
    assert classify("9. Indemnification", None).title == "Indemnification"
    assert classify("30 days after receipt", None) is None          # body text, not a number
    assert classify("ARTICLE IV", None).kind == "article"
    assert classify("(a) infringement of IP", None).kind == "letter"
    assert classify("(i) first item", "h").kind == "letter"         # (i) after (h)
    assert classify("(i) first item", None).kind == "roman"
    assert classify("(ii) second item", "a").kind == "roman"


def test_normalize_strips_hidden_characters():
    assert normalize("Sup​plier shall") == "Supplier shall"


# ---------- the key invariant ----------
@pytest.mark.parametrize("name", ["msa_acme.docx", "sow_acme_01.docx",
                                  "nda_globex.docx", "msa_initech.pdf"])
def test_every_span_roundtrips(samples, data_dir, name):
    doc = ingest(samples / name, data_dir)
    assert doc.clauses, "no clauses found"
    for c in doc.clauses:
        assert doc.text_of(c.span) == c.text          # clause span == clause text
        for sub in c.sub_spans:
            assert doc.text_of(sub)                   # sub-spans are non-empty
            assert sub.start >= c.span.start and sub.end <= c.span.end
    assert not [w for w in doc.warnings if "mismatch" in w or "out of range" in w]


# ---------- structure and segmentation ----------
def test_msa_clause_segmentation(samples, data_dir):
    doc = ingest(samples / "msa_acme.docx", data_dir)
    paths = [c.section_path for c in doc.clauses]
    assert "9.1" in paths and "9.2" in paths and "10.2" in paths
    c92 = next(c for c in doc.clauses if c.section_path == "9.2")
    assert c92.heading == "Cap on Indemnity"
    assert c92.parent_heading == "Indemnification"
    assert "one times (1x)" in c92.text
    # 9.1 keeps its (a)(b)(c) list inside the clause
    c91 = next(c for c in doc.clauses if c.section_path == "9.1")
    assert [s.section_path for s in c91.sub_spans] == ["9.1(a)", "9.1(b)", "9.1(c)"]
    assert "breach of Section 4" in c91.text
    # a section with no numbered sub-sections is one clause
    c4 = next(c for c in doc.clauses if c.section_path == "4")
    assert c4.heading == "Confidentiality" and "five (5) years" in c4.text


def test_article_style_nda(samples, data_dir):
    doc = ingest(samples / "nda_globex.docx", data_dir)
    assert [s.path for s in doc.sections] == ["Art.I", "Art.II", "Art.III"]
    assert doc.sections[1].title == "Obligations"
    paths = [c.section_path for c in doc.clauses]
    assert paths == ["1.1", "2.1", "2.2", "3.1"]
    c22 = next(c for c in doc.clauses if c.section_path == "2.2")
    assert len(c22.sub_spans) == 3


def test_pdf_matches_docx_structure(samples, data_dir):
    pdf = ingest(samples / "msa_initech.pdf", data_dir)
    docx = ingest(samples / "msa_acme.docx", data_dir)
    assert [c.section_path for c in pdf.clauses] == [c.section_path for c in docx.clauses]
    assert pdf.num_pages >= 1
    c92 = next(c for c in pdf.clauses if c.section_path == "9.2")
    assert "two times (2x)" in c92.text


# ---------- metadata ----------
def test_metadata_msa(samples, data_dir):
    doc = ingest(samples / "msa_acme.docx", data_dir)
    assert doc.doc_type == "MSA"
    assert str(doc.effective_date) == "2023-12-01"
    assert doc.parent_ref is None


def test_metadata_sow_parent_link(samples, data_dir):
    doc = ingest(samples / "sow_acme_01.docx", data_dir)
    assert doc.doc_type == "SOW"
    assert str(doc.effective_date) == "2024-01-15"        # not the MSA's date
    assert doc.parent_ref.doc_type == "MSA"
    assert str(doc.parent_ref.date) == "2023-12-01"


def test_metadata_nda_date_format(samples, data_dir):
    doc = ingest(samples / "nda_globex.docx", data_dir)
    assert doc.doc_type == "NDA"
    assert str(doc.effective_date) == "2025-03-03"        # '3 March 2025'


# ---------- versioning ----------
def test_versioning(samples, data_dir, tmp_path):
    f = tmp_path / "contract.docx"
    shutil.copy(samples / "msa_acme.docx", f)
    v1 = ingest(f, data_dir)
    again = ingest(f, data_dir)
    assert v1.version == again.version == 1               # same text -> same version

    # change one clause -> new version; the old JSON stays untouched
    from docx import Document
    d = Document(str(f))
    for p in d.paragraphs:
        if "one times (1x)" in p.text:
            p.text = p.text.replace("one times (1x)", "two times (2x)")
    d.save(str(f))
    v2 = ingest(f, data_dir)
    assert v2.version == 2
    assert v2.sha256 != v1.sha256
    old = json.loads((data_dir / "parsed" / "contract.v1.json").read_text())
    assert "one times (1x)" in old["full_text"]
    assert v2.clauses[0].span.span_id.startswith("contract:v2:")
