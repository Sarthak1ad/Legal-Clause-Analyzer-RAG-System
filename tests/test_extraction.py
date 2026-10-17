import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_samples  # noqa: E402
from app.extraction.extractor import extract_clause, extract_document  # noqa: E402
from app.extraction.schemas import ClauseType  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402
from app.llm.client import FakeLLM, LLMError, NullLLM, OllamaClient  # noqa: E402


@pytest.fixture(scope="session")
def docs(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    make_samples.OUT = out
    make_samples.write_docx("msa_acme.docx", make_samples.MSA_ACME)
    make_samples.write_docx("sow_acme_01.docx", make_samples.SOW_ACME)
    make_samples.write_docx("nda_globex.docx", make_samples.NDA_GLOBEX)
    make_samples.write_pdf("msa_initech.pdf", make_samples.MSA_INITECH)
    dd = tmp_path_factory.mktemp("data")
    return {n: ingest(out / f, dd) for n, f in [
        ("msa", "msa_acme.docx"), ("sow", "sow_acme_01.docx"),
        ("nda", "nda_globex.docx"), ("initech", "msa_initech.pdf")]}


@pytest.fixture(scope="session")
def ex(docs):
    """Rules-only extraction (no LLM) of every sample."""
    return {k: extract_document(d, NullLLM()) for k, d in docs.items()}


def clause(e, path):
    return next(c for c in e.clauses if c.section_path == path)


def val(e, path, name):
    a = clause(e, path).attr(name)
    return a.value if a else None


# ---------- classification by rules ----------
def test_rule_types_msa(ex):
    t = {c.section_path: c.type_by_rule.value for c in ex["msa"].clauses}
    assert t["9.1"] == t["9.2"] == "indemnification"
    assert t["10.1"] == t["10.2"] == "limitation_of_liability"
    assert t["2.1"] == t["2.2"] == "term_renewal"
    assert t["3.1"] == t["3.2"] == "payment"
    assert t["4"] == "confidentiality"
    assert t["12.2"] == "governing_law"
    assert t["12.3"] == "order_of_precedence"
    assert t["5.1"] == t["5.2"] == "ip_ownership"
    assert t["1"] == "definitions"


# ---------- exact attribute values ----------
def test_indemnity_attributes(ex):
    assert val(ex["msa"], "9.1", "indemnitor") == "Supplier"
    assert val(ex["msa"], "9.1", "indemnitee") == "Customer"
    assert val(ex["msa"], "9.1", "direction") == "one_way"
    assert val(ex["msa"], "9.1", "covers_gross_negligence") is True
    assert len(val(ex["msa"], "9.1", "covered_events")) == 3
    assert val(ex["msa"], "9.2", "cap_multiple") == 1.0
    assert val(ex["msa"], "9.2", "cap_basis") == "annual fees"
    assert val(ex["msa"], "9.2", "cap_lookback_months") == 12


def test_headline_difference_between_acme_and_initech(ex):
    # the deviation the whole project is built to find: 1x vs 2x
    assert val(ex["msa"], "9.2", "cap_multiple") == 1.0
    assert val(ex["initech"], "9.2", "cap_multiple") == 2.0


def test_limitation_of_liability(ex):
    assert val(ex["msa"], "10.1", "excludes_indirect_damages") is True
    assert val(ex["msa"], "10.2", "cap_multiple") == 1.0     # 'not exceed the annual Fees'
    assert val(ex["msa"], "10.2", "cap_carve_outs") == ["9.1(b)"]
    assert val(ex["msa"], "10.2", "mutual") is True


def test_term_payment_confidentiality(ex):
    assert val(ex["msa"], "2.1", "term_years") == 3
    assert val(ex["msa"], "2.2", "auto_renew") is True
    assert val(ex["msa"], "2.2", "renewal_term_years") == 1
    assert val(ex["msa"], "2.2", "non_renewal_notice_days") == 90
    assert clause(ex["msa"], "2.2").attr("term_years") is None   # 'one (1) year' is not the term
    assert val(ex["msa"], "3.1", "payment_days") == 30
    assert val(ex["msa"], "3.2", "late_interest_pct_per_month") == 1.0
    assert val(ex["msa"], "4", "survival_years") == 5


def test_governing_law_conflict_inputs(ex):
    # MSA says New York, SOW says California: Layer 6 will flag this
    assert val(ex["msa"], "12.2", "governing_law") == "New York"
    assert val(ex["sow"], "4.1", "governing_law") == "California"
    assert val(ex["msa"], "12.3", "prevailing_document") == "agreement"
    assert val(ex["msa"], "12.3", "subordinate_document") == "statement of work"


def test_sow_and_nda(ex):
    assert val(ex["sow"], "2.1", "fee_amount") == 240000.0
    assert val(ex["sow"], "2.1", "fee_currency") == "USD"
    assert val(ex["sow"], "3", "start_date") == "2024-01-15"
    assert val(ex["sow"], "3", "end_date") == "2027-01-14"
    assert val(ex["nda"], "3.1", "term_years") == 2
    assert val(ex["nda"], "3.1", "survival_years") == 3


def test_references_and_defined_terms(ex):
    refs = {r.target for r in clause(ex["msa"], "5.1").references}
    assert refs == {"12.3"}
    assert {r.target for r in clause(ex["msa"], "10.2").references} == {"9.1(b)"}
    terms = {t.term for t in clause(ex["msa"], "1").defined_terms}
    assert terms == {"Fees", "Confidential Information"}


# ---------- the key invariant: every evidence span is real ----------
def test_every_evidence_span_roundtrips(docs, ex):
    n = 0
    for key, e in ex.items():
        doc = docs[key]
        for c in e.clauses:
            evs = [a.evidence for a in c.attributes if a.evidence]
            evs += [r.evidence for r in c.references] + [t.evidence for t in c.defined_terms]
            for ev in evs:
                assert doc.text_of(ev.span) == ev.quote, (key, c.section_path, ev.quote)
                n += 1
    assert n > 60


# ---------- ensemble: rules + LLM ----------
def fake_classifier(mapping):
    def handler(system, prompt, schema):
        if "clause_type" in schema["properties"]:
            for heading, ctype in mapping.items():
                if f"Heading: {heading}" in prompt:
                    return {"clause_type": ctype}
            return {"clause_type": "other"}
        raise AssertionError("unexpected call")
    return handler


def test_llm_agrees_gives_high_confidence(docs):
    c = next(x for x in docs["msa"].clauses if x.section_path == "9.2")
    r = extract_clause(c, FakeLLM(fake_classifier({"Cap on Indemnity": "indemnification"})))
    assert r.clause_type == ClauseType.indemnification and r.confidence == 0.95
    assert not r.needs_review


def test_llm_disagrees_keeps_heading_rule_and_flags_review(docs):
    c = next(x for x in docs["msa"].clauses if x.section_path == "9.2")
    r = extract_clause(c, FakeLLM(fake_classifier({"Cap on Indemnity": "payment"})))
    assert r.clause_type == ClauseType.indemnification      # rule wins on a strong heading
    assert r.type_by_llm == ClauseType.payment
    assert r.needs_review and "LLM says payment" in r.review_reasons[0]


def test_llm_failure_falls_back_to_rules_with_review(docs):
    def boom(*a):
        raise LLMError("down")
    c = next(x for x in docs["msa"].clauses if x.section_path == "9.2")
    r = extract_clause(c, FakeLLM(boom))
    assert r.clause_type == ClauseType.indemnification
    assert r.needs_review and r.confidence <= 0.6
    assert r.attr("cap_multiple").value == 1.0               # rules still extracted it


# ---------- grounded LLM fill ----------
@pytest.fixture
def odd_doc(tmp_path):
    p = tmp_path / "odd.txt"
    p.write_text("ODD AGREEMENT\n1. Governing Law\nThis Agreement shall be construed under English law.\n")
    return ingest(p, tmp_path / "d")


def fill_handler(value, quote):
    def h(system, prompt, schema):
        if "clause_type" in schema["properties"]:
            return {"clause_type": "governing_law"}
        return {"value": value, "quote": quote}
    return h


def test_llm_fill_accepts_grounded_answer(odd_doc):
    r = extract_clause(odd_doc.clauses[0],
                       FakeLLM(fill_handler("English", "construed under English law")))
    a = r.attr("governing_law")
    assert a and a.value == "English" and a.source == "llm"
    assert odd_doc.text_of(a.evidence.span) == "construed under English law"
    assert not r.needs_review


def test_llm_fill_rejects_fabricated_quote(odd_doc):
    r = extract_clause(odd_doc.clauses[0],
                       FakeLLM(fill_handler("Delaware", "governed by the laws of Delaware")))
    assert r.attr("governing_law") is None                    # quote is not in the clause
    assert r.needs_review and "governing_law" in r.review_reasons[0]


def test_llm_fill_rejects_quote_that_does_not_support_value(odd_doc):
    r = extract_clause(odd_doc.clauses[0],
                       FakeLLM(fill_handler("Delaware", "This Agreement shall be construed")))
    assert r.attr("governing_law") is None and r.needs_review


# ---------- Ollama client (against a local stub server) ----------
def test_ollama_client_sends_schema_and_zero_temperature():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"message": {"content": json.dumps({"clause_type": "payment"})}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = OllamaClient(model="qwen2.5:3b", host=f"http://127.0.0.1:{srv.server_port}")
        schema = {"type": "object", "properties": {"clause_type": {"type": "string"}}}
        out = client.generate_json("sys", "prompt", schema)
    finally:
        srv.shutdown()
    assert out == {"clause_type": "payment"}
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:3b"
    assert seen["body"]["format"] == schema
    assert seen["body"]["options"]["temperature"] == 0
    assert seen["body"]["stream"] is False


def test_ollama_unreachable_raises_llmerror():
    client = OllamaClient(host="http://127.0.0.1:9", timeout=1, retries=0)
    with pytest.raises(LLMError):
        client.generate_json("s", "p", {"type": "object"})
