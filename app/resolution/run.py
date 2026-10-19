"""Layer 3 CLI.

    python -m app.resolution.run \\
        data/parsed/msa-acme.v1.json \\
        data/extracted/msa-acme.v1.json

Options:
    --provider rules   no LLM (default if LLM_PROVIDER not set)
    --provider ollama  use Ollama for borderline name merges
    --data-dir         root data directory (default: data)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..extraction.schemas import ExtractedDocument
from ..llm.client import get_client
from ..models import ParsedDocument
from ..playbook.loader import load_playbook
from .pipeline import resolve


def _fmt_review(match) -> str:
    """Format review flags for display."""
    if not match.needs_review:
        return ""
    short = "; ".join(r[:60] for r in match.review_reasons[:2])
    return f"  [REVIEW: {short}]"


def main():
    ap = argparse.ArgumentParser(description="Layer 3: entity resolution + template matching")
    ap.add_argument("parsed_json",  help="data/parsed/<doc>.v<n>.json")
    ap.add_argument("extracted_json", help="data/extracted/<doc>.v<n>.json")
    ap.add_argument("--provider", default="rules",
                    help="ollama | rules (default: rules; no LLM needed)")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()

    parsed = ParsedDocument.model_validate_json(Path(a.parsed_json).read_text())
    extracted = ExtractedDocument.model_validate_json(Path(a.extracted_json).read_text())
    data_dir = Path(a.data_dir)
    llm = get_client(a.provider)
    playbook = load_playbook()

    result = resolve(parsed, extracted, data_dir, llm=llm, playbook=playbook)

    # ---- pretty print ----
    print(f"\n{'='*60}")
    print(f"  {parsed.doc_id}  v{parsed.version}  ({parsed.doc_type})")
    print(f"{'='*60}")

    print("\nPARTIES")
    for p in result.parties:
        role = f" [{p.role_label}]" if p.role_label else ""
        review = " [REVIEW]" if p.needs_review else ""
        print(f"  {p.party_id}  {p.raw_name!r}{role}{review}")
        if p.needs_review and p.review_reason:
            print(f"       reason: {p.review_reason}")

    rm = result.role_mapping
    print("\nROLE MAPPING")
    print(f"  Customer -> {rm.customer_party_id or '(none)'}")
    print(f"  Supplier -> {rm.supplier_party_id or '(none)'}")

    print("\nTEMPLATE MATCHES")
    header = f"  {'section':<8} {'clause_type':<26} {'template_id':<16} {'score':<6} flags"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in result.template_matches:
        tpl = m.template_id or "(none)"
        review_str = " [REVIEW]" if m.needs_review else ""
        print(f"  {m.section_path:<8} {m.clause_type:<26} {tpl:<16} {m.score:<6.2f}{review_str}")
        if m.needs_review and m.review_reasons:
            for r in m.review_reasons[:2]:
                print(f"           ! {r[:70]}")

    n_rev = sum(m.needs_review for m in result.template_matches)
    print(f"\n{len(result.template_matches)} clauses matched, {n_rev} need review")
    print(f"Saved -> {data_dir}/resolved/{parsed.doc_id}.v{parsed.version}.json\n")


if __name__ == "__main__":
    main()
