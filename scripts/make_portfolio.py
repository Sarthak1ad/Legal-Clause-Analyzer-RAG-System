"""Generate demo portfolio of ~12 contracts for Layer 4 & Layer 5 testing.

Usage:
    python scripts/make_portfolio.py

Generates (in ./samples/portfolio):
  - 9 MSAs (DOCX)
  - 2 SOWs (DOCX)
  - 1 NDA (DOCX)
  - ground_truth.json (true attributes per contract)
  - expected_queries.json (expected answers for portfolio benchmark queries)

Portfolio properties:
  - Customer: Acme Corp (with 3 name variants: Acme Corp, ACME Corporation,
    Acme Corp., Inc. that must merge into one party, plus Acme Europe Ltd
    which must remain separate).
  - Suppliers: Hooli Systems Inc, Umbrella Data Ltd, Stark Cloud Corp,
    Wayne Analytics LLC, Wonka Services Ltd.
  - Indemnity caps: 0.5x, 1x, 1.5x, 2x, 3x.
  - Expiry spread:
      * 4 ending between 2026-11-01 and 2026-12-31
      * 1 expired on 2026-06-30
      * 4 ending in 2027
      * 2 ending in 2028
  - ~30% contracts use paraphrased wording.
"""
from __future__ import annotations

import json
import random
import textwrap
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from docx import Document

OUT = Path(__file__).resolve().parent.parent / "samples" / "portfolio"


def _fmt_date(d: date) -> str:
    """Format date as 'Month D, YYYY' without leading zero on day (Windows-compatible)."""
    return d.strftime("%B") + f" {d.day}, " + d.strftime("%Y")


COUNTERPARTIES = [
    "Hooli Systems Inc",
    "Umbrella Data Ltd",
    "Stark Cloud Corp",
    "Wayne Analytics LLC",
    "Wonka Services Ltd",
]

ACME_VARIANTS = [
    "Acme Corp",
    "ACME Corporation",
    "Acme Corp., Inc.",
]
ACME_EUROPE = "Acme Europe Ltd"

CAP_WORDINGS = {
    "0.5x": "0.5 times (0.5x) the annual",
    "1x": "one times (1x) the annual",
    "1.5x": "1.5 times (1.5x) the annual",
    "2x": "two times (2x) the annual",
    "3x": "three times (3x) the annual",
}

CAP_VALUES = {
    "0.5x": 0.5,
    "1x": 1.0,
    "1.5x": 1.5,
    "2x": 2.0,
    "3x": 3.0,
}

GOV_LAWS = ["New York", "California", "Delaware", "Texas", "England and Wales"]

_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    15: "fifteen", 30: "thirty", 45: "forty-five", 60: "sixty",
    90: "ninety", 120: "one hundred twenty",
}


def _n_word(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


def _msa_text(
    customer: str,
    supplier: str,
    eff_date: date,
    term_years: int,
    auto_renew: bool,
    notice_days: int,
    payment_days: int,
    cap_key: str,
    gov_law: str,
    gross_neg_exclusion: bool,
    mutual_indemnity: bool,
    survival_years: int,
    paraphrased: bool = False,
) -> str:
    cap_phrase = CAP_WORDINGS[cap_key]
    n_w = _n_word(term_years)
    s = "s" if term_years != 1 else ""
    notice_w = _n_word(notice_days)
    pay_w = _n_word(payment_days)
    surv_w = _n_word(survival_years)

    if paraphrased:
        term_clause = (
            f"This Agreement commences on the Effective Date and has an initial term of "
            f"{n_w} ({term_years}) year{s}."
        )
        renew_text = (
            f"This Agreement automatically extends for additional one-year periods unless "
            f"either party gives {notice_w} ({notice_days}) days' prior written notice of non-renewal."
            if auto_renew else "This Agreement does not extend automatically."
        )
        pay_clause = (
            f"Client shall remit Fees within {pay_w} ({payment_days}) days following "
            f"receipt of a correct invoice."
        )
        surv_clause = (
            f"This obligation survives for {surv_w} ({survival_years}) years following "
            f"the expiration or termination of this Agreement."
        )
        indem_clause = (
            f"Provider shall hold harmless Client from third-party claims arising from "
            f"Provider's gross negligence or willful misconduct."
        )
        if mutual_indemnity:
            indem_clause += " Client shall reciprocally hold harmless Provider from such claims."
        gross_neg_text = (
            "" if gross_neg_exclusion
            else " This exclusion does not apply to claims arising from gross negligence."
        )
        lol_clause = (
            f"Neither party is liable for indirect, incidental, special or consequential damages.{gross_neg_text}"
        )
    else:
        term_clause = (
            f"This Agreement begins on the Effective Date and continues for {n_w} "
            f"({term_years}) year{s} unless terminated earlier in accordance with this Agreement."
        )
        renew_text = (
            f"This Agreement renews automatically for successive one (1) year periods "
            f"unless either party gives {notice_w} ({notice_days}) days written notice of non-renewal."
            if auto_renew else "This Agreement does not renew automatically."
        )
        pay_clause = (
            f"Customer shall pay the Fees within {pay_w} ({payment_days}) days "
            f"of receipt of an undisputed invoice."
        )
        surv_clause = (
            f"This obligation survives for {surv_w} ({survival_years}) years after termination."
        )
        indem_clause = (
            f"Supplier shall defend and indemnify Customer against third-party claims arising from "
            f"Supplier's gross negligence or willful misconduct."
        )
        if mutual_indemnity:
            indem_clause += (
                " Customer shall similarly defend and indemnify Supplier against third-party claims "
                "arising from Customer's gross negligence or willful misconduct."
            )
        gross_neg_text = (
            "" if gross_neg_exclusion
            else " This exclusion does not apply to claims arising from gross negligence."
        )
        lol_clause = f"Neither party is liable for indirect or consequential damages.{gross_neg_text}"

    return textwrap.dedent(f"""\
        MASTER SERVICES AGREEMENT
        This Master Services Agreement is effective as of {_fmt_date(eff_date)} (the "Effective Date") between {customer} ("Customer") and {supplier} ("Supplier").
        1. Definitions
        "Fees" means the annual fees payable by Customer under each Statement of Work.
        "Confidential Information" means non-public information disclosed by one party to the other.
        2. Term and Renewal
        2.1 Term. {term_clause}
        2.2 Renewal. {renew_text}
        3. Fees and Payment
        3.1 Fees. {pay_clause}
        3.2 Late Payment. Overdue amounts accrue interest at one percent (1%) per month.
        4. Confidentiality
        Each party shall keep the other party's Confidential Information confidential and shall not disclose it to any third party without prior written consent, except as required by law. {surv_clause}
        5. Intellectual Property
        5.1 Ownership. Customer owns all deliverables created specifically for Customer.
        5.2 Supplier Materials. Supplier retains ownership of its pre-existing materials and tools.
        9. Indemnification
        9.1 Indemnity. {indem_clause}
        9.2 Cap on Indemnity. The indemnifying party's total liability under this Section 9 shall not exceed {cap_phrase} Fees paid in the twelve (12) months preceding the claim.
        10. Limitation of Liability
        10.1 Exclusion. {lol_clause}
        10.2 Cap. Each party's aggregate liability shall not exceed the annual Fees.
        12. General
        12.1 Notices. Notices must be in writing and delivered to the addresses set out in the applicable Statement of Work.
        12.2 Governing Law. This Agreement is governed by the laws of {gov_law}.
        12.3 Order of Precedence. In case of conflict, this Agreement prevails over any Statement of Work.
    """)


def _sow_text(
    customer: str,
    supplier: str,
    sow_eff: date,
    msa_eff: date,
    end_date: date,
    fee_usd: int,
    gov_law: str,
    paraphrased: bool = False,
) -> str:
    if paraphrased:
        svc = "Supplier shall render the consulting and data architecture services in Schedule A."
    else:
        svc = "Supplier shall provide the software development services described in Schedule A."

    return textwrap.dedent(f"""\
        STATEMENT OF WORK NO. 1
        This Statement of Work is effective as of {_fmt_date(sow_eff)} and is issued under the Master Services Agreement dated {_fmt_date(msa_eff)} between {customer} and {supplier}.
        1. Services
        {svc}
        2. Fees
        2.1 Fixed Fee. Customer shall pay a fixed annual fee of USD {fee_usd:,}.
        3. Term
        This Statement of Work begins on {_fmt_date(sow_eff)} and ends on {_fmt_date(end_date)}.
        4. General
        4.1 Governing Law. This Statement of Work is governed by the laws of {gov_law}.
    """)


def _nda_text(
    party_a: str,
    party_b: str,
    eff_date: date,
    term_years: int,
    survival_years: int,
) -> str:
    n_w = _n_word(term_years)
    surv_w = _n_word(survival_years)
    s = "s" if term_years != 1 else ""
    return textwrap.dedent(f"""\
        MUTUAL NON-DISCLOSURE AGREEMENT
        This Mutual Non-Disclosure Agreement is entered into as of {_fmt_date(eff_date)} between {party_a} and {party_b}.
        1. Confidential Information
        "Confidential Information" means all technical or business information disclosed by either party.
        2. Non-Disclosure Obligations
        Each party agrees to protect the Confidential Information of the other party with the same degree of care it uses for its own confidential information.
        3. Term
        3.1 Term. This Agreement begins on the Effective Date and continues for {n_w} ({term_years}) year{s}. Obligations survive for {surv_w} ({survival_years}) years after termination.
        4. General
        4.1 Governing Law. This Agreement is governed by the laws of the State of New York.
    """)


def write_docx(path: Path, text: str) -> None:
    d = Document()
    for line in text.strip().splitlines():
        d.add_paragraph(line)
    d.save(path)


def make_portfolio(out_dir: Path = OUT) -> None:
    """Generate 12 contracts and ground truth files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truth: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Contract definitions
    # Expiry spread targets:
    #   4 ending 2026-11-01 to 2026-12-31:
    #     - msa-hooli-acme (eff 2024-11-02, term 2 -> 2026-11-01)
    #     - msa-umbrella-acme (eff 2024-12-01, term 2 -> 2026-11-30)
    #     - msa-stark-acme (eff 2025-01-01, term 2 -> 2026-12-31)
    #     - sow-hooli-acme-01 (ends 2026-12-31)
    #   1 expired on 2026-06-30:
    #     - msa-wayne-acme (eff 2024-07-01, term 2 -> 2026-06-30)
    #   4 in 2027:
    #     - msa-wonka-acme (eff 2025-03-01, term 2 -> 2027-02-28)
    #     - msa-hooli-acme-eu (eff 2025-04-01, term 2 -> 2027-03-31)
    #     - msa-umbrella-acme-eu (eff 2025-06-01, term 2 -> 2027-05-31)
    #     - sow-wonka-acme-01 (ends 2027-04-30)
    #   2 in 2028:
    #     - msa-stark-acme-eu (eff 2026-01-15, term 2 -> 2028-01-14)
    #     - msa-wayne-acme-eu (eff 2026-03-01, term 2 -> 2028-02-28)
    # ------------------------------------------------------------------

    contracts = [
        # 1. msa-hooli-acme: 2026-11-01, cap 1x, notice 90d (passed as of 2026-10-19)
        dict(
            doc_id="msa-hooli-acme",
            doc_type="MSA",
            customer=ACME_VARIANTS[0],   # "Acme Corp"
            supplier=COUNTERPARTIES[0],  # Hooli Systems Inc
            eff_date=date(2024, 11, 2),
            term_years=2,
            auto_renew=True,
            notice_days=90,
            payment_days=30,
            cap_key="1x",
            gov_law="New York",
            gross_neg_exclusion=True,
            mutual_indemnity=False,
            survival_years=5,
            paraphrased=False,
        ),
        # 2. msa-umbrella-acme: 2026-11-30, cap 0.5x, non-renewing, California
        dict(
            doc_id="msa-umbrella-acme",
            doc_type="MSA",
            customer=ACME_VARIANTS[1],   # "ACME Corporation"
            supplier=COUNTERPARTIES[1],  # Umbrella Data Ltd
            eff_date=date(2024, 12, 1),
            term_years=2,
            auto_renew=False,
            notice_days=60,
            payment_days=45,
            cap_key="0.5x",
            gov_law="California",
            gross_neg_exclusion=False,
            mutual_indemnity=True,
            survival_years=3,
            paraphrased=False,
        ),
        # 3. msa-stark-acme: 2026-12-31, cap 1.5x, notice 30d (pending as of 2026-10-19), Delaware
        dict(
            doc_id="msa-stark-acme",
            doc_type="MSA",
            customer=ACME_VARIANTS[2],   # "Acme Corp., Inc."
            supplier=COUNTERPARTIES[2],  # Stark Cloud Corp
            eff_date=date(2025, 1, 1),
            term_years=2,
            auto_renew=True,
            notice_days=30,
            payment_days=30,
            cap_key="1.5x",
            gov_law="Delaware",
            gross_neg_exclusion=True,
            mutual_indemnity=False,
            survival_years=5,
            paraphrased=False,
        ),
        # 4. msa-wayne-acme: 2026-06-30 (expired), cap 2x, Texas
        dict(
            doc_id="msa-wayne-acme",
            doc_type="MSA",
            customer=ACME_VARIANTS[0],   # "Acme Corp"
            supplier=COUNTERPARTIES[3],  # Wayne Analytics LLC
            eff_date=date(2024, 7, 1),
            term_years=2,
            auto_renew=False,
            notice_days=60,
            payment_days=30,
            cap_key="2x",
            gov_law="Texas",
            gross_neg_exclusion=False,
            mutual_indemnity=False,
            survival_years=3,
            paraphrased=False,
        ),
        # 5. msa-wonka-acme: 2027-02-28, cap 3x, England, Paraphrased
        dict(
            doc_id="msa-wonka-acme",
            doc_type="MSA",
            customer=ACME_VARIANTS[0],   # "Acme Corp"
            supplier=COUNTERPARTIES[4],  # Wonka Services Ltd
            eff_date=date(2025, 3, 1),
            term_years=2,
            auto_renew=True,
            notice_days=90,
            payment_days=30,
            cap_key="3x",
            gov_law="England and Wales",
            gross_neg_exclusion=True,
            mutual_indemnity=True,
            survival_years=5,
            paraphrased=True,
        ),
        # 6. msa-hooli-acme-eu: 2027-03-31, cap 1x, England (Acme Europe)
        dict(
            doc_id="msa-hooli-acme-eu",
            doc_type="MSA",
            customer=ACME_EUROPE,
            supplier=COUNTERPARTIES[0],
            eff_date=date(2025, 4, 1),
            term_years=2,
            auto_renew=True,
            notice_days=90,
            payment_days=30,
            cap_key="1x",
            gov_law="England and Wales",
            gross_neg_exclusion=True,
            mutual_indemnity=False,
            survival_years=5,
            paraphrased=False,
        ),
        # 7. msa-umbrella-acme-eu: 2027-05-31, cap 2x, England, Paraphrased
        dict(
            doc_id="msa-umbrella-acme-eu",
            doc_type="MSA",
            customer=ACME_EUROPE,
            supplier=COUNTERPARTIES[1],
            eff_date=date(2025, 6, 1),
            term_years=2,
            auto_renew=False,
            notice_days=60,
            payment_days=45,
            cap_key="2x",
            gov_law="England and Wales",
            gross_neg_exclusion=False,
            mutual_indemnity=False,
            survival_years=3,
            paraphrased=True,
        ),
        # 8. msa-stark-acme-eu: 2028-01-14, cap 1.5x, Delaware
        dict(
            doc_id="msa-stark-acme-eu",
            doc_type="MSA",
            customer=ACME_EUROPE,
            supplier=COUNTERPARTIES[2],
            eff_date=date(2026, 1, 15),
            term_years=2,
            auto_renew=True,
            notice_days=90,
            payment_days=30,
            cap_key="1.5x",
            gov_law="Delaware",
            gross_neg_exclusion=True,
            mutual_indemnity=True,
            survival_years=5,
            paraphrased=False,
        ),
        # 9. msa-wayne-acme-eu: 2028-02-28, cap 0.5x, New York, Paraphrased
        dict(
            doc_id="msa-wayne-acme-eu",
            doc_type="MSA",
            customer=ACME_EUROPE,
            supplier=COUNTERPARTIES[3],
            eff_date=date(2026, 3, 1),
            term_years=2,
            auto_renew=False,
            notice_days=60,
            payment_days=30,
            cap_key="0.5x",
            gov_law="New York",
            gross_neg_exclusion=False,
            mutual_indemnity=False,
            survival_years=3,
            paraphrased=True,
        ),
    ]

    for c in contracts:
        text = _msa_text(
            customer=c["customer"],
            supplier=c["supplier"],
            eff_date=c["eff_date"],
            term_years=c["term_years"],
            auto_renew=c["auto_renew"],
            notice_days=c["notice_days"],
            payment_days=c["payment_days"],
            cap_key=c["cap_key"],
            gov_law=c["gov_law"],
            gross_neg_exclusion=c["gross_neg_exclusion"],
            mutual_indemnity=c["mutual_indemnity"],
            survival_years=c["survival_years"],
            paraphrased=c["paraphrased"],
        )
        filename = c["doc_id"].replace("-", "_") + ".docx"
        write_docx(out_dir / filename, text)

        eff = c["eff_date"]
        try:
            initial_term_end = eff.replace(year=eff.year + c["term_years"]) - timedelta(days=1)
        except ValueError:
            initial_term_end = eff.replace(year=eff.year + c["term_years"], day=28) - timedelta(days=1)

        gt: dict[str, Any] = {
            "doc_id": c["doc_id"],
            "doc_type": c["doc_type"],
            "filename": filename,
            "customer_name": c["customer"],
            "supplier_name": c["supplier"],
            "effective_date": eff.isoformat(),
            "term_years": c["term_years"],
            "auto_renew": c["auto_renew"],
            "initial_term_end": initial_term_end.isoformat(),
            "expiry_as_of_2026_10_19": initial_term_end.isoformat(),
            "cap_key": c["cap_key"],
            "cap_multiple": CAP_VALUES[c["cap_key"]],
            "governing_law": c["gov_law"],
            "mutual_indemnity": c["mutual_indemnity"],
            "gross_neg_exclusion": c["gross_neg_exclusion"],
            "survival_years": c["survival_years"],
            "payment_days": c["payment_days"],
            "is_paraphrased": c["paraphrased"],
            "standard_vs_paraphrased": "paraphrased" if c["paraphrased"] else "standard",
        }
        ground_truth.append(gt)

    # ------------------------------------------------------------------
    # SOWs (2 SOWs)
    # ------------------------------------------------------------------
    sow_configs = [
        # SOW 1: under msa-hooli-acme, ends 2026-12-31 (Q4 2026)
        dict(
            doc_id="sow-hooli-acme-01",
            doc_type="SOW",
            parent_doc_id="msa-hooli-acme",
            customer=ACME_VARIANTS[0],
            supplier=COUNTERPARTIES[0],
            msa_eff=date(2024, 11, 2),
            sow_eff=date(2025, 1, 1),
            end_date=date(2026, 12, 31),
            fee_usd=180_000,
            gov_law="New York",
            paraphrased=False,
        ),
        # SOW 2: under msa-wonka-acme, ends 2027-04-30 (2027)
        dict(
            doc_id="sow-wonka-acme-01",
            doc_type="SOW",
            parent_doc_id="msa-wonka-acme",
            customer=ACME_VARIANTS[0],
            supplier=COUNTERPARTIES[4],
            msa_eff=date(2025, 3, 1),
            sow_eff=date(2025, 5, 1),
            end_date=date(2027, 4, 30),
            fee_usd=300_000,
            gov_law="England and Wales",
            paraphrased=True,
        ),
    ]

    for s in sow_configs:
        text = _sow_text(
            customer=s["customer"],
            supplier=s["supplier"],
            sow_eff=s["sow_eff"],
            msa_eff=s["msa_eff"],
            end_date=s["end_date"],
            fee_usd=s["fee_usd"],
            gov_law=s["gov_law"],
            paraphrased=s["paraphrased"],
        )
        filename = s["doc_id"].replace("-", "_") + ".docx"
        write_docx(out_dir / filename, text)
        ground_truth.append({
            "doc_id": s["doc_id"],
            "doc_type": s["doc_type"],
            "filename": filename,
            "customer_name": s["customer"],
            "supplier_name": s["supplier"],
            "parent_doc_id": s["parent_doc_id"],
            "effective_date": s["sow_eff"].isoformat(),
            "end_date": s["end_date"].isoformat(),
            "initial_term_end": s["end_date"].isoformat(),
            "expiry_as_of_2026_10_19": s["end_date"].isoformat(),
            "cap_multiple": None,
            "auto_renew": False,
            "governing_law": s["gov_law"],
            "is_paraphrased": s["paraphrased"],
            "standard_vs_paraphrased": "paraphrased" if s["paraphrased"] else "standard",
        })

    # ------------------------------------------------------------------
    # NDA
    # ------------------------------------------------------------------
    nda_eff = date(2025, 9, 1)
    nda_end = date(2027, 8, 31)
    nda_text = _nda_text(
        party_a=ACME_VARIANTS[0],
        party_b=COUNTERPARTIES[2],  # Stark Cloud
        eff_date=nda_eff,
        term_years=2,
        survival_years=3,
    )
    nda_file = "nda_stark_acme.docx"
    write_docx(out_dir / nda_file, nda_text)
    ground_truth.append({
        "doc_id": "nda-stark-acme",
        "doc_type": "NDA",
        "filename": nda_file,
        "customer_name": ACME_VARIANTS[0],
        "supplier_name": COUNTERPARTIES[2],
        "effective_date": nda_eff.isoformat(),
        "term_years": 2,
        "auto_renew": False,
        "initial_term_end": nda_end.isoformat(),
        "expiry_as_of_2026_10_19": nda_end.isoformat(),
        "cap_multiple": None,
        "governing_law": "New York",
        "survival_years": 3,
        "is_paraphrased": False,
        "standard_vs_paraphrased": "standard",
    })

    # Write ground_truth.json
    gt_path = out_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # expected_queries.json
    # ------------------------------------------------------------------
    expected_queries = {
        "party_merge": {
            "description": "The 3 Acme Corp name variants must resolve to one canonical party.",
            "acme_variants": ACME_VARIANTS,
            "acme_europe": ACME_EUROPE,
            "rule": "All ACME_VARIANTS map to the same party_id; ACME_EUROPE gets a different party_id.",
        },
        "query_1_cap_below_standard_and_expiring_2026": {
            "description": "contracts whose indemnity cap is below the standard (2x) AND whose current term ends in 2026",
            "expected_doc_ids": [
                "msa-hooli-acme",     # cap 1x, ends 2026-11-01
                "msa-umbrella-acme",  # cap 0.5x, ends 2026-11-30
                "msa-stark-acme",     # cap 1.5x, ends 2026-12-31
            ],
        },
        "query_2_governing_law_not_new_york": {
            "description": "contracts with governing law other than New York",
            "expected_doc_ids": [
                "msa-umbrella-acme",     # California
                "msa-stark-acme",        # Delaware
                "msa-wayne-acme",        # Texas
                "msa-wonka-acme",        # England and Wales
                "msa-hooli-acme-eu",     # England and Wales
                "msa-umbrella-acme-eu",  # England and Wales
                "msa-stark-acme-eu",     # Delaware
                "sow-wonka-acme-01",     # England and Wales
            ],
        },
        "query_3_auto_renew_and_non_renewal_deadline_passed": {
            "description": "contracts that auto-renew and whose non-renewal deadline has already passed as of 2026-10-19",
            "expected_doc_ids": [
                "msa-hooli-acme",  # notice deadline 2026-08-03 passed
            ],
        },
        "expiring_2026_q4": {
            "description": "Contracts with initial_term_end between 2026-11-01 and 2026-12-31.",
            "expected_doc_ids": [
                "msa-hooli-acme",     # 2026-11-01
                "msa-umbrella-acme",  # 2026-11-30
                "msa-stark-acme",     # 2026-12-31
                "sow-hooli-acme-01",  # 2026-12-31
            ],
        },
        "expired_before_today": {
            "description": "Contracts whose initial_term_end <= 2026-10-19 (as_of date).",
            "expected_doc_ids": ["msa-wayne-acme"],  # 2026-06-30
        },
        "expiring_2027": {
            "description": "Contracts with initial_term_end in 2027.",
            "expected_doc_ids": [
                "msa-wonka-acme",        # 2027-02-28
                "msa-hooli-acme-eu",     # 2027-03-31
                "msa-umbrella-acme-eu",  # 2027-05-31
                "sow-wonka-acme-01",     # 2027-04-30
            ],
        },
        "expiring_2028": {
            "description": "Contracts with initial_term_end in 2028.",
            "expected_doc_ids": [
                "msa-stark-acme-eu",  # 2028-01-14
                "msa-wayne-acme-eu",  # 2028-02-28
            ],
        },
        "cap_variants": {
            "description": "Each cap_key should appear in at least one contract.",
            "expected_caps": ["0.5x", "1x", "1.5x", "2x", "3x"],
        },
        "auto_renew_contracts": {
            "description": "Contracts with auto_renew=True.",
            "expected_doc_ids": [
                "msa-hooli-acme",
                "msa-stark-acme",
                "msa-wonka-acme",
                "msa-hooli-acme-eu",
                "msa-stark-acme-eu",
            ],
        },
    }
    eq_path = out_dir / "expected_queries.json"
    eq_path.write_text(json.dumps(expected_queries, indent=2), encoding="utf-8")

    print(f"Portfolio written to {out_dir}")
    print(f"  {len(ground_truth)} contracts")
    print(f"  ground_truth.json: {gt_path}")
    print(f"  expected_queries.json: {eq_path}")


if __name__ == "__main__":
    make_portfolio()
