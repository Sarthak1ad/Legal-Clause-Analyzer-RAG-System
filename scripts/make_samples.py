"""Generate realistic sample contracts for development and tests.

    python scripts/make_samples.py

Creates (in ./samples):
  msa_acme.docx   Master Services Agreement, decimal numbering, 1x-fees indemnity cap
  sow_acme_01.docx  SOW issued under that MSA (will conflict on governing law, tested in Layer 6)
  nda_globex.docx   NDA using ARTICLE headings
  msa_initech.pdf   Same kind of MSA as a digital PDF
"""
import textwrap
from pathlib import Path

import pymupdf
from docx import Document

OUT = Path(__file__).resolve().parent.parent / "samples"

MSA_ACME = """MASTER SERVICES AGREEMENT
This Master Services Agreement is effective as of December 1, 2023 (the "Effective Date") between Acme Corp ("Customer") and Northwind Software Ltd ("Supplier").
1. Definitions
"Fees" means the annual fees payable by Customer under each Statement of Work.
"Confidential Information" means non-public information disclosed by one party to the other.
2. Term and Renewal
2.1 Term. This Agreement begins on the Effective Date and continues for three (3) years unless terminated earlier in accordance with this Agreement.
2.2 Renewal. This Agreement renews automatically for successive one (1) year periods unless either party gives ninety (90) days written notice of non-renewal.
3. Fees and Payment
3.1 Fees. Customer shall pay the Fees within thirty (30) days of receipt of an undisputed invoice.
3.2 Late Payment. Overdue amounts accrue interest at one percent (1%) per month.
4. Confidentiality
Each party shall keep the other party's Confidential Information confidential and shall not disclose it to any third party without prior written consent, except as required by law. This obligation survives for five (5) years after termination.
5. Intellectual Property
5.1 Ownership. Customer owns all deliverables created specifically for Customer, subject to Section 12.3.
5.2 Supplier Materials. Supplier retains ownership of its pre-existing materials and tools.
9. Indemnification
9.1 Supplier Indemnity. Supplier shall defend and indemnify Customer against third-party claims arising from:
(a) infringement of any intellectual property right by the deliverables;
(b) Supplier's gross negligence or willful misconduct; or
(c) breach of Section 4 (Confidentiality).
9.2 Cap on Indemnity. Supplier's total liability under this Section 9 shall not exceed one times (1x) the annual Fees paid in the twelve (12) months preceding the claim.
10. Limitation of Liability
10.1 Exclusion. Neither party is liable for indirect or consequential damages.
10.2 Cap. Each party's aggregate liability shall not exceed the annual Fees, except for liability under Section 9.1(b).
12. General
12.1 Notices. Notices must be in writing and delivered to the addresses set out in the applicable Statement of Work.
12.2 Governing Law. This Agreement is governed by the laws of the State of New York.
12.3 Order of Precedence. In case of conflict, this Agreement prevails over any Statement of Work.
"""

SOW_ACME = """STATEMENT OF WORK NO. 1
This Statement of Work is effective as of January 15, 2024 and is issued under the Master Services Agreement dated December 1, 2023 between Acme Corp and Northwind Software Ltd.
1. Services
Supplier shall provide the data migration services described in Schedule A.
2. Fees
2.1 Fixed Fee. Customer shall pay a fixed annual fee of USD 240,000.
3. Term
This Statement of Work begins on January 15, 2024 and ends on January 14, 2027.
4. General
4.1 Governing Law. This Statement of Work is governed by the laws of the State of California.
"""

NDA_GLOBEX = """MUTUAL NON-DISCLOSURE AGREEMENT
This Mutual Non-Disclosure Agreement is dated 3 March 2025 between Globex Inc and Initech LLC.
ARTICLE I
DEFINITIONS
1.1 Confidential Information. Any non-public business, technical or financial information disclosed by either party.
ARTICLE II
OBLIGATIONS
2.1 Non-Use. The receiving party shall use Confidential Information solely to evaluate the proposed transaction.
2.2 Standard of Care. The receiving party shall protect Confidential Information with at least reasonable care and shall:
(a) limit access to personnel who need to know;
(b) not copy the information except as necessary; and
(c) promptly notify the disclosing party of any unauthorized disclosure.
ARTICLE III
TERM
3.1 Term. This Agreement expires two (2) years after the date above. Obligations survive for three (3) years thereafter.
"""

MSA_INITECH = MSA_ACME.replace("Acme Corp", "Initech LLC").replace(
    "December 1, 2023", "June 1, 2024").replace(
    "one times (1x) the annual Fees", "two times (2x) the annual Fees")


def write_docx(name: str, text: str):
    d = Document()
    for line in text.strip().splitlines():
        d.add_paragraph(line)
    d.save(OUT / name)


def write_pdf(name: str, text: str):
    pdf = pymupdf.open()
    page, y = pdf.new_page(), 60
    for line in text.strip().splitlines():
        for i, wrapped in enumerate(textwrap.wrap(line, 95) or [""]):
            if y > 780:
                page, y = pdf.new_page(), 60
            page.insert_text((50, y), wrapped, fontsize=10)
            y += 13
        y += 9  # paragraph gap
    pdf.save(OUT / name)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write_docx("msa_acme.docx", MSA_ACME)
    write_docx("sow_acme_01.docx", SOW_ACME)
    write_docx("nda_globex.docx", NDA_GLOBEX)
    write_pdf("msa_initech.pdf", MSA_INITECH)
    print("samples written to", OUT)
