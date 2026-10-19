"""Tests for Layer 4 Part B: Pure deterministic derived dates."""
from __future__ import annotations

from datetime import date
from app.graph.dates import derive_dates_for_contract, parse_iso_date


def test_msa_acme_dates():
    """msa-acme: 3-year term from 2023-12-01, auto-renew 1 year, 90 days notice."""
    attrs = {
        "term_years": 3,
        "auto_renew": True,
        "renewal_term_years": 1,
        "non_renewal_notice_days": 90,
        "survival_years": 5,
    }

    # As of 2026-10-19 (before initial term ends 2026-11-30, notice deadline 2026-09-01 passed)
    events_oct = {
        e.kind: e
        for e in derive_dates_for_contract(
            "msa-acme", "2023-12-01", attrs, as_of="2026-10-19"
        )
    }

    assert events_oct["initial_term_end"].date == "2026-11-30"
    assert events_oct["current_term_end"].date == "2026-11-30"
    assert events_oct["next_renewal"].date == "2026-12-01"
    assert events_oct["non_renewal_deadline"].date == "2026-09-01"
    assert events_oct["non_renewal_deadline"].passed is True
    assert events_oct["survival_end"].date == "2031-11-30"

    # As of 2026-12-15 (after first renewal on 2026-12-01, current term ends 2027-11-30)
    events_dec = {
        e.kind: e
        for e in derive_dates_for_contract(
            "msa-acme", "2023-12-01", attrs, as_of="2026-12-15"
        )
    }

    assert events_dec["current_term_end"].date == "2027-11-30"
    assert events_dec["non_renewal_deadline"].date == "2027-09-01"
    assert events_dec["non_renewal_deadline"].passed is False
    assert events_dec["next_renewal"].date == "2027-12-01"


def test_sow_acme_dates():
    """sow-acme-01: explicit end_date 2027-01-14, no renewal."""
    attrs = {
        "start_date": "2024-01-15",
        "end_date": "2027-01-14",
    }
    events = {
        e.kind: e
        for e in derive_dates_for_contract(
            "sow-acme-01", "2024-01-15", attrs, as_of="2026-10-19"
        )
    }

    assert events["initial_term_end"].date == "2027-01-14"
    assert events["current_term_end"].date == "2027-01-14"
    assert "next_renewal" not in events
    assert "non_renewal_deadline" not in events


def test_nda_globex_dates():
    """nda-globex: 2-year term from 2025-03-03, 3-year survival."""
    attrs = {
        "term_years": 2,
        "survival_years": 3,
    }
    events = {
        e.kind: e
        for e in derive_dates_for_contract(
            "nda-globex", "2025-03-03", attrs, as_of="2026-10-19"
        )
    }

    assert events["initial_term_end"].date == "2027-03-02"
    assert events["survival_end"].date == "2030-03-02"


def test_msa_initech_dates():
    """msa-initech: 3-year term from 2024-06-01 -> ends 2027-05-31."""
    attrs = {
        "term_years": 3,
        "auto_renew": True,
        "renewal_term_years": 1,
        "non_renewal_notice_days": 90,
    }
    events = {
        e.kind: e
        for e in derive_dates_for_contract(
            "msa-initech", "2024-06-01", attrs, as_of="2026-10-19"
        )
    }

    assert events["initial_term_end"].date == "2027-05-31"


def test_cannot_derive_evergreen_yields_null_and_reason():
    """When a contract is evergreen without term_years or end_date, date is null with reason."""
    attrs = {"auto_renew": False}
    events = {
        e.kind: e
        for e in derive_dates_for_contract(
            "evergreen-doc", "2024-01-01", attrs, as_of="2026-10-19"
        )
    }

    ite = events["initial_term_end"]
    assert ite.date is None
    assert "no term_years or end_date" in ite.derivation
