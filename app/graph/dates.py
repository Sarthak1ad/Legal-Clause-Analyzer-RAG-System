"""Pure deterministic date derivation engine for contracts (Layer 4 Part B).

Derives key contract lifecycle dates from Layer 2 attributes and effective dates:
- initial_term_end: start + term_years - 1 day, or explicit end_date
- current_term_end: first renewal end >= as_of (if auto_renew)
- non_renewal_deadline: current_term_end - non_renewal_notice_days
- next_renewal: current_term_end + 1 day
- survival_end: initial_term_end + survival_years

Rules & Conventions:
1. start_date overrides effective_date when explicitly present.
2. initial_term_end = start + term_years - 1 day.
3. Every derived date records a human-readable derivation formula and input span_ids.
4. If a date cannot be derived, date is set to None with an explanation; NEVER guess.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional, Union

from .model import DateEventNode


def parse_iso_date(d: Union[str, date, None]) -> Optional[date]:
    """Parse a date object or ISO string into a datetime.date."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return datetime.strptime(str(d).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def add_years(d: date, years: int) -> date:
    """Add a whole number of years to a date, handling leap years safely."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        # Leap day Feb 29 -> Feb 28
        return d.replace(year=d.year + years, day=28)


def derive_dates_for_contract(
    doc_id: str,
    effective_date_str: Optional[str],
    attributes: dict[str, Any],
    attribute_spans: Optional[dict[str, str]] = None,
    as_of: Union[str, date] = "2026-10-19",
    doc_type: Optional[str] = None,
) -> list[DateEventNode]:
    """Derive all date events for a contract based on effective attributes.

    Args:
        doc_id: document identifier (e.g. 'msa-acme')
        effective_date_str: document effective date string (YYYY-MM-DD)
        attributes: dictionary of flattened attributes from effective clauses
        attribute_spans: map of attribute_name -> span_id providing evidence
        as_of: evaluation date for renewal status
        doc_type: document type (e.g. 'AMENDMENT', 'MSA', 'SOW')

    Returns:
        list of DateEventNode objects
    """
    spans = attribute_spans or {}
    as_of_date = parse_iso_date(as_of) or date.today()
    events: list[DateEventNode] = []

    # 1. Effective Date
    eff_date = parse_iso_date(effective_date_str)
    if eff_date:
        events.append(
            DateEventNode(
                id=f"{doc_id}:date:effective",
                kind="effective",
                date=eff_date.isoformat(),
                derivation=f"document effective date {eff_date.isoformat()}",
                input_span_ids=[spans["effective_date"]] if "effective_date" in spans else [],
                passed=(eff_date <= as_of_date),
            )
        )

    # Amendments only get an effective DateEvent (no initial_term_end / renewals)
    if doc_type == "AMENDMENT":
        return events

    # 2. Determine base start date
    raw_start = attributes.get("start_date")
    start_date = parse_iso_date(raw_start) if raw_start else eff_date
    start_span = [spans["start_date"]] if "start_date" in spans else []

    # 3. Initial Term End
    raw_end = attributes.get("end_date")
    explicit_end = parse_iso_date(raw_end) if raw_end else None
    term_years = attributes.get("term_years")

    initial_term_end: Optional[date] = None
    initial_end_spans: list[str] = []
    initial_end_derivation: str = ""

    if explicit_end:
        initial_term_end = explicit_end
        initial_end_spans = [spans["end_date"]] if "end_date" in spans else []
        initial_end_derivation = f"explicit end_date attribute {explicit_end.isoformat()}"
    elif start_date and term_years is not None:
        try:
            years_int = int(float(term_years))
            target_anniversary = add_years(start_date, years_int)
            initial_term_end = target_anniversary - timedelta(days=1)
            initial_end_spans = ([spans["term_years"]] if "term_years" in spans else []) + start_span
            initial_end_derivation = (
                f"start ({start_date.isoformat()}) + {years_int} years - 1 day = "
                f"{initial_term_end.isoformat()}"
            )
        except (ValueError, TypeError) as e:
            initial_end_derivation = f"invalid term_years value {term_years!r}: {e}"
    elif not start_date and term_years:
        initial_end_derivation = "term_years present but document has no start or effective date"
    else:
        initial_end_derivation = "no term_years or end_date specified (evergreen or indeterminate)"

    events.append(
        DateEventNode(
            id=f"{doc_id}:date:initial_term_end",
            kind="initial_term_end",
            date=initial_term_end.isoformat() if initial_term_end else None,
            derivation=initial_end_derivation,
            input_span_ids=initial_end_spans,
            passed=(initial_term_end <= as_of_date) if initial_term_end else None,
        )
    )

    # 4. Auto-renewal & Current Term End
    auto_renew = attributes.get("auto_renew", False)
    renewal_years = attributes.get("renewal_term_years", 1)
    notice_days = attributes.get("non_renewal_notice_days")

    if initial_term_end:
        if auto_renew:
            try:
                renew_int = int(float(renewal_years))
            except (ValueError, TypeError):
                renew_int = 1

            curr_end = initial_term_end
            renewals_count = 0
            while curr_end < as_of_date:
                curr_end = add_years(curr_end + timedelta(days=1), renew_int) - timedelta(days=1)
                renewals_count += 1

            current_term_end = curr_end
            curr_spans = (
                initial_end_spans
                + ([spans["auto_renew"]] if "auto_renew" in spans else [])
                + ([spans["renewal_term_years"]] if "renewal_term_years" in spans else [])
            )
            curr_deriv = (
                f"initial_term_end {initial_term_end.isoformat()} with {renewals_count} "
                f"renewal(s) of {renew_int} year(s) to encompass as_of {as_of_date.isoformat()}"
            )

            events.append(
                DateEventNode(
                    id=f"{doc_id}:date:current_term_end",
                    kind="current_term_end",
                    date=current_term_end.isoformat(),
                    derivation=curr_deriv,
                    input_span_ids=curr_spans,
                    passed=(current_term_end <= as_of_date),
                )
            )

            # Next renewal date: current_term_end + 1 day
            next_renew = current_term_end + timedelta(days=1)
            events.append(
                DateEventNode(
                    id=f"{doc_id}:date:next_renewal",
                    kind="next_renewal",
                    date=next_renew.isoformat(),
                    derivation=f"current_term_end ({current_term_end.isoformat()}) + 1 day",
                    input_span_ids=curr_spans,
                    passed=(next_renew <= as_of_date),
                )
            )

            # Non-renewal deadline: current_term_end - non_renewal_notice_days
            if notice_days is not None:
                try:
                    n_days = int(float(notice_days))
                    deadline = current_term_end - timedelta(days=n_days)
                    deadline_spans = curr_spans + (
                        [spans["non_renewal_notice_days"]]
                        if "non_renewal_notice_days" in spans
                        else []
                    )
                    events.append(
                        DateEventNode(
                            id=f"{doc_id}:date:non_renewal_deadline",
                            kind="non_renewal_deadline",
                            date=deadline.isoformat(),
                            derivation=(
                                f"current_term_end ({current_term_end.isoformat()}) - "
                                f"{n_days} days notice = {deadline.isoformat()}"
                            ),
                            input_span_ids=deadline_spans,
                            passed=(as_of_date > deadline),
                        )
                    )
                except (ValueError, TypeError) as e:
                    events.append(
                        DateEventNode(
                            id=f"{doc_id}:date:non_renewal_deadline",
                            kind="non_renewal_deadline",
                            date=None,
                            derivation=f"invalid non_renewal_notice_days {notice_days!r}: {e}",
                            input_span_ids=curr_spans,
                            passed=None,
                        )
                    )
        else:
            # No auto-renew -> current term end is initial term end
            events.append(
                DateEventNode(
                    id=f"{doc_id}:date:current_term_end",
                    kind="current_term_end",
                    date=initial_term_end.isoformat(),
                    derivation="no auto_renew; current_term_end equals initial_term_end",
                    input_span_ids=initial_end_spans,
                    passed=(initial_term_end <= as_of_date),
                )
            )

    # 5. Survival End (NDA / confidentiality)
    survival_years = attributes.get("survival_years")
    if survival_years is not None:
        if initial_term_end:
            try:
                surv_int = int(float(survival_years))
                surv_end = add_years(initial_term_end, surv_int)
                surv_spans = initial_end_spans + (
                    [spans["survival_years"]] if "survival_years" in spans else []
                )
                events.append(
                    DateEventNode(
                        id=f"{doc_id}:date:survival_end",
                        kind="survival_end",
                        date=surv_end.isoformat(),
                        derivation=(
                            f"initial_term_end ({initial_term_end.isoformat()}) + "
                            f"{surv_int} survival years = {surv_end.isoformat()}"
                        ),
                        input_span_ids=surv_spans,
                        passed=(surv_end <= as_of_date),
                    )
                )
            except (ValueError, TypeError) as e:
                events.append(
                    DateEventNode(
                        id=f"{doc_id}:date:survival_end",
                        kind="survival_end",
                        date=None,
                        derivation=f"invalid survival_years value {survival_years!r}: {e}",
                        input_span_ids=[],
                        passed=None,
                    )
                )
        else:
            events.append(
                DateEventNode(
                    id=f"{doc_id}:date:survival_end",
                    kind="survival_end",
                    date=None,
                    derivation="survival_years specified but initial_term_end could not be determined",
                    input_span_ids=[spans["survival_years"]] if "survival_years" in spans else [],
                    passed=None,
                )
            )

    return events
