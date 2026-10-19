"""Persistent cross-document party registry (data/parties.json).

The registry stores canonical party records across ALL contracts in the
portfolio.  The key design goals are:

1. IDEMPOTENCY: running resolution on the same document twice leaves the
   registry unchanged.  We track (doc_id, version) per party so re-runs
   are detected and skipped.

2. CANONICALIZATION: the 'canonical_name' is the first-seen alias; later
   aliases are added to the aliases list.  Party IDs are stable
   strings like 'party-0001'.

3. CROSS-DOCUMENT MERGING: when a new document mentions a name that fuzzy-
   matches an existing registry entry (via should_merge), they share the
   same party_id.

The file is a JSON array of RegistryEntry objects.  We read the whole file
on every call (it stays small—typically < 1000 entries for a portfolio).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..llm.client import LLMClient, NullLLM
from ..models import ParsedDocument
from .parties import extract_parties, should_merge
from .schemas import AppearanceRecord, PartyMention, RegistryEntry

_PARTIES_FILE = "parties.json"


def _load(registry_path: Path) -> list[RegistryEntry]:
    """Load existing registry from disk, or return empty list."""
    if registry_path.exists():
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        return [RegistryEntry.model_validate(r) for r in raw]
    return []


def _save(registry_path: Path, entries: list[RegistryEntry]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps([e.model_dump() for e in entries], indent=2),
        encoding="utf-8",
    )


def _next_party_id(entries: list[RegistryEntry]) -> str:
    """Generate the next sequential party ID, e.g. 'party-0001'."""
    if not entries:
        return "party-0001"
    # Find the highest existing numeric suffix
    nums = []
    for e in entries:
        m = e.party_id.split("-")
        if len(m) == 2 and m[1].isdigit():
            nums.append(int(m[1]))
    return f"party-{max(nums) + 1:04d}"


def _find_match(
    raw_name: str,
    entries: list[RegistryEntry],
    llm: LLMClient,
) -> tuple[Optional[RegistryEntry], bool, Optional[str]]:
    """Find the best matching registry entry for a raw party name.

    Returns (entry, needs_review, reason).
      entry is None if no match found or not merged.
    """
    best_entry: Optional[RegistryEntry] = None
    best_needs_review = False
    best_reason: Optional[str] = None

    for entry in entries:
        # Check against the canonical name and all aliases
        for alias in [entry.canonical_name] + entry.aliases:
            merge, nr, reason = should_merge(raw_name, alias, llm)
            if merge:
                return entry, False, None
            if nr:
                # Borderline: remember this as a candidate review but keep looking
                # for a definitive match
                best_entry = None  # do NOT merge yet
                best_needs_review = True
                best_reason = reason

    return best_entry, best_needs_review, best_reason


def resolve_parties(
    doc: ParsedDocument,
    data_dir: Path,
    llm: Optional[LLMClient] = None,
) -> list[PartyMention]:
    """Extract parties from doc, resolve against registry, update registry.

    Idempotency: if (doc_id, version) already appears in an entry's
    appearances, we skip adding a duplicate appearance record.

    Returns the list of PartyMention objects with party_ids populated.
    """
    if llm is None:
        llm = NullLLM()

    registry_path = data_dir / _PARTIES_FILE
    entries = _load(registry_path)

    # Extract raw mentions from the preamble
    mentions = extract_parties(doc)

    for mention in mentions:
        # Check whether this exact (doc_id, version, raw_name) was already
        # processed → idempotency guard
        already_registered = False
        for entry in entries:
            if mention.raw_name in ([entry.canonical_name] + entry.aliases):
                app = AppearanceRecord(
                    doc_id=doc.doc_id,
                    version=doc.version,
                    role_label=mention.role_label,
                )
                # Only add appearance if not already present
                exists = any(
                    a.doc_id == doc.doc_id and a.version == doc.version
                    and a.role_label == mention.role_label
                    for a in entry.appearances
                )
                if not exists:
                    entry.appearances.append(app)
                mention.party_id = entry.party_id
                already_registered = True
                break

        if already_registered:
            continue

        # Try fuzzy matching against existing entries
        matched_entry, needs_review, reason = _find_match(
            mention.raw_name, entries, llm
        )

        if matched_entry is not None:
            # Merged with an existing party
            if mention.raw_name not in matched_entry.aliases:
                matched_entry.aliases.append(mention.raw_name)
            app = AppearanceRecord(
                doc_id=doc.doc_id,
                version=doc.version,
                role_label=mention.role_label,
            )
            exists = any(
                a.doc_id == doc.doc_id and a.version == doc.version
                and a.role_label == mention.role_label
                for a in matched_entry.appearances
            )
            if not exists:
                matched_entry.appearances.append(app)
            mention.party_id = matched_entry.party_id
        else:
            # New canonical party
            new_id = _next_party_id(entries)
            new_entry = RegistryEntry(
                party_id=new_id,
                canonical_name=mention.raw_name,
                aliases=[],
                appearances=[AppearanceRecord(
                    doc_id=doc.doc_id,
                    version=doc.version,
                    role_label=mention.role_label,
                )],
            )
            entries.append(new_entry)
            mention.party_id = new_id
            mention.needs_review = needs_review
            mention.review_reason = reason

    _save(registry_path, entries)
    return mentions
