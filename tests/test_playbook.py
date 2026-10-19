"""Tests for the playbook loader (Part B).

Tests:
  8a. Loader accepts the shipped templates.
  8b. Loader rejects an unknown attribute name with a clear error.
  8c. Loader rejects an invalid clause_type with a clear error.
  8d. Loader rejects two approved templates for the same clause_type.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.playbook.loader import load_playbook  # noqa: E402
from app.playbook.schemas import KNOWN_ATTRIBUTES  # noqa: E402

# Path to the shipped playbook directory (not a temp directory).
PLAYBOOK_DIR = ROOT / "playbook"


# ---------- test 8a: shipped templates load without error ----------
def test_playbook_loader_accepts_shipped_templates():
    """The shipped YAML templates must all validate cleanly."""
    pb = load_playbook(PLAYBOOK_DIR)
    # Config loaded
    assert pb.config.our_role == "customer"
    assert pb.config.playbook_version

    # We shipped 7 templates (indem, lol, term, pay, conf, gov, oop)
    assert len(pb.templates) >= 6

    # All approved templates have unique clause_types
    approved = [t for t in pb.templates if t.status == "approved"]
    types_seen = {t.clause_type for t in approved}
    assert len(approved) == len(types_seen), "Duplicate approved templates found"

    # History loaded
    assert len(pb.history) >= 8


# ---------- test 8b: unknown attribute name rejected ----------
def test_playbook_loader_rejects_unknown_attribute(tmp_path):
    """A template with an unrecognised attribute name must raise ValueError."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()

    bad_tpl = {
        "template_id": "TPL-TEST-v1",
        "clause_type": "indemnification",
        "version": "1",
        "status": "approved",
        "effective_from": "2024-01-01",
        "standard_text": "Test text.",
        "attributes": [
            {
                "name": "this_attribute_does_not_exist",  # invalid!
                "value_kind": "number",
                "preferred": 2.0,
                "acceptable": 1.5,
                "fallback": 1.0,
                "better_when": "higher",
                "favours_role": "customer",
            }
        ],
    }
    (tpl_dir / "TPL-TEST-v1.yaml").write_text(yaml.dump(bad_tpl), encoding="utf-8")

    # Write a minimal config so the loader can initialise
    (tmp_path / "config.yaml").write_text(
        yaml.dump({
            "our_organization": "Test", "our_role": "customer",
            "default_perspective": "customer", "playbook_version": "test",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a known Layer 2 attribute"):
        load_playbook(tmp_path)


# ---------- test 8c: invalid clause_type rejected ----------
def test_playbook_loader_rejects_invalid_clause_type(tmp_path):
    """A template whose clause_type is not in the ClauseType enum must fail."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()

    bad_tpl = {
        "template_id": "TPL-BAD-v1",
        "clause_type": "totally_made_up_type",  # invalid!
        "version": "1",
        "status": "approved",
        "effective_from": "2024-01-01",
        "standard_text": "Text.",
        "attributes": [],
    }
    (tpl_dir / "TPL-BAD-v1.yaml").write_text(yaml.dump(bad_tpl), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        yaml.dump({
            "our_organization": "Test", "our_role": "customer",
            "default_perspective": "customer", "playbook_version": "test",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a valid ClauseType"):
        load_playbook(tmp_path)


# ---------- test 8d: duplicate approved templates rejected ----------
def test_playbook_loader_rejects_two_approved_for_same_type(tmp_path):
    """Two 'approved' templates for the same clause_type must raise ValueError."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()

    for suffix in ("a", "b"):
        dup = {
            "template_id": f"TPL-INDEM-{suffix}",
            "clause_type": "indemnification",
            "version": "1",
            "status": "approved",
            "effective_from": "2024-01-01",
            "standard_text": "Text.",
            "attributes": [],
        }
        (tpl_dir / f"TPL-INDEM-{suffix}.yaml").write_text(yaml.dump(dup), encoding="utf-8")

    (tmp_path / "config.yaml").write_text(
        yaml.dump({
            "our_organization": "Test", "our_role": "customer",
            "default_perspective": "customer", "playbook_version": "test",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Two approved templates"):
        load_playbook(tmp_path)
