"""Playbook loader: read YAML templates + JSON history, validate, and expose.

Design rule: FAIL LOUDLY.  Any structural problem (bad attribute name, invalid
clause_type, duplicate approved templates) raises ValueError with a precise
message, not a silent warning.  A misconfigured playbook would make the deviation
engine produce nonsense, so we catch it at load time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from ..extraction.schemas import ClauseType
from .schemas import (
    DealRecord, KNOWN_ATTRIBUTES, Playbook, PlaybookConfig, StandardTemplate,
    TemplateAttribute,
)

# Default location: playbook/ at project root (sibling of app/).
_DEFAULT_PLAYBOOK_DIR = Path(__file__).resolve().parent.parent.parent / "playbook"


def _load_template(path: Path) -> StandardTemplate:
    """Parse one YAML template file and validate its contents."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Validate clause_type is a real ClauseType value
    ct = raw.get("clause_type", "")
    valid_types = {e.value for e in ClauseType}
    if ct not in valid_types:
        raise ValueError(
            f"Template {path.name}: clause_type={ct!r} is not a valid ClauseType.  "
            f"Valid values: {sorted(valid_types)}"
        )

    # Validate each attribute name against the known set
    raw_attrs = raw.get("attributes", [])
    for a in raw_attrs:
        if a.get("name") not in KNOWN_ATTRIBUTES:
            raise ValueError(
                f"Template {path.name}: attribute name {a.get('name')!r} is not a "
                f"known Layer 2 attribute.  Known names: {sorted(KNOWN_ATTRIBUTES)}"
            )

    # Build Pydantic model (field_validator also checks names as a safety net)
    attrs = [TemplateAttribute(**a) for a in raw_attrs]
    return StandardTemplate(
        template_id=raw["template_id"],
        clause_type=ct,
        version=str(raw["version"]),
        status=raw["status"],
        effective_from=str(raw["effective_from"]),
        standard_text=raw["standard_text"].strip(),
        attributes=attrs,
    )


def load_playbook(playbook_dir: Optional[Path] = None) -> Playbook:
    """Load and validate the complete playbook from disk.

    Args:
        playbook_dir: path to the playbook/ folder.  Defaults to
            <project_root>/playbook/.

    Returns:
        Playbook  (config + templates + history)

    Raises:
        ValueError  with a clear message if any validation fails.
        FileNotFoundError  if playbook_dir or config.yaml does not exist.
    """
    pb_dir = Path(playbook_dir) if playbook_dir else _DEFAULT_PLAYBOOK_DIR

    # --- config ---
    config_path = pb_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Playbook config not found: {config_path}")
    config_raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PlaybookConfig(**config_raw)

    # --- templates ---
    tpl_dir = pb_dir / "templates"
    templates: list[StandardTemplate] = []
    if tpl_dir.exists():
        for p in sorted(tpl_dir.glob("*.yaml")):
            templates.append(_load_template(p))

    # Validate: at most ONE approved template per clause_type
    approved: dict[str, str] = {}  # clause_type -> template_id
    for t in templates:
        if t.status != "approved":
            continue
        if t.clause_type in approved:
            raise ValueError(
                f"Two approved templates for clause_type={t.clause_type!r}: "
                f"{approved[t.clause_type]!r} and {t.template_id!r}.  "
                f"Only one approved template per clause type is allowed."
            )
        approved[t.clause_type] = t.template_id

    # --- history ---
    history: list[DealRecord] = []
    hist_path = pb_dir / "history.json"
    if hist_path.exists():
        raw_hist = json.loads(hist_path.read_text(encoding="utf-8"))
        history = [DealRecord(**r) for r in raw_hist]

    return Playbook(config=config, templates=templates, history=history)
