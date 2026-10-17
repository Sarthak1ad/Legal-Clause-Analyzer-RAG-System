"""Step 5: document versioning.

Rule: a document's identity is its doc_id; its content identity is the SHA-256
of its normalised text. Same text -> same version (idempotent re-ingest).
Changed text -> next version number. Old versions are never overwritten, so
an old citation (which contains v1 in its span_id) still points at the exact
text it originally quoted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def register(data_dir: Path, doc_id: str, sha256: str) -> int:
    reg_path = data_dir / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    versions = reg.setdefault(doc_id, [])
    for v in versions:
        if v["sha256"] == sha256:
            return v["version"]
    version = len(versions) + 1
    versions.append({"version": version, "sha256": sha256,
                     "ingested_at": datetime.now(timezone.utc).isoformat()})
    data_dir.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg, indent=2))
    return version
