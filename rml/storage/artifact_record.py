from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    kind: str
    name: str
    relpath: str
    sha256: str
    size_bytes: int
    mime: Optional[str]
