from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List

from rml.storage.sqlite_store import SQLiteStore


def sweep_orphans(store: SQLiteStore, artifact_root: Path, delete: bool = False) -> Dict[str, List[str]]:
    root = Path(artifact_root)
    orphan_dir = root / "_orphaned"
    orphan_dir.mkdir(parents=True, exist_ok=True)

    moved: List[str] = []
    removed: List[str] = []
    kept: List[str] = []

    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("_"):
            continue
        run_id = path.name
        if store.has_run(run_id):
            kept.append(run_id)
            continue
        if delete:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(run_id)
        else:
            dest = orphan_dir / run_id
            if dest.exists():
                dest = orphan_dir / f"{run_id}_{int(time.time())}"
            shutil.move(str(path), str(dest))
            moved.append(run_id)

    return {"kept": kept, "moved": moved, "removed": removed}
