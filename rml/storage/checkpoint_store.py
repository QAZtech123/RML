from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CheckpointRecord:
    path: Path
    meta: Dict[str, Any]
    state_dict: Dict[str, Any]


class CheckpointStore:
    """
    Simple file-backed checkpoint store. Keeps an append-only index.jsonl for fast lookup.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _safe_name(self, s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s)

    def save(self, arch_type: str, signature: str, state_dict: Dict[str, Any], meta: Dict[str, Any]) -> Path:
        import torch

        meta = dict(meta or {})
        meta.setdefault("arch_type", arch_type)
        meta.setdefault("model_signature", signature)
        meta.setdefault("created_at", time.time())

        arch_safe = self._safe_name(arch_type)
        sig_safe = self._safe_name(signature)
        step = int(meta.get("step", -1))
        unseen = meta.get("unseen_score")
        unseen_str = f"{float(unseen):.4f}" if unseen is not None else "nan"
        fname = f"{arch_safe}__{sig_safe}__step{step}__unseen{unseen_str}.pt"
        path = self.root / fname
        meta.setdefault("checkpoint_id", fname)
        meta.setdefault("path", str(path))
        payload = {"state_dict": state_dict, "meta": meta}
        torch.save(payload, path)
        record = {
            "path": str(path),
            "arch_type": arch_type,
            "signature": signature,
            "step": step,
            "unseen_score": unseen,
            "created_at": meta.get("created_at", time.time()),
            "checkpoint_id": meta.get("checkpoint_id"),
        }
        for key in (
            "opt_type",
            "obj_primary",
            "lr_bin",
            "unseen_set_id",
            "transfer_set_id",
            "unseen_pool_idx",
            "regime_id",
            "regime_family_id",
            "transfer_unseen_accuracy",
        ):
            if key in meta:
                record[key] = meta.get(key)
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def load_best(
        self,
        arch_type: str,
        signature: Optional[str] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[CheckpointRecord]:
        import torch

        if not self.index_path.exists():
            return None
        best = None
        best_score = None
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("arch_type") != arch_type:
                    continue
                if signature is not None and rec.get("signature") != signature:
                    continue
                if meta_filter:
                    mismatch = False
                    for key, val in meta_filter.items():
                        if val is None:
                            continue
                        if rec.get(key) != val:
                            mismatch = True
                            break
                    if mismatch:
                        continue
                path = Path(rec.get("path", ""))
                if not path.exists():
                    continue
                score = rec.get("unseen_score")
                try:
                    score_val = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score_val = None
                if best is None or (score_val is not None and (best_score is None or score_val > best_score)):
                    best = path
                    best_score = score_val
        if best is None:
            return None
        payload = torch.load(best, map_location="cpu")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        state_dict = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
        return CheckpointRecord(path=best, meta=meta, state_dict=state_dict)
