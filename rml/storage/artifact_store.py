from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Optional

from rml.core.ids import canonical_bytes
from rml.storage.artifact_record import ArtifactRecord
from rml.storage.sqlite_store import SQLiteStore


class ArtifactIntegrityError(IOError):
    def __init__(self, msg: str, diagnostics: dict):
        super().__init__(msg)
        self.diagnostics = diagnostics


class ArtifactStore:
    def __init__(
        self,
        artifact_root: Path,
        db: Optional[SQLiteStore] = None,
        default_verify: str = "on_demand",
        default_strict: bool = True,
    ):
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.default_verify = default_verify
        self.default_strict = default_strict

    def _ensure_dir(self, run_id: str, kind: str) -> Path:
        path = self.artifact_root / run_id / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _atomic_write(self, dest: Path, data: bytes) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)

    def _compute_sha256(self, path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def save_bytes(
        self,
        run_id: str,
        kind: str,
        name: str,
        data: bytes,
        ext: str,
        mime: Optional[str] = None,
        register: bool = True,
        conn=None,
    ) -> ArtifactRecord:
        if ext and not ext.startswith("."):
            ext = "." + ext
        dest_dir = self._ensure_dir(run_id, kind)
        dest = dest_dir / f"{name}{ext}"
        self._atomic_write(dest, data)
        sha = self._compute_sha256(dest)
        size = dest.stat().st_size
        relpath = dest.relative_to(self.artifact_root).as_posix()
        record = ArtifactRecord(
            artifact_id=sha,
            run_id=run_id,
            kind=kind,
            name=name,
            relpath=relpath,
            sha256=sha,
            size_bytes=size,
            mime=mime,
        )
        if self.db and register:
            self.db.insert_artifact(record, conn=conn)
        return record

    def save_json(
        self,
        run_id: str,
        kind: str,
        name: str,
        obj: Any,
        gzip_compress: bool = False,
        register: bool = True,
        conn=None,
    ) -> ArtifactRecord:
        data = canonical_bytes(obj)
        if gzip_compress:
            data = gzip.compress(data)
            ext = ".json.gz"
            mime = "application/json+gzip"
        else:
            ext = ".json"
            mime = "application/json"
        return self.save_bytes(run_id, kind, name, data, ext=ext, mime=mime, register=register, conn=conn)

    def save_file(
        self,
        run_id: str,
        kind: str,
        name: str,
        src_path: Path,
        mime: Optional[str] = None,
        register: bool = True,
        conn=None,
    ) -> ArtifactRecord:
        src_path = Path(src_path)
        data = src_path.read_bytes()
        ext = src_path.suffix
        return self.save_bytes(run_id, kind, name, data, ext=ext, mime=mime, register=register, conn=conn)

    def resolve_path(self, relpath: str) -> Path:
        return self.artifact_root / relpath

    def verify_artifact(self, record: ArtifactRecord) -> dict:
        abs_path = self.resolve_path(record.relpath)
        exists = abs_path.exists()
        actual_size = abs_path.stat().st_size if exists else None
        size_ok = (actual_size == record.size_bytes) if exists else False
        actual_sha = self._compute_sha256(abs_path) if exists else None
        sha_ok = (actual_sha == record.sha256) if exists else False
        ok = exists and size_ok and sha_ok
        return {
            "ok": ok,
            "path": str(abs_path),
            "exists": exists,
            "size_ok": size_ok,
            "sha_ok": sha_ok,
            "expected_size": record.size_bytes,
            "actual_size": actual_size,
            "expected_sha": record.sha256,
            "actual_sha": actual_sha,
        }

    def _maybe_verify(self, record: ArtifactRecord, mode: str, strict: bool) -> dict:
        if mode == "never":
            return {"ok": True}
        diag = self.verify_artifact(record)
        if not diag["ok"] and strict:
            raise ArtifactIntegrityError("Artifact verification failed", diag)
        return diag

    def read_bytes(
        self,
        record: ArtifactRecord,
        verify: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> bytes:
        verify = verify if verify is not None else self.default_verify
        strict = strict if strict is not None else self.default_strict
        self._maybe_verify(record, verify, strict)
        return self.resolve_path(record.relpath).read_bytes()

    def read_json(
        self,
        record: ArtifactRecord,
        verify: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Any:
        data = self.read_bytes(record, verify=verify, strict=strict)
        return json.loads(data.decode("utf-8"))

    def read_json_gz(
        self,
        record: ArtifactRecord,
        verify: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Any:
        data = self.read_bytes(record, verify=verify, strict=strict)
        decompressed = gzip.decompress(data)
        return json.loads(decompressed.decode("utf-8"))
