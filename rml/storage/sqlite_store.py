from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from rml.storage.artifact_record import ArtifactRecord

PRAGMAS = [
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
]


@dataclass(frozen=True)
class CacheHit:
    run_id: str
    loaded_run: Dict[str, Any]
    artifacts: List[ArtifactRecord]


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        for key, value in PRAGMAS:
            conn.execute(f"PRAGMA {key}={value}")
        return conn

    @contextmanager
    def transaction(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS programs (
                  program_id TEXT PRIMARY KEY,
                  graph_canonical_json TEXT NOT NULL,
                  schema_version INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                  parents_json TEXT NOT NULL DEFAULT '[]',
                  edit_trace_json TEXT NOT NULL DEFAULT '[]',
                  meta_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_programs_created_at ON programs(created_at);

                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  program_id TEXT NOT NULL,
                  taskset_id TEXT NOT NULL,
                  budget_id TEXT NOT NULL,
                  eval_contract_id TEXT NOT NULL,
                  seed INTEGER NOT NULL,
                  runner_version TEXT NOT NULL,
                  engine_step INTEGER NOT NULL,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

                  task_specs_json TEXT NOT NULL,
                  budget_json TEXT NOT NULL,
                  eval_contract_json TEXT NOT NULL,

                  metrics_json TEXT NOT NULL DEFAULT '{}',
                  traces_json TEXT NOT NULL DEFAULT '{}',
                  artifacts_json TEXT NOT NULL DEFAULT '{}',

                  status TEXT NOT NULL DEFAULT 'ok',
                  error_code TEXT,
                  error_json TEXT,

                  FOREIGN KEY (program_id) REFERENCES programs(program_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_program_id ON runs(program_id);
                CREATE INDEX IF NOT EXISTS idx_runs_taskset ON runs(taskset_id);
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

                CREATE TABLE IF NOT EXISTS batches (
                  batch_id TEXT PRIMARY KEY,
                  engine_step INTEGER NOT NULL,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                  rng INTEGER NOT NULL,
                  dist_snapshot_json TEXT NOT NULL,
                  updater_snapshot_json TEXT NOT NULL,
                  run_ids_json TEXT NOT NULL,
                episode_summaries_json TEXT NOT NULL DEFAULT '[]'
                ,meta_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_batches_engine_step ON batches(engine_step);
                CREATE INDEX IF NOT EXISTS idx_batches_created_at ON batches(created_at);

                CREATE TABLE IF NOT EXISTS artifacts (
                  artifact_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  name TEXT NOT NULL,
                  relpath TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  size_bytes INTEGER NOT NULL,
                  mime TEXT,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                  FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
                """
            )

    # --- programs ---
    def upsert_program(
        self,
        program_id: str,
        graph_canonical_json: str,
        schema_version: int = 1,
        parents: Optional[Iterable[str]] = None,
        edit_trace: Optional[Iterable[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        parents_json = _json_dumps(list(parents or []))
        edit_trace_json = _json_dumps(list(edit_trace or []))
        meta_json = _json_dumps(meta or {})
        if conn is not None:
            conn.execute(
                """
                INSERT OR IGNORE INTO programs(program_id, graph_canonical_json, schema_version, parents_json, edit_trace_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (program_id, graph_canonical_json, schema_version, parents_json, edit_trace_json, meta_json),
            )
        else:
            with self._connect() as c:
                c.execute(
                    """
                    INSERT OR IGNORE INTO programs(program_id, graph_canonical_json, schema_version, parents_json, edit_trace_json, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (program_id, graph_canonical_json, schema_version, parents_json, edit_trace_json, meta_json),
                )

    def get_program(self, program_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM programs WHERE program_id=?", (program_id,))
            row = cur.fetchone()
            if not row:
                return None
            columns = [col[0] for col in cur.description]
            return dict(zip(columns, row))

    # --- runs ---
    def has_run(self, run_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM runs WHERE run_id=? LIMIT 1", (run_id,))
            return cur.fetchone() is not None

    def insert_run(self, run: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> bool:
        """Insert a run; returns True if inserted, False if already exists."""
        payload = {
            "run_id": run["run_id"],
            "program_id": run["program_id"],
            "taskset_id": run["taskset_id"],
            "budget_id": run["budget_id"],
            "eval_contract_id": run["eval_contract_id"],
            "seed": run["seed"],
            "runner_version": run["runner_version"],
            "engine_step": run["engine_step"],
            "task_specs_json": _json_dumps(run["task_specs"]),
            "budget_json": _json_dumps(run["budget"]),
            "eval_contract_json": _json_dumps(run["eval_contract"]),
            "metrics_json": _json_dumps(run.get("metrics", {})),
            "traces_json": _json_dumps(run.get("traces", {})),
            "artifacts_json": _json_dumps(run.get("artifacts", {})),
            "status": run.get("status", "ok"),
            "error_code": run.get("error_code"),
            "error_json": _json_dumps(run.get("error_json")) if run.get("error_json") is not None else None,
        }
        try:
            if conn is not None:
                conn.execute(
                    """
                    INSERT INTO runs(run_id, program_id, taskset_id, budget_id, eval_contract_id, seed, runner_version, engine_step,
                                     task_specs_json, budget_json, eval_contract_json, metrics_json, traces_json, artifacts_json,
                                     status, error_code, error_json)
                    VALUES (:run_id, :program_id, :taskset_id, :budget_id, :eval_contract_id, :seed, :runner_version, :engine_step,
                            :task_specs_json, :budget_json, :eval_contract_json, :metrics_json, :traces_json, :artifacts_json,
                            :status, :error_code, :error_json)
                    """,
                    payload,
                )
            else:
                with self._connect() as c:
                    c.execute(
                        """
                        INSERT INTO runs(run_id, program_id, taskset_id, budget_id, eval_contract_id, seed, runner_version, engine_step,
                                         task_specs_json, budget_json, eval_contract_json, metrics_json, traces_json, artifacts_json,
                                         status, error_code, error_json)
                        VALUES (:run_id, :program_id, :taskset_id, :budget_id, :eval_contract_id, :seed, :runner_version, :engine_step,
                                :task_specs_json, :budget_json, :eval_contract_json, :metrics_json, :traces_json, :artifacts_json,
                                :status, :error_code, :error_json)
                        """,
                        payload,
                    )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            columns = [col[0] for col in cur.description]
            rec = dict(zip(columns, row))
            # decode json fields back to python objects
            for key in ("task_specs_json", "budget_json", "eval_contract_json", "metrics_json", "traces_json", "artifacts_json", "error_json"):
                if rec.get(key) is not None:
                    rec[key] = json.loads(rec[key])
            # provide normalized aliases expected by evaluator/runner
            if "metrics_json" in rec and "metrics" not in rec:
                rec["metrics"] = rec["metrics_json"]
            if "traces_json" in rec and "traces" not in rec:
                rec["traces"] = rec["traces_json"]
            if "artifacts_json" in rec and "artifacts" not in rec:
                rec["artifacts"] = rec["artifacts_json"]
            if "task_specs_json" in rec and "task_specs" not in rec:
                rec["task_specs"] = rec["task_specs_json"]
            if "budget_json" in rec and "budget" not in rec:
                rec["budget"] = rec["budget_json"]
            if "eval_contract_json" in rec and "eval_contract" not in rec:
                rec["eval_contract"] = rec["eval_contract_json"]
            return rec

    def list_recent_runs(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM runs"
        params: List[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            out: List[Dict[str, Any]] = []
            for row in rows:
                rec = dict(zip(cols, row))
                for key in ("task_specs_json", "budget_json", "eval_contract_json", "metrics_json", "traces_json", "artifacts_json", "error_json"):
                    if rec.get(key) is not None:
                        rec[key] = json.loads(rec[key])
                out.append(rec)
            return out

    # --- batches history helpers ---
    def list_recent_batches(self, limit: int = 20) -> List[Dict[str, Any]]:
        query = "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            cur = conn.execute(query, (limit,))
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            out: List[Dict[str, Any]] = []
            for row in rows:
                rec = dict(zip(cols, row))
                for key in ("run_ids_json", "episode_summaries_json", "meta_json", "dist_snapshot_json", "updater_snapshot_json"):
                    if rec.get(key) is not None:
                        rec[key] = json.loads(rec[key])
                out.append(rec)
            return out

    def get_rolling_best(self, metric_key: str, window: int = 20) -> Optional[float]:
        """Return the max value of meta[metric_key] over the most recent batches."""
        batches = self.list_recent_batches(limit=window)
        vals: List[float] = []
        for b in batches:
            meta = b.get("meta_json") or {}
            if metric_key in meta and meta[metric_key] is not None:
                try:
                    vals.append(float(meta[metric_key]))
                except Exception:
                    continue
        if not vals:
            return None
        return max(vals)

    # --- batches ---
    def insert_batch(self, batch: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> bool:
        payload = {
            "batch_id": batch["batch_id"],
            "engine_step": batch["engine_step"],
            "rng": batch["rng"],
            "dist_snapshot_json": _json_dumps(batch["dist_snapshot"]),
            "updater_snapshot_json": _json_dumps(batch["updater_snapshot"]),
            "run_ids_json": _json_dumps(batch["run_ids"]),
            "episode_summaries_json": _json_dumps(batch.get("episode_summaries", [])),
            "meta_json": _json_dumps(batch.get("meta", {})),
        }
        try:
            if conn is not None:
                conn.execute(
                    """
                    INSERT INTO batches(batch_id, engine_step, rng, dist_snapshot_json, updater_snapshot_json, run_ids_json, episode_summaries_json, meta_json)
                    VALUES (:batch_id, :engine_step, :rng, :dist_snapshot_json, :updater_snapshot_json, :run_ids_json, :episode_summaries_json, :meta_json)
                    """,
                    payload,
                )
            else:
                with self._connect() as c:
                    c.execute(
                        """
                        INSERT INTO batches(batch_id, engine_step, rng, dist_snapshot_json, updater_snapshot_json, run_ids_json, episode_summaries_json, meta_json)
                        VALUES (:batch_id, :engine_step, :rng, :dist_snapshot_json, :updater_snapshot_json, :run_ids_json, :episode_summaries_json, :meta_json)
                        """,
                        payload,
                    )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_recent_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            result = []
            for row in rows:
                rec = dict(zip(cols, row))
                rec["dist_snapshot_json"] = json.loads(rec["dist_snapshot_json"])
                rec["updater_snapshot_json"] = json.loads(rec["updater_snapshot_json"])
                rec["run_ids_json"] = json.loads(rec["run_ids_json"])
                rec["episode_summaries_json"] = json.loads(rec["episode_summaries_json"])
                result.append(rec)
            return result

    # --- artifacts ---
    def insert_artifact(self, record: "ArtifactRecord", conn: Optional[sqlite3.Connection] = None) -> bool:
        try:
            if conn is not None:
                conn.execute(
                    """
                    INSERT INTO artifacts(artifact_id, run_id, kind, name, relpath, sha256, size_bytes, mime)
                    VALUES (:artifact_id, :run_id, :kind, :name, :relpath, :sha256, :size_bytes, :mime)
                    """,
                    asdict(record),
                )
            else:
                with self._connect() as c:
                    c.execute(
                        """
                        INSERT INTO artifacts(artifact_id, run_id, kind, name, relpath, sha256, size_bytes, mime)
                        VALUES (:artifact_id, :run_id, :kind, :name, :relpath, :sha256, :size_bytes, :mime)
                        """,
                        asdict(record),
                    )
            return True
        except sqlite3.IntegrityError:
            return False

    def list_artifacts(self, run_id: str) -> List[ArtifactRecord]:
        with self._connect() as conn:
            cur = conn.execute("SELECT artifact_id, run_id, kind, name, relpath, sha256, size_bytes, mime FROM artifacts WHERE run_id=?", (run_id,))
            rows = cur.fetchall()
            return [
                ArtifactRecord(
                    artifact_id=row[0],
                    run_id=row[1],
                    kind=row[2],
                    name=row[3],
                    relpath=row[4],
                    sha256=row[5],
                    size_bytes=row[6],
                    mime=row[7],
                )
                for row in rows
            ]

    def maybe_get_cached_run(
        self,
        run_id: str,
        require_status_ok: bool = True,
        require_runner_version: Optional[str] = None,
        allow_version_mismatch: bool = False,
    ) -> Optional[CacheHit]:
        run = self.get_run(run_id)
        if not run:
            return None
        if require_status_ok and run.get("status") != "ok":
            return None
        if require_runner_version and not allow_version_mismatch and run.get("runner_version") != require_runner_version:
            return None
        artifacts = self.list_artifacts(run_id)
        return CacheHit(run_id=run_id, loaded_run=run, artifacts=artifacts)
