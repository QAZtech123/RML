from pathlib import Path

import pytest

from rml.core.ids import budget_id, eval_contract_id, program_id, run_id, taskset_id
from rml.core.program import to_canonical_json_bytes
from rml.storage.artifact_store import ArtifactStore, ArtifactIntegrityError
from rml.storage.sqlite_store import SQLiteStore
from rml.storage.sweeper import sweep_orphans
from tests.test_program_graph import make_min_program


def test_sqlite_store_and_artifacts(tmp_path):
    db_path = tmp_path / "rml.db"
    store = SQLiteStore(db_path)

    lp = make_min_program()
    pid = program_id(lp)
    store.upsert_program(
        program_id=pid,
        graph_canonical_json=to_canonical_json_bytes(lp.graph).decode("utf-8"),
        parents=[],
        edit_trace=[],
        meta={"creator": "test"},
    )

    task_specs = [{"family": "toy", "config": {"a": 1}}]
    budget = {"steps": 10, "seconds": 1.0}
    evalc = lp.graph.nodes["EVALC:0"].spec
    tset_id = taskset_id(task_specs)
    bud_id = budget_id(budget)
    eval_id = eval_contract_id(evalc)

    rid = run_id(
        program_id=pid,
        taskset_id=tset_id,
        budget_id=bud_id,
        evalc_id=eval_id,
        rng_seed=123,
        runner_version="testver",
        engine_step=0,
    )

    run = {
        "run_id": rid,
        "program_id": pid,
        "taskset_id": tset_id,
        "budget_id": bud_id,
        "eval_contract_id": eval_id,
        "seed": 123,
        "runner_version": "testver",
        "engine_step": 0,
        "task_specs": task_specs,
        "budget": budget,
        "eval_contract": evalc,
        "metrics": {"loss": 1.0},
        "traces": {},
        "artifacts": {},
        "status": "ok",
    }
    inserted = store.insert_run(run)
    assert inserted is True
    assert store.has_run(rid)
    assert store.get_run(rid)["metrics_json"]["loss"] == 1.0

    artifact_root = tmp_path / "artifacts"
    astore = ArtifactStore(artifact_root, db=store)
    rec1 = astore.save_json(rid, kind="context", name="task_specs", obj=task_specs, gzip_compress=False)
    rec2 = astore.save_bytes(rid, kind="trace", name="loss_curve", data=b"[0.5, 0.4]", ext="json", mime="application/json")

    assert (artifact_root / rec1.relpath).exists()
    assert (artifact_root / rec2.relpath).exists()

    artifacts = store.list_artifacts(rid)
    assert len(artifacts) == 2
    names = {a.name for a in artifacts}
    assert "task_specs" in names and "loss_curve" in names

    # cache hit
    hit = store.maybe_get_cached_run(rid, require_runner_version="testver")
    assert hit is not None
    assert len(hit.artifacts) == 2


def test_artifact_verify_and_read(tmp_path):
    db_path = tmp_path / "rml.db"
    store = SQLiteStore(db_path)
    artifact_root = tmp_path / "artifacts"
    astore = ArtifactStore(artifact_root, db=store)

    # Minimal run to attach artifacts
    lp = make_min_program()
    pid = program_id(lp)
    store.upsert_program(pid, to_canonical_json_bytes(lp.graph).decode("utf-8"))
    rid = "run1"
    store.insert_run(
        {
            "run_id": rid,
            "program_id": pid,
            "taskset_id": "ts",
            "budget_id": "b",
            "eval_contract_id": "e",
            "seed": 0,
            "runner_version": "v",
            "engine_step": 0,
            "task_specs": [],
            "budget": {},
            "eval_contract": {},
            "metrics": {},
            "traces": {},
            "artifacts": {},
            "status": "ok",
        }
    )

    rec = astore.save_json(rid, "trace", "full", {"a": 1}, gzip_compress=True)
    diag_ok = astore.verify_artifact(rec)
    assert diag_ok["ok"]
    loaded = astore.read_json_gz(rec, verify="on_demand", strict=True)
    assert loaded == {"a": 1}

    # tamper file to force sha mismatch
    path = astore.resolve_path(rec.relpath)
    path.write_bytes(b"corrupt")
    diag_bad = astore.verify_artifact(rec)
    assert not diag_bad["ok"]
    assert diag_bad["sha_ok"] is False
    with pytest.raises(ArtifactIntegrityError):
        astore.read_json_gz(rec, verify=None, strict=None)

    # missing file
    path.unlink(missing_ok=True)
    diag_missing = astore.verify_artifact(rec)
    assert diag_missing["exists"] is False


def test_read_default_strict_raises_on_mismatch(tmp_path):
    db_path = tmp_path / "rml.db"
    store = SQLiteStore(db_path)
    artifact_root = tmp_path / "artifacts"
    astore = ArtifactStore(artifact_root, db=store, default_verify="on_demand", default_strict=True)

    rid = "run2"
    pid = "pid"
    store.upsert_program(pid, "{}")
    store.insert_run(
        {
            "run_id": rid,
            "program_id": pid,
            "taskset_id": "ts",
            "budget_id": "b",
            "eval_contract_id": "e",
            "seed": 0,
            "runner_version": "v",
            "engine_step": 0,
            "task_specs": [],
            "budget": {},
            "eval_contract": {},
            "metrics": {},
            "traces": {},
            "artifacts": {},
            "status": "ok",
        }
    )

    rec = astore.save_json(rid, "context", "foo", {"x": 1})
    # tamper
    tamper = astore.resolve_path(rec.relpath)
    tamper.write_bytes(b"bad")
    with pytest.raises(ArtifactIntegrityError):
        astore.read_json(rec)  # should use defaults and raise


def test_sweep_orphans(tmp_path):
    db_path = tmp_path / "rml.db"
    store = SQLiteStore(db_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    store.upsert_program("p", "{}")

    # existing run folder without DB entry
    orphan = artifact_root / "orphan_run"
    (orphan / "trace").mkdir(parents=True, exist_ok=True)
    (orphan / "trace" / "dummy").write_text("x")

    # legit run folder with DB entry
    legit = artifact_root / "legit_run"
    (legit / "trace").mkdir(parents=True, exist_ok=True)
    (legit / "trace" / "dummy").write_text("y")
    store.insert_run(
        {
            "run_id": "legit_run",
            "program_id": "p",
            "taskset_id": "ts",
            "budget_id": "b",
            "eval_contract_id": "e",
            "seed": 0,
            "runner_version": "v",
            "engine_step": 0,
            "task_specs": [],
            "budget": {},
            "eval_contract": {},
            "metrics": {},
            "traces": {},
            "artifacts": {},
            "status": "ok",
        }
    )

    result = sweep_orphans(store, artifact_root, delete=False)
    assert "orphan_run" in result["moved"]
    assert "legit_run" in result["kept"]
    assert (artifact_root / "_orphaned" / "orphan_run").exists()
    assert (artifact_root / "legit_run").exists()
