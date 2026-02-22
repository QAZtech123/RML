from __future__ import annotations

import json
from pathlib import Path

from rml.storage.artifact_store import ArtifactStore, ArtifactIntegrityError
from rml.storage.sqlite_store import SQLiteStore
from rml.storage.sweeper import sweep_orphans


def _load_store(args) -> SQLiteStore:
    return SQLiteStore(Path(args.db))


def _load_artifact_store(args, store: SQLiteStore) -> ArtifactStore:
    return ArtifactStore(Path(args.artifact_root), db=store)


def sweep_orphans_cmd(args) -> None:
    store = _load_store(args)
    result = sweep_orphans(store, Path(args.artifact_root), delete=args.delete)
    action = "deleted" if args.delete else "moved"
    print(f"Orphan sweep: kept={len(result['kept'])}, {action}={len(result['removed' if args.delete else 'moved'])}")
    if result["moved"]:
        print("Moved:", ", ".join(result["moved"]))
    if result["removed"]:
        print("Removed:", ", ".join(result["removed"]))


def verify_runs_cmd(args) -> None:
    store = _load_store(args)
    astore = _load_artifact_store(args, store)
    runs = store.list_recent_runs(limit=args.n, status=args.status)
    total = len(runs)
    ok = 0
    failures = []
    for run in runs:
        run_id = run["run_id"]
        artifacts = store.list_artifacts(run_id)
        all_ok = True
        for rec in artifacts:
            diag = astore.verify_artifact(rec)
            if not diag["ok"]:
                all_ok = False
                failures.append((run_id, rec.name, diag))
                if args.strict:
                    raise ArtifactIntegrityError(f"Artifact failed verify: {run_id}/{rec.name}", diag)
        if all_ok:
            ok += 1
    print(f"Verify runs: total={total}, ok={ok}, failures={len(failures)}")
    if failures:
        for run_id, name, diag in failures:
            print(f"- {run_id} :: {name} :: exists={diag['exists']} sha_ok={diag['sha_ok']} size_ok={diag['size_ok']}")


def show_run_cmd(args) -> None:
    store = _load_store(args)
    run = store.get_run(args.run_id)
    if not run:
        print(f"Run not found: {args.run_id}")
        return
    astore = _load_artifact_store(args, store)
    print(json.dumps({k: v for k, v in run.items() if not k.endswith("_json")}, indent=2))
    artifacts = store.list_artifacts(args.run_id)
    if not artifacts:
        print("No artifacts.")
        return
    print("Artifacts:")
    for rec in artifacts:
        line = f"- {rec.kind}:{rec.name} path={rec.relpath} sha={rec.sha256}"
        if args.verify:
            diag = astore.verify_artifact(rec)
            line += f" verify_ok={diag['ok']}"
        print(line)
