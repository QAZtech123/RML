from __future__ import annotations

import argparse
import json
from pathlib import Path

from rml.app.engine_factory import AppConfig, build_engine
from rml.core.program import ProgramGraph, ProgramNode, LearningProgram
from rml.storage.sqlite_store import SQLiteStore


def _program_from_canonical(json_str: str) -> LearningProgram:
    payload = json.loads(json_str)
    nodes = {}
    for node in payload["nodes"]:
        nodes[node["id"]] = ProgramNode(id=node["id"], kind=node["kind"], spec=node["spec"])
    edges = [(e["src"], e["dst"], e["rel"]) for e in payload["edges"]]
    graph = ProgramGraph(nodes=nodes, edges=edges)
    prog = LearningProgram(graph=graph, constraints={}, meta={"parents": []})
    graph.validate()
    return prog


def replay_run_cmd(args) -> None:
    store = SQLiteStore(Path(args.db))
    run = store.get_run(args.run_id)
    if not run:
        raise SystemExit(f"Run not found: {args.run_id}")

    # Build engine (runner only needed)
    app_cfg = AppConfig(
        db_path=Path(args.db),
        artifact_root=Path(args.artifact_root),
        runner_version="replay",
        cache_scope="step",
    )
    engine = build_engine(app_cfg)

    # Load program from DB
    program_row = store.get_program(run["program_id"])
    if not program_row:
        raise SystemExit(f"Program not found for run: {run['program_id']}")
    program = _program_from_canonical(program_row["graph_canonical_json"])

    task_specs = run["task_specs_json"]
    budget = run["budget_json"]
    eval_contract = run["eval_contract_json"]
    seed = int(run.get("seed", 0))

    result = engine.runner.run(program=program, task_specs=task_specs, budget=budget, rng=seed)
    print(json.dumps({"original_metrics": run["metrics_json"], "replay_metrics": result.get("metrics", {})}, indent=2))


def add_replay_subparser(sub):
    p = sub.add_parser("replay-run", help="Replay a run by run_id using stored program/task specs")
    p.add_argument("run_id", help="Run ID to replay")
    p.add_argument("--db", default="rml.db")
    p.add_argument("--artifact-root", default="artifacts")
    return p
