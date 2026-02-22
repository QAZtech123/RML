from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Sequence

from rml.core.program import _canonicalize, LearningProgram, ProgramGraph


def _to_primitive(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON bytes for arbitrary JSON-like objects."""
    canon = _canonicalize(_to_primitive(obj))
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def program_id(program: ProgramGraph | LearningProgram) -> str:
    return program.hash() if hasattr(program, "hash") else hash_graph(program)  # type: ignore


def task_spec_id(task_spec: Any) -> str:
    return _hash_bytes(canonical_bytes(_to_primitive(task_spec)))


def taskset_id(task_specs: Sequence[Any]) -> str:
    ids = sorted(task_spec_id(ts) for ts in task_specs)
    return _hash_bytes(canonical_bytes(ids))


def budget_id(budget: Any) -> str:
    return _hash_bytes(canonical_bytes(_to_primitive(budget)))


def eval_contract_id(evalc_spec: Any) -> str:
    return _hash_bytes(canonical_bytes(_to_primitive(evalc_spec)))

def batch_id(step_idx: int, run_ids: Sequence[str], dist_snapshot: Any, updater_snapshot: Any) -> str:
    payload = {
        "step_idx": step_idx,
        "run_ids": sorted(run_ids),
        "dist_snapshot": _to_primitive(dist_snapshot),
        "updater_snapshot": _to_primitive(updater_snapshot),
    }
    return _hash_bytes(canonical_bytes(payload))

def run_id(
    program_id: str,
    taskset_id: str,
    budget_id: str,
    evalc_id: str,
    rng_seed: int,
    runner_version: str,
    engine_step: int,
    include_step: bool = True,
    include_taskset: bool = True,
    warm_start_key: str | None = None,
) -> str:
    payload = {
        "program_id": program_id,
        "budget_id": budget_id,
        "evalc_id": evalc_id,
        "rng_seed": rng_seed,
        "runner_version": runner_version,
    }
    if include_taskset:
        payload["taskset_id"] = taskset_id
    if include_step:
        payload["engine_step"] = engine_step
    if warm_start_key:
        payload["warm_start_key"] = warm_start_key
    return _hash_bytes(canonical_bytes(payload))


# backward import for ProgramGraph hashing when program_id receives a graph
def hash_graph(graph: ProgramGraph) -> str:
    return graph.hash()
