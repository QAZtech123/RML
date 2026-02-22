from __future__ import annotations

from dataclasses import dataclass

from rml.core import ids
from rml.core.program import LearningProgram

@dataclass(frozen=True)
class RunContext:
    program_id: str
    run_id: str
    taskset_id: str
    budget_id: str
    eval_contract_id: str
    seed: int
    runner_version: str
    engine_step: int

    @classmethod
    def from_parts(
        cls,
        *,
        program: LearningProgram,
        task_specs: list,
        budget,
        eval_contract,
        seed: int,
        runner_version: str,
        engine_step: int,
        cache_scope: str = "step",
        warm_start_key: str | None = None,
    ) -> "RunContext":
        pid = ids.program_id(program.graph)
        tset = ids.taskset_id(task_specs)
        bid = ids.budget_id(budget)
        evid = ids.eval_contract_id(eval_contract)
        include_step = cache_scope == "step"
        include_taskset = cache_scope != "program"
        rid = ids.run_id(
            program_id=pid,
            taskset_id=tset,
            budget_id=bid,
            evalc_id=evid,
            rng_seed=seed,
            runner_version=runner_version,
            engine_step=engine_step,
            include_step=include_step,
            include_taskset=include_taskset,
            warm_start_key=warm_start_key,
        )
        return cls(
            program_id=pid,
            run_id=rid,
            taskset_id=tset,
            budget_id=bid,
            eval_contract_id=evid,
            seed=seed,
            runner_version=runner_version,
            engine_step=engine_step,
        )
