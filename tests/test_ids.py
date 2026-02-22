from rml.core.ids import budget_id, eval_contract_id, program_id, run_id, task_spec_id, taskset_id
from tests.test_program_graph import make_min_program


def test_program_id_matches_hash():
    lp = make_min_program()
    assert program_id(lp) == lp.hash()


def test_taskset_id_order_invariant():
    t1 = {"family": "algos", "config": {"a": 1}}
    t2 = {"family": "algos", "config": {"a": 2}}
    assert taskset_id([t1, t2]) == taskset_id([t2, t1])


def test_run_id_changes_with_seed():
    pid = "p"
    tid = taskset_id([{"f": 1}])
    bid = budget_id({"steps": 10})
    eid = eval_contract_id({"budget": {"steps": 10}})
    r1 = run_id(pid, tid, bid, eid, rng_seed=1, runner_version="v0", engine_step=0)
    r2 = run_id(pid, tid, bid, eid, rng_seed=2, runner_version="v0", engine_step=0)
    assert r1 != r2
