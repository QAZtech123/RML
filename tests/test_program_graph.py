import copy
from pathlib import Path

import pytest

from rml.core.program import (
    LearningProgram,
    ProgramGraph,
    ProgramNode,
    ProgramValidationError,
    apply_edits_atomic,
    get_by_path,
    hash_graph,
    set_by_path,
    to_canonical_json_bytes,
    try_apply_edits,
)
from rml.updater.meta_updater import GraphEdit


def make_min_program() -> LearningProgram:
    nodes = {
        "ARCH:0": ProgramNode(
            "ARCH:0",
            "ARCH",
            {
                "type": "transformer",
                "io": {"input_space": "token", "input_dim": 16, "output_space": "token", "output_dim": 16},
                "modules": [
                    {"name": "embed", "kind": "embedding", "params": {"vocab_size": 128, "embed_dim": 16}},
                    {"name": "core", "kind": "transformer_block_stack", "params": {"d_model": 16, "n_heads": 2, "n_layers": 1, "mlp_ratio": 2, "pos": "rope"}},
                    {"name": "head", "kind": "mlp_head", "params": {"in_dim": 16, "out_dim": 16, "depth": 1}},
                ],
                "wiring": [{"src": "embed", "dst": "core"}, {"src": "core", "dst": "head"}],
                "init": {"scheme": "xavier", "seeded": True},
                "regularization": {"dropout": 0.0, "layer_norm": True, "weight_decay": 0.0},
            },
        ),
        "LRULE:0": ProgramNode(
            "LRULE:0",
            "LRULE",
            {
                "type": "adam",
                "hyper": {"base_lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08},
                "schedule": {"kind": "constant", "warmup_steps": 0, "max_steps": 50, "min_lr": 1e-05},
                "grad": {"clip_norm": 1.0, "clip_value": 0.0, "normalize": False, "noise_std": 0.0},
                "meta": {"supports_hyper_updates": True, "supports_rule_edits": True},
            },
        ),
        "OBJ:0": ProgramNode(
            "OBJ:0",
            "OBJ",
            {
                "type": "supervised",
                "losses": [{"name": "task", "kind": "cross_entropy", "params": {}}],
                "weights": {"combine": "weighted_sum", "values": {"task": 1.0}},
                "constraints": [{"name": "l2", "kind": "weight_decay", "strength": 0.0}],
                "targets": {"head": "head", "label_space": "class"},
            },
        ),
        "CURR:0": ProgramNode(
            "CURR:0",
            "CURR",
            {
                "type": "uniform",
                "family_weights": {"base": 1.0},
                "state": {"memory": "none", "ema_beta": 0.0, "window": 1},
                "update_rule": {"signal": "loss", "exploration": "softmax", "temperature": 1.0, "epsilon": 0.0},
            },
        ),
        "MEM:0": ProgramNode(
            "MEM:0",
            "MEM",
            {"type": "none", "capacity": 0, "encoding": {}, "retrieval": {}, "write_policy": {}, "decay": {}},
        ),
        "EVALC:0": ProgramNode(
            "EVALC:0",
            "EVALC",
            {
                "budgets": {"inner": {"max_steps": 50, "max_seconds": 1.0, "max_memory_mb": 128}},
                "protocol": {"train_tasks": 1, "shift_tasks": 1, "unseen_tasks": 0},
                "metrics_required": ["train_loss", "shift_score"],
                "traces_required": ["loss_curve"],
                "determinism": {"seeded": True, "replayable": True},
            },
        ),
    }
    edges = [
        ("ARCH:0", "OBJ:0", "depends_on"),
        ("LRULE:0", "ARCH:0", "compatible_with"),
        ("CURR:0", "OBJ:0", "compatible_with"),
        ("MEM:0", "ARCH:0", "compatible_with"),
        ("EVALC:0", "ARCH:0", "depends_on"),
        ("EVALC:0", "OBJ:0", "depends_on"),
        ("EVALC:0", "CURR:0", "depends_on"),
    ]
    return LearningProgram(graph=ProgramGraph(nodes=nodes, edges=edges), constraints={}, meta={})


def test_validate_minimal_graph_ok():
    lp = make_min_program()
    lp.graph.validate()


def test_hash_is_stable_and_ignores_meta(tmp_path):
    lp1 = make_min_program()
    # Same graph, different construction order
    lp2 = make_min_program()
    lp2.graph.edges = list(reversed(lp2.graph.edges))
    lp2.meta["note"] = "changed"
    assert lp1.graph.hash() == lp2.graph.hash()
    assert lp1.hash() == lp2.hash()


def test_hash_matches_canonical_snapshot():
    lp = make_min_program()
    canonical = Path("tests/data/canonical.json").read_bytes()
    assert to_canonical_json_bytes(lp.graph) == canonical
    assert lp.graph.hash() == "230562b4d47026369f9b536d9aabfe19a694c75b4dac7c651680c7491fdd519f"


def test_get_set_by_path_roundtrip_and_no_inplace_mutation():
    lp = make_min_program()
    set_by_path(lp, "LRULE:0.spec.hyper.base_lr", 0.123)
    assert get_by_path(lp, "LRULE:0.spec.hyper.base_lr") == 0.123
    modules_copy = get_by_path(lp, "ARCH:0.spec.modules")
    modules_copy.append({"name": "extra", "kind": "mlp_head", "params": {"in_dim": 1, "out_dim": 1, "depth": 1}})
    assert len(lp.graph.nodes["ARCH:0"].spec["modules"]) == 3


def test_apply_edits_atomic_success():
    lp = make_min_program()
    edits = [
        GraphEdit(kind="tune_param", target="LRULE:0.spec.hyper.base_lr", value=0.0005, meta={"id": "lr"}),
        GraphEdit(
            kind="add_edge",
            target="",
            value={"src": "LRULE:0", "dst": "MEM:0", "rel": "shares_state_with"},
            meta={"id": "edge"},
        ),
    ]
    new_lp, applied = apply_edits_atomic(lp, edits)
    assert applied == ["lr", "edge"]
    assert get_by_path(new_lp, "LRULE:0.spec.hyper.base_lr") == 0.0005
    new_lp.graph.validate()


def test_singleton_missing_and_duplicate():
    lp = make_min_program()
    del lp.graph.nodes["OBJ:0"]
    with pytest.raises(ProgramValidationError) as excinfo:
        lp.graph.validate()
    assert excinfo.value.code == "GRAPH_SINGLETON_VIOLATION"

    lp2 = make_min_program()
    lp2.graph.nodes["ARCH:1"] = ProgramNode("ARCH:1", "ARCH", copy.deepcopy(lp2.graph.nodes["ARCH:0"].spec))
    with pytest.raises(ProgramValidationError) as excinfo2:
        lp2.graph.validate()
    assert excinfo2.value.code == "GRAPH_SINGLETON_VIOLATION"


def test_edge_violations():
    lp = make_min_program()
    lp.graph.edges.append(("ARCH:0", "OBJ:0", "bogus"))
    with pytest.raises(ProgramValidationError) as excinfo:
        lp.graph.validate()
    assert excinfo.value.code == "EDGE_UNKNOWN_RELATION"

    lp2 = make_min_program()
    lp2.graph.edges.append(("UNKNOWN:0", "OBJ:0", "depends_on"))
    with pytest.raises(ProgramValidationError) as excinfo2:
        lp2.graph.validate()
    assert excinfo2.value.code == "EDGE_MISSING_NODE"

    lp3 = make_min_program()
    lp3.graph.edges.append(lp3.graph.edges[0])
    with pytest.raises(ProgramValidationError) as excinfo3:
        lp3.graph.validate()
    assert excinfo3.value.code == "EDGE_DUPLICATE"

    lp4 = make_min_program()
    lp4.graph.edges.append(("ARCH:0", "OBJ:0", "compatible_with"))
    lp4.graph.edges.append(("OBJ:0", "ARCH:0", "incompatible_with"))
    with pytest.raises(ProgramValidationError) as excinfo4:
        lp4.graph.validate()
    assert excinfo4.value.code == "EDGE_CONTRADICTION"


def test_schema_and_path_violations():
    lp = make_min_program()
    set_by_path(lp, "LRULE:0.spec.hyper.base_lr", float("nan"))
    with pytest.raises(ProgramValidationError) as excinfo:
        lp.graph.validate()
    assert excinfo.value.code == "NODE_SCHEMA_BAD_HYPER"

    lp2 = make_min_program()
    lp2.graph.nodes["ARCH:0"].spec.pop("modules")
    with pytest.raises(ProgramValidationError) as excinfo2:
        lp2.graph.validate()
    assert excinfo2.value.code == "NODE_SCHEMA_MISSING_FIELD"

    lp3 = make_min_program()
    with pytest.raises(IndexError):
        set_by_path(lp3, "ARCH:0.spec.modules[5]", {})


def test_compatibility_violations():
    lp = make_min_program()
    set_by_path(lp, "OBJ:0.spec.losses", [{"name": "pg", "kind": "policy_gradient", "params": {}}])
    set_by_path(lp, "OBJ:0.spec.weights", {"combine": "weighted_sum", "values": {"pg": 1.0}})
    with pytest.raises(ProgramValidationError) as excinfo:
        lp.graph.validate()
    assert excinfo.value.code == "COMPAT_POLICY_HEAD_REQUIRED"

    lp2 = make_min_program()
    set_by_path(lp2, "MEM:0.spec.type", "episodic_kv")
    set_by_path(lp2, "MEM:0.spec.capacity", 10)
    set_by_path(lp2, "MEM:0.spec.encoding", {"key_dim": 8, "value_dim": 8, "compress": "none"})
    set_by_path(lp2, "MEM:0.spec.retrieval", {"kind": "knn", "k": 1, "temperature": 1.0})
    set_by_path(lp2, "MEM:0.spec.write_policy", {"kind": "always", "threshold": 0.0})
    set_by_path(lp2, "MEM:0.spec.decay", {"kind": "none"})
    with pytest.raises(ProgramValidationError) as excinfo2:
        lp2.graph.validate()
    assert excinfo2.value.code == "COMPAT_MEMORY_READER_REQUIRED"

    lp3 = make_min_program()
    set_by_path(lp3, "LRULE:0.spec.schedule.max_steps", 200)
    with pytest.raises(ProgramValidationError) as excinfo3:
        lp3.graph.validate()
    assert excinfo3.value.code in {"COMPAT_SCHEDULE_EXCEEDS_BUDGET", "EVALC_SCHEDULE_MISMATCH"}


def test_edit_rollback_on_failure():
    lp = make_min_program()
    original_hash = lp.hash()
    edits = [
        GraphEdit(kind="tune_param", target="LRULE:0.spec.hyper.base_lr", value=0.0007, meta={"id": "lr"}),
        GraphEdit(kind="add_edge", target="", value={"src": "ARCH:0", "dst": "MISSING:0", "rel": "depends_on"}, meta={"id": "bad"}),
    ]
    new_lp, diag = try_apply_edits(lp, edits)
    assert diag["ok"] is False
    assert diag["error_code"] == "EDGE_MISSING_NODE"
    assert diag["legacy_code"] == "EDGE_UNKNOWN_NODE"
    assert lp.hash() == original_hash
    assert new_lp is lp
    # original untouched
    assert get_by_path(lp, "LRULE:0.spec.hyper.base_lr") == 0.001


def test_get_by_path_returns_copy_for_mutables():
    lp = make_min_program()
    mods = get_by_path(lp, "ARCH:0.spec.modules")
    mods[0]["params"]["embed_dim"] = 999
    assert get_by_path(lp, "ARCH:0.spec.modules")[0]["params"]["embed_dim"] == 16
