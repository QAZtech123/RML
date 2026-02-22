from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rml.core.program import LearningProgram, ProgramGraph, ProgramNode

# Domains for discrete variables
ARCH_TYPES = ["transformer", "mlp"]
CORE_KINDS = ["transformer_block_stack", "mlp_stack"]
N_LAYERS = [2, 4, 6, 8, 12]
WIDTHS = [128, 256, 384, 512]
N_HEADS = [2, 4, 8, 12]
DROPOUTS = [0.0, 0.1, 0.2]
MEM_READER = ["none", "attention", "knn"]

LRULE_TYPES = ["adam", "sgd", "rmsprop", "learned_optimizer"]
LR_BINS = [8e-6, 2.4e-5, 8e-5, 2.4e-4, 8e-4]
SCHEDULE_KINDS = ["constant", "cosine", "linear_warmup_cosine"]
CLIP_NORM = [0.0, 0.5, 1.0, 2.0]

OBJ_TYPES = ["supervised", "self_supervised", "rl", "multi"]
OBJ_PRIMARY = ["cross_entropy", "mse", "contrastive", "policy_gradient"]
OBJ_ENTROPY_BONUS = ["off", "on"]

CURR_TYPES = ["uniform", "bandit", "adaptive", "self_paced"]
CURR_SIGNALS = ["loss", "reward", "learning_progress", "gen_gap"]
CURR_EXPLORATION = ["softmax", "epsilon_greedy", "ucb"]
CURR_MAX_LEN_TRAIN = "CURR.curriculum.max_len_train"

MEM_TYPES = ["none", "replay", "episodic_kv", "vector_db"]
MEM_CAPACITY = [0, 256, 1024, 4096]
MEM_RETRIEVAL = ["knn", "attention"]

BUDGET_STEPS = "BUDGET.steps"
# CPU-proof domain for constrained environments; adjust when scaling up.
BUDGET_STEPS_DOMAIN = [100, 200, 300]


@dataclass(frozen=True)
class Variable:
    name: str
    domain: List[Any]


def default_variables() -> List[Variable]:
    return [
        Variable("ARCH.type", ARCH_TYPES),
        Variable("ARCH.core.kind", CORE_KINDS),
        Variable("ARCH.core.n_layers", N_LAYERS),
        Variable("ARCH.core.width", WIDTHS),
        Variable("ARCH.core.n_heads", N_HEADS),
        Variable("ARCH.reg.dropout", DROPOUTS),
        Variable("ARCH.memory_reader", MEM_READER),
        Variable("LRULE.type", LRULE_TYPES),
        Variable("LRULE.lr_bin", LR_BINS),
        Variable("LRULE.schedule.kind", SCHEDULE_KINDS),
        Variable("LRULE.grad.clip_norm", CLIP_NORM),
        Variable("OBJ.type", OBJ_TYPES),
        Variable("OBJ.primary", OBJ_PRIMARY),
        Variable("OBJ.entropy_bonus", OBJ_ENTROPY_BONUS),
        Variable("CURR.type", CURR_TYPES),
        Variable("CURR.signal", CURR_SIGNALS),
        Variable("CURR.exploration", CURR_EXPLORATION),
        Variable(CURR_MAX_LEN_TRAIN, [16, 24, 32]),
        Variable("MEM.type", MEM_TYPES),
        Variable("MEM.capacity_bin", MEM_CAPACITY),
        Variable("MEM.retrieval.kind", MEM_RETRIEVAL),
        Variable(BUDGET_STEPS, BUDGET_STEPS_DOMAIN),
    ]


def _head_modules(primary: str) -> List[Dict[str, Any]]:
    if primary == "policy_gradient":
        return [{"name": "policy_head", "kind": "policy_head", "params": {"action_dim": 8, "distribution": "categorical"}}]
    return [{"name": "head", "kind": "mlp_head", "params": {"in_dim": 0, "out_dim": 0, "depth": 2}}]


def render_program_from_assignment(assignment: Dict[str, Any]) -> LearningProgram:
    # Cast numpy scalar samples to plain Python types to satisfy validators.
    arch_type = str(assignment["ARCH.type"])
    core_kind = str(assignment["ARCH.core.kind"])
    n_layers = int(assignment["ARCH.core.n_layers"])
    width = int(assignment["ARCH.core.width"])
    n_heads = int(assignment["ARCH.core.n_heads"])
    dropout = float(assignment["ARCH.reg.dropout"])
    mem_reader = str(assignment["ARCH.memory_reader"])

    # ARCH
    modules: List[Dict[str, Any]] = [
        {"name": "embed", "kind": "embedding", "params": {"vocab_size": 8000, "embed_dim": width}},
    ]
    if core_kind == "transformer_block_stack":
        modules.append(
            {"name": "core", "kind": "transformer_block_stack", "params": {"d_model": width, "n_heads": n_heads, "n_layers": n_layers, "mlp_ratio": 4, "pos": "rope"}}
        )
    elif core_kind == "gnn_stack":
        modules.append({"name": "core", "kind": "gnn_stack", "params": {"message_passing": "gat", "n_layers": n_layers, "hidden_dim": width, "agg": "mean"}})
    else:
        modules.append({"name": "core", "kind": "mlp_stack", "params": {"width": width, "depth": n_layers, "activation": "gelu"}})

    primary_loss = str(assignment["OBJ.primary"])
    head_modules = _head_modules(primary_loss)
    for hm in head_modules:
        if hm["kind"] == "mlp_head":
            hm["params"]["in_dim"] = width
            hm["params"]["out_dim"] = 8000 if primary_loss != "policy_gradient" else 1
    modules.extend(head_modules)

    if mem_reader != "none" and assignment["MEM.type"] != "none":
        modules.append({"name": "mem_reader", "kind": "memory_reader", "params": {"kind": mem_reader}})

    wiring = [{"src": "embed", "dst": "core"}, {"src": "core", "dst": head_modules[0]["name"]}]
    if mem_reader != "none" and assignment["MEM.type"] != "none":
        wiring.append({"src": "mem_reader", "dst": head_modules[0]["name"]})

    arch_node = ProgramNode(
        "ARCH:0",
        "ARCH",
        {
            "type": arch_type,
            "io": {"input_space": "token", "input_dim": width, "output_space": "token", "output_dim": width},
            "modules": modules,
            "wiring": wiring,
            "init": {"scheme": "xavier", "seeded": True},
            "regularization": {"dropout": dropout, "layer_norm": True, "weight_decay": 0.01},
        },
    )

    # LRULE
    lr = assignment["LRULE.lr_bin"]
    lrule_type = str(assignment["LRULE.type"])
    lrule_node = ProgramNode(
        "LRULE:0",
        "LRULE",
        {
            "type": lrule_type,
            "hyper": {"base_lr": float(lr), "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "momentum": 0.9},
            "schedule": {
                "kind": str(assignment["LRULE.schedule.kind"]),
                "warmup_steps": 100,
                "max_steps": 2000,
                "min_lr": 1e-5,
            },
            "grad": {
                "clip_norm": float(assignment["LRULE.grad.clip_norm"]),
                "clip_value": 0.0,
                "normalize": False,
                "noise_std": 0.0,
            },
            "meta": {"supports_hyper_updates": True, "supports_rule_edits": True},
        },
    )
    if lrule_type == "learned_optimizer":
        lrule_node.spec["learned"] = {
            "update_model": {"arch": "mlp", "hidden_dim": 64, "depth": 2, "state_dim": 16},
            "inputs": ["grad", "param", "moment", "loss", "step", "layer_id"],
            "outputs": ["delta_param", "delta_state"],
            "per_parameter": True,
            "shared_across_layers": False,
        }

    # OBJ
    entropy_bonus = str(assignment["OBJ.entropy_bonus"]) == "on"
    primary_kind = primary_loss
    obj_type = assignment["OBJ.type"]
    if primary_kind == "policy_gradient":
        obj_type = "rl"
    losses = [{"name": "primary", "kind": primary_kind, "params": {}}]
    constraints = [{"name": "l2", "kind": "weight_decay", "strength": 0.01}]
    if entropy_bonus and primary_kind == "policy_gradient":
        constraints.append({"name": "entropy", "kind": "policy_entropy_bonus", "strength": 0.01})
    obj_node = ProgramNode(
        "OBJ:0",
        "OBJ",
        {
            "type": obj_type,
            "losses": losses,
            "weights": {"combine": "weighted_sum", "values": {"primary": 1.0}},
            "constraints": constraints,
            "targets": {"head": head_modules[0]["name"], "label_space": "action" if primary_kind == "policy_gradient" else "class"},
        },
    )

    # CURR
    curr_node = ProgramNode(
        "CURR:0",
        "CURR",
        {
            "type": str(assignment["CURR.type"]),
            "curriculum": {"max_len_train": int(assignment.get(CURR_MAX_LEN_TRAIN, 32))},
            "family_weights": {"algos": 1.0},
            "state": {"memory": "ema", "ema_beta": 0.9, "window": 200},
            "update_rule": {
                "signal": str(assignment["CURR.signal"]),
                "exploration": str(assignment["CURR.exploration"]),
                "temperature": 1.0,
                "epsilon": 0.05,
            },
        },
    )

    # MEM
    mem_type = str(assignment["MEM.type"])
    mem_cap = int(assignment["MEM.capacity_bin"])
    mem_retrieval = str(assignment["MEM.retrieval.kind"])
    mem_node = ProgramNode(
        "MEM:0",
        "MEM",
        {
            "type": mem_type,
            "capacity": mem_cap,
            "encoding": {"key_dim": width // 2 or 1, "value_dim": width // 2 or 1, "compress": "none"},
            "retrieval": {"kind": mem_retrieval, "k": 4, "temperature": 1.0},
            "write_policy": {"kind": "always", "threshold": 0.0},
            "decay": {"kind": "none"},
        },
    )
    if mem_type == "none":
        mem_node.spec["capacity"] = 0
        mem_node.spec["encoding"] = {}
        mem_node.spec["retrieval"] = {}
        mem_node.spec["write_policy"] = {}
        mem_node.spec["decay"] = {}

    # EVALC default
    evalc_node = ProgramNode(
        "EVALC:0",
        "EVALC",
        {
            "budgets": {"inner": {"max_steps": 2000, "max_seconds": 30.0, "max_memory_mb": 4096}},
            "protocol": {"train_tasks": 4, "shift_tasks": 2, "unseen_tasks": 2},
            "metrics_required": ["train_loss", "shift_score", "unseen_score", "compute_seconds", "steps_to_threshold"],
            "traces_required": ["loss_curve", "grad_norm_curve"],
            "determinism": {"seeded": True, "replayable": True},
        },
    )

    nodes = {
        "ARCH:0": arch_node,
        "LRULE:0": lrule_node,
        "OBJ:0": obj_node,
        "CURR:0": curr_node,
        "MEM:0": mem_node,
        "EVALC:0": evalc_node,
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

    graph = ProgramGraph(nodes=nodes, edges=edges)
    meta = {"parents": [], "budget_steps": int(assignment.get(BUDGET_STEPS, BUDGET_STEPS_DOMAIN[0]))}
    program = LearningProgram(graph=graph, constraints={"no_network_tools": True}, meta=meta)
    graph.validate()
    return program


def extract_assignment(program: LearningProgram) -> Dict[str, Any]:
    """Extract discrete variable assignment from a LearningProgram."""
    g = program.graph
    arch = g.nodes["ARCH:0"].spec
    lrule = g.nodes["LRULE:0"].spec
    obj = g.nodes["OBJ:0"].spec
    curr = g.nodes["CURR:0"].spec
    mem = g.nodes["MEM:0"].spec

    primary_loss = obj["losses"][0]["kind"]
    entropy_bonus = any(c.get("kind") == "policy_entropy_bonus" for c in obj.get("constraints", []))

    mem_reader = "none"
    for m in arch["modules"]:
        if m["kind"] == "memory_reader":
            mem_reader = m["params"].get("kind", "attention")

    curr_curriculum = curr.get("curriculum", {}) if isinstance(curr, dict) else {}
    max_len_train = curr_curriculum.get("max_len_train", 32)

    return {
        "ARCH.type": arch["type"],
        "ARCH.core.kind": arch["modules"][1]["kind"],
        "ARCH.core.n_layers": arch["modules"][1]["params"].get("n_layers", arch["modules"][1]["params"].get("depth", 2)),
        "ARCH.core.width": arch["modules"][1]["params"].get("d_model", arch["modules"][1]["params"].get("hidden_dim", arch["modules"][1]["params"].get("width", 128))),
        "ARCH.core.n_heads": arch["modules"][1]["params"].get("n_heads", 2),
        "ARCH.reg.dropout": arch["regularization"].get("dropout", 0.0),
        "ARCH.memory_reader": mem_reader,
        "LRULE.type": lrule["type"],
        "LRULE.lr_bin": lrule["hyper"]["base_lr"],
        "LRULE.schedule.kind": lrule["schedule"]["kind"],
        "LRULE.grad.clip_norm": lrule["grad"].get("clip_norm", 0.0),
        "OBJ.type": obj["type"],
        "OBJ.primary": primary_loss,
        "OBJ.entropy_bonus": "on" if entropy_bonus else "off",
        "CURR.type": curr["type"],
        "CURR.signal": curr["update_rule"]["signal"],
        "CURR.exploration": curr["update_rule"]["exploration"],
        CURR_MAX_LEN_TRAIN: max_len_train,
        "MEM.type": mem["type"],
        "MEM.capacity_bin": mem.get("capacity", 0),
        "MEM.retrieval.kind": mem.get("retrieval", {}).get("kind", "knn"),
        BUDGET_STEPS: program.meta.get("budget_steps") if hasattr(program, "meta") and isinstance(program.meta, dict) else BUDGET_STEPS_DOMAIN[0],
    }
