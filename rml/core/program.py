from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

NodeId = str
Edge = Tuple[NodeId, NodeId, str]  # (src, dst, relation)


class ProgramValidationError(ValueError):
    """Structured validation failure for ProgramGraph and LearningProgram."""

    def __init__(self, code: str, msg: str, details: Optional[Dict[str, Any]] = None, legacy_code: Optional[str] = None):
        canonical = CODE_MAP.get(code, code)
        super().__init__(f"[{canonical}] {msg}")
        self.code = canonical
        self.legacy_code = legacy_code or (code if canonical != code else None)
        parts = canonical.split("_")
        self.layer = parts[0] if parts else "UNKNOWN"
        self.category = parts[1] if len(parts) > 1 else None
        self.details = details or {}


ALLOWED_KINDS = {"ARCH", "LRULE", "OBJ", "CURR", "MEM", "TOOLS", "EVALC"}
ALLOWED_RELATIONS = {
    "depends_on",
    "compatible_with",
    "incompatible_with",
    "shares_state_with",
    "routes_through",
}
REQUIRED_SINGLETONS = {"ARCH", "LRULE", "OBJ", "CURR", "MEM", "EVALC"}

CODE_MAP = {
    "EMPTY_GRAPH": "GRAPH_EMPTY",
    "NODE_ID_MISMATCH": "GRAPH_NODE_ID_MISMATCH",
    "BAD_KIND": "NODE_BAD_KIND",
    "BAD_SPEC": "NODE_BAD_SPEC",
    "SINGLETON_VIOLATION": "GRAPH_SINGLETON_VIOLATION",
    "ARCH_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "ARCH_BAD_TYPE": "NODE_SCHEMA_BAD_TYPE",
    "ARCH_BAD_IO": "NODE_SCHEMA_BAD_IO",
    "ARCH_BAD_MODULES": "NODE_SCHEMA_BAD_MODULES",
    "ARCH_DUPLICATE_MODULE": "NODE_SCHEMA_DUPLICATE_MODULE",
    "ARCH_BAD_MODULE_KIND": "NODE_SCHEMA_BAD_MODULE_KIND",
    "ARCH_NO_HEAD": "NODE_SCHEMA_NO_HEAD",
    "ARCH_BAD_WIRING": "NODE_SCHEMA_BAD_WIRING",
    "ARCH_WIRING_UNKNOWN_MODULE": "NODE_SCHEMA_WIRING_UNKNOWN_MODULE",
    "ARCH_DISCONNECTED": "NODE_SCHEMA_DISCONNECTED",
    "ARCH_BAD_INIT": "NODE_SCHEMA_BAD_INIT",
    "ARCH_BAD_REG": "NODE_SCHEMA_BAD_REG",
    "LRULE_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "LRULE_BAD_TYPE": "NODE_SCHEMA_BAD_TYPE",
    "LRULE_BAD_HYPER": "NODE_SCHEMA_BAD_HYPER",
    "LRULE_BAD_SCHEDULE": "NODE_SCHEMA_BAD_SCHEDULE",
    "LRULE_BAD_GRAD": "NODE_SCHEMA_BAD_GRAD",
    "LRULE_BAD_META": "NODE_SCHEMA_BAD_META",
    "LRULE_MISSING_LEARNED": "NODE_SCHEMA_MISSING_LEARNED",
    "LRULE_BAD_LEARNED": "NODE_SCHEMA_BAD_LEARNED",
    "OBJ_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "OBJ_BAD_TYPE": "NODE_SCHEMA_BAD_TYPE",
    "OBJ_BAD_LOSSES": "NODE_SCHEMA_BAD_LOSSES",
    "OBJ_DUP_LOSS": "NODE_SCHEMA_DUP_LOSS",
    "OBJ_BAD_WEIGHTS": "NODE_SCHEMA_BAD_WEIGHTS",
    "OBJ_BAD_CONSTRAINTS": "NODE_SCHEMA_BAD_CONSTRAINTS",
    "OBJ_BAD_TARGETS": "NODE_SCHEMA_BAD_TARGETS",
    "CURR_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "CURR_BAD_TYPE": "NODE_SCHEMA_BAD_TYPE",
    "CURR_BAD_WEIGHTS": "NODE_SCHEMA_BAD_WEIGHTS",
    "CURR_BAD_STATE": "NODE_SCHEMA_BAD_STATE",
    "CURR_BAD_UPDATE_RULE": "NODE_SCHEMA_BAD_UPDATE_RULE",
    "MEM_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "MEM_BAD_TYPE": "NODE_SCHEMA_BAD_TYPE",
    "MEM_BAD_CAPACITY": "NODE_SCHEMA_BAD_CAPACITY",
    "MEM_BAD_ENCODING": "NODE_SCHEMA_BAD_ENCODING",
    "MEM_BAD_RETRIEVAL": "NODE_SCHEMA_BAD_RETRIEVAL",
    "MEM_BAD_WRITE_POLICY": "NODE_SCHEMA_BAD_WRITE_POLICY",
    "MEM_BAD_DECAY": "NODE_SCHEMA_BAD_DECAY",
    "TOOLS_BAD_SPEC": "NODE_SCHEMA_BAD_TOOLS",
    "EVALC_MISSING_FIELD": "NODE_SCHEMA_MISSING_FIELD",
    "EVALC_BAD_BUDGETS": "NODE_SCHEMA_BAD_BUDGETS",
    "EVALC_BAD_PROTOCOL": "NODE_SCHEMA_BAD_PROTOCOL",
    "EVALC_BAD_METRICS": "NODE_SCHEMA_BAD_METRICS",
    "EVALC_BAD_TRACES": "NODE_SCHEMA_BAD_TRACES",
    "EVALC_BAD_DETERMINISM": "NODE_SCHEMA_BAD_DETERMINISM",
    "EDGE_UNKNOWN_NODE": "EDGE_MISSING_NODE",
    "EDGE_BAD_RELATION": "EDGE_UNKNOWN_RELATION",
    "EDGE_SELF_LOOP": "EDGE_SELF_LOOP",
    "EDGE_DUPLICATE": "EDGE_DUPLICATE",
    "EDGE_CONTRADICTION": "EDGE_CONTRADICTION",
    "EDGE_REQUIRED_MISSING": "EDGE_REQUIRED_MISSING",
    "COMPAT_ARCH_OBJ": "COMPAT_POLICY_HEAD_REQUIRED",
    "COMPAT_MEM_ARCH": "COMPAT_MEMORY_READER_REQUIRED",
    "COMPAT_LRULE_ARCH": "COMPAT_LEARNED_OPTIMIZER_REQUIRES_PARAMS",
    "COMPAT_EVALC_LRULE": "COMPAT_SCHEDULE_EXCEEDS_BUDGET",
    "EVALC_SCHEDULE_MISMATCH": "EVALC_SCHEDULE_MISMATCH",
    "EVALC_BAD_FIELD": "NODE_SCHEMA_BAD_FIELD",
    "DISCONNECTED_NODE": "GRAPH_DISCONNECTED_NODE",
    "NONFINITE_FLOAT": "HASH_NONFINITE_FLOAT",
    "NONJSON_TYPE": "HASH_NONJSON_TYPE",
    "NODE_EXISTS": "EDIT_NODE_EXISTS",
    "NO_SUCH_NODE": "EDIT_NO_SUCH_NODE",
    "CANNOT_REMOVE_SINGLETON": "EDIT_CANNOT_REMOVE_SINGLETON",
    "NO_SUCH_EDGE": "EDIT_NO_SUCH_EDGE",
    "UNKNOWN_EDIT": "EDIT_UNKNOWN_KIND",
}

ARCH_TYPES = {"transformer", "gnn", "mlp", "hybrid"}
ARCH_INPUT_SPACES = {"vector", "token", "graph"}
ARCH_OUTPUT_SPACES = {"vector", "token", "scalar", "action"}
ARCH_INIT_SCHEMES = {"xavier", "kaiming", "orthogonal"}
ARCH_MODULE_KINDS = {
    "embedding",
    "transformer_block_stack",
    "gnn_stack",
    "mlp_stack",
    "memory_reader",
    "mlp_head",
    "policy_head",
    "value_head",
}

LRULE_TYPES = {"adam", "sgd", "rmsprop", "learned_optimizer"}
LRULE_SCHEDULES = {"constant", "cosine", "linear_warmup_cosine", "piecewise"}

OBJ_TYPES = {"supervised", "self_supervised", "rl", "multi"}
LOSS_KINDS = {"cross_entropy", "mse", "policy_gradient", "contrastive", "next_token", "reconstruction"}
LOSS_COMBINE = {"weighted_sum", "pareto_scalarization"}
CONSTRAINT_KINDS = {"weight_decay", "policy_entropy_bonus"}
TARGET_LABEL_SPACES = {"class", "real", "action"}

CURR_TYPES = {"uniform", "adaptive", "bandit", "self_paced"}
CURR_MEMORY = {"none", "ema", "replay"}
CURR_EXPLORATION = {"epsilon_greedy", "ucb", "softmax"}
CURR_SIGNALS = {"loss", "reward", "gen_gap", "learning_progress"}

MEM_TYPES = {"none", "replay", "episodic_kv", "vector_db"}
RETRIEVAL_KINDS = {"knn", "attention"}
WRITE_KINDS = {"always", "surprise", "reward"}
DECAY_KINDS = {"none", "ttl", "ema"}

TOOLS_ALLOWLIST: set[str] = set()


@dataclass(frozen=True)
class ProgramNode:
    id: NodeId
    kind: str
    spec: Dict[str, Any]


@dataclass
class ProgramGraph:
    nodes: Dict[NodeId, ProgramNode]
    edges: List[Edge]

    def validate(self) -> None:
        if not self.nodes:
            raise ProgramValidationError("EMPTY_GRAPH", "No nodes in ProgramGraph")
        for nid, node in self.nodes.items():
            if node.id != nid:
                raise ProgramValidationError("NODE_ID_MISMATCH", f"Node id mismatch: {nid} vs {node.id}")
            if node.kind not in ALLOWED_KINDS:
                raise ProgramValidationError("BAD_KIND", f"Unknown node kind: {node.kind}", {"node_id": nid})
            if not isinstance(node.spec, dict):
                raise ProgramValidationError("BAD_SPEC", "Spec must be dict", {"node_id": nid})

        by_kind: Dict[str, List[str]] = {}
        for node in self.nodes.values():
            by_kind.setdefault(node.kind, []).append(node.id)
        for kind in REQUIRED_SINGLETONS:
            ids = by_kind.get(kind, [])
            if len(ids) != 1:
                raise ProgramValidationError(
                    "SINGLETON_VIOLATION",
                    f"Expected exactly one {kind}, found {len(ids)}",
                    {"kind": kind, "ids": ids},
                )
        if len(by_kind.get("TOOLS", [])) > 1:
            raise ProgramValidationError(
                "SINGLETON_VIOLATION", f"Expected at most one TOOLS, found {len(by_kind['TOOLS'])}", {"ids": by_kind["TOOLS"]}
            )

        self._validate_arch(self._get_single("ARCH"))
        self._validate_lrule(self._get_single("LRULE"))
        self._validate_obj(self._get_single("OBJ"))
        self._validate_curr(self._get_single("CURR"))
        self._validate_mem(self._get_single("MEM"))
        self._validate_evalc(self._get_single("EVALC"))
        tools = self._get_optional("TOOLS")
        if tools:
            self._validate_tools(tools)

        self._validate_edges()
        self._validate_compatibility()
        self._validate_eval_contract()
        self._validate_basic_connectivity()

    def _get_single(self, kind: str) -> ProgramNode:
        for node in self.nodes.values():
            if node.kind == kind:
                return node
        raise ProgramValidationError("MISSING_NODE", f"No node of kind {kind}")

    def _get_optional(self, kind: str) -> Optional[ProgramNode]:
        for node in self.nodes.values():
            if node.kind == kind:
                return node
        return None

    def subgraph(self, kinds: List[str]) -> "ProgramGraph":
        sub_nodes = {nid: node for nid, node in self.nodes.items() if node.kind in kinds}
        sub_edges = [(s, d, r) for (s, d, r) in self.edges if s in sub_nodes and d in sub_nodes]
        return ProgramGraph(nodes=sub_nodes, edges=sub_edges)

    def clone(self) -> "ProgramGraph":
        return ProgramGraph(nodes=copy.deepcopy(self.nodes), edges=list(self.edges))
    def _validate_arch(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("type", "io", "modules", "wiring", "init", "regularization"):
            if key not in spec:
                raise ProgramValidationError("ARCH_MISSING_FIELD", f"ARCH missing {key}")
        if spec["type"] not in ARCH_TYPES:
            raise ProgramValidationError("ARCH_BAD_TYPE", f"ARCH.type must be one of {sorted(ARCH_TYPES)}")

        io = spec["io"]
        for key in ("input_space", "input_dim", "output_space", "output_dim"):
            if key not in io:
                raise ProgramValidationError("ARCH_BAD_IO", f"ARCH.io missing {key}")
        if io["input_space"] not in ARCH_INPUT_SPACES:
            raise ProgramValidationError("ARCH_BAD_IO", "Unsupported input_space")
        if io["output_space"] not in ARCH_OUTPUT_SPACES:
            raise ProgramValidationError("ARCH_BAD_IO", "Unsupported output_space")
        if not _is_positive_int(io["input_dim"]) or not _is_positive_int(io["output_dim"]):
            raise ProgramValidationError("ARCH_BAD_IO", "input_dim/output_dim must be positive integers")

        modules = spec["modules"]
        if not isinstance(modules, list) or not modules:
            raise ProgramValidationError("ARCH_BAD_MODULES", "ARCH.modules must be non-empty list")
        names_seen: set[str] = set()
        head_present = False
        memory_reader_present = False
        for mod in modules:
            if not isinstance(mod, dict):
                raise ProgramValidationError("ARCH_BAD_MODULES", "Module entries must be dicts")
            for key in ("name", "kind", "params"):
                if key not in mod:
                    raise ProgramValidationError("ARCH_BAD_MODULES", f"Module missing {key}")
            name = mod["name"]
            if not isinstance(name, str) or not name:
                raise ProgramValidationError("ARCH_BAD_MODULES", "Module name must be non-empty string")
            if name in names_seen:
                raise ProgramValidationError("ARCH_DUPLICATE_MODULE", f"Duplicate module name: {name}")
            names_seen.add(name)
            kind = mod["kind"]
            if kind not in ARCH_MODULE_KINDS:
                raise ProgramValidationError("ARCH_BAD_MODULE_KIND", f"Unsupported module kind: {kind}")
            if not isinstance(mod["params"], dict):
                raise ProgramValidationError("ARCH_BAD_MODULES", f"Module params must be dict for {name}")
            head_present = head_present or kind in {"mlp_head", "policy_head", "value_head"}
            memory_reader_present = memory_reader_present or kind == "memory_reader"

        if not head_present:
            raise ProgramValidationError("ARCH_NO_HEAD", "ARCH must include at least one head module")

        wiring = spec["wiring"]
        if not isinstance(wiring, list) or not wiring:
            raise ProgramValidationError("ARCH_BAD_WIRING", "ARCH.wiring must be non-empty list")
        for conn in wiring:
            if not isinstance(conn, dict):
                raise ProgramValidationError("ARCH_BAD_WIRING", "Wiring entries must be dicts")
            if "src" not in conn or "dst" not in conn:
                raise ProgramValidationError("ARCH_BAD_WIRING", "Wiring entries require src and dst")
            if conn["src"] not in names_seen or conn["dst"] not in names_seen:
                raise ProgramValidationError(
                    "ARCH_WIRING_UNKNOWN_MODULE", f"Wiring references unknown module: {conn}"
                )

        outputs = {m["name"] for m in modules if m["kind"] in {"mlp_head", "policy_head", "value_head"}}
        incoming: Dict[str, int] = {name: 0 for name in names_seen}
        adj: Dict[str, List[str]] = {name: [] for name in names_seen}
        for conn in wiring:
            adj[conn["src"]].append(conn["dst"])
            incoming[conn["dst"]] = incoming.get(conn["dst"], 0) + 1
        start_nodes = {n for n, deg in incoming.items() if deg == 0} or {next(iter(names_seen))}
        reachable: set[str] = set()
        for start in start_nodes:
            _dfs(start, adj, reachable)
        if not (outputs & reachable):
            raise ProgramValidationError("ARCH_DISCONNECTED", "No path from input modules to head modules")

        init = spec["init"]
        if init.get("scheme") not in ARCH_INIT_SCHEMES:
            raise ProgramValidationError("ARCH_BAD_INIT", "Unknown init.scheme")
        if not isinstance(init.get("seeded"), bool):
            raise ProgramValidationError("ARCH_BAD_INIT", "init.seeded must be boolean")

        reg = spec["regularization"]
        if not isinstance(reg, dict):
            raise ProgramValidationError("ARCH_BAD_REG", "regularization must be dict")
        if "dropout" in reg and (not _is_finite_number(reg["dropout"]) or reg["dropout"] < 0):
            raise ProgramValidationError("ARCH_BAD_REG", "dropout must be non-negative number")
        if "weight_decay" in reg and (not _is_finite_number(reg["weight_decay"]) or reg["weight_decay"] < 0):
            raise ProgramValidationError("ARCH_BAD_REG", "weight_decay must be non-negative number")
        if "layer_norm" in reg and not isinstance(reg["layer_norm"], bool):
            raise ProgramValidationError("ARCH_BAD_REG", "layer_norm must be boolean")

    def _validate_lrule(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("type", "hyper", "schedule", "grad", "meta"):
            if key not in spec:
                raise ProgramValidationError("LRULE_MISSING_FIELD", f"LRULE missing {key}")
        if spec["type"] not in LRULE_TYPES:
            raise ProgramValidationError("LRULE_BAD_TYPE", f"LRULE.type must be one of {sorted(LRULE_TYPES)}")

        hyper = spec["hyper"]
        if not isinstance(hyper, dict):
            raise ProgramValidationError("LRULE_BAD_HYPER", "hyper must be dict")
        base_lr = hyper.get("base_lr")
        if not _is_positive_number(base_lr):
            raise ProgramValidationError("LRULE_BAD_HYPER", "base_lr must be positive")
        for key in ("beta1", "beta2", "eps", "momentum"):
            if key in hyper and not _is_finite_number(hyper[key]):
                raise ProgramValidationError("LRULE_BAD_HYPER", f"{key} must be finite if provided")

        schedule = spec["schedule"]
        if schedule.get("kind") not in LRULE_SCHEDULES:
            raise ProgramValidationError("LRULE_BAD_SCHEDULE", "Unknown schedule.kind")
        for key in ("warmup_steps", "max_steps"):
            if not isinstance(schedule.get(key), int) or schedule[key] < 0:
                raise ProgramValidationError("LRULE_BAD_SCHEDULE", f"{key} must be non-negative int")
        if schedule.get("max_steps", 0) <= 0:
            raise ProgramValidationError("LRULE_BAD_SCHEDULE", "max_steps must be positive")
        if "min_lr" in schedule and not _is_finite_number(schedule["min_lr"]):
            raise ProgramValidationError("LRULE_BAD_SCHEDULE", "min_lr must be finite number")

        grad = spec["grad"]
        if not isinstance(grad, dict):
            raise ProgramValidationError("LRULE_BAD_GRAD", "grad must be dict")
        for key in ("clip_norm", "clip_value", "noise_std"):
            if key in grad and grad[key] is not None:
                if not _is_finite_number(grad[key]) or grad[key] < 0:
                    raise ProgramValidationError("LRULE_BAD_GRAD", f"{key} must be non-negative number")
        if "normalize" in grad and not isinstance(grad["normalize"], bool):
            raise ProgramValidationError("LRULE_BAD_GRAD", "normalize must be boolean")

        meta = spec["meta"]
        if not isinstance(meta, dict):
            raise ProgramValidationError("LRULE_BAD_META", "meta must be dict")
        for key in ("supports_hyper_updates", "supports_rule_edits"):
            if key in meta and not isinstance(meta[key], bool):
                raise ProgramValidationError("LRULE_BAD_META", f"{key} must be boolean")

        if spec["type"] == "learned_optimizer":
            if "learned" not in spec:
                raise ProgramValidationError("LRULE_MISSING_LEARNED", "learned_optimizer requires 'learned' field")
            learned = spec["learned"]
            if not isinstance(learned, dict):
                raise ProgramValidationError("LRULE_BAD_LEARNED", "learned must be dict")
            update_model = learned.get("update_model", {})
            if update_model.get("arch") not in {"mlp", "rnn", "transformer"}:
                raise ProgramValidationError("LRULE_BAD_LEARNED", "update_model.arch must be mlp|rnn|transformer")
            for key in ("hidden_dim", "depth", "state_dim"):
                if not _is_positive_int(update_model.get(key)):
                    raise ProgramValidationError("LRULE_BAD_LEARNED", f"update_model.{key} must be positive int")
            inputs = learned.get("inputs")
            outputs = learned.get("outputs")
            if not isinstance(inputs, list) or not inputs:
                raise ProgramValidationError("LRULE_BAD_LEARNED", "learned.inputs must be non-empty list")
            if not isinstance(outputs, list) or not outputs:
                raise ProgramValidationError("LRULE_BAD_LEARNED", "learned.outputs must be non-empty list")
            for flag in ("per_parameter", "shared_across_layers"):
                if flag in learned and not isinstance(learned[flag], bool):
                    raise ProgramValidationError("LRULE_BAD_LEARNED", f"learned.{flag} must be boolean")

    def _validate_obj(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("type", "losses", "weights", "constraints", "targets"):
            if key not in spec:
                raise ProgramValidationError("OBJ_MISSING_FIELD", f"OBJ missing {key}")
        if spec["type"] not in OBJ_TYPES:
            raise ProgramValidationError("OBJ_BAD_TYPE", f"OBJ.type must be one of {sorted(OBJ_TYPES)}")
        losses = spec["losses"]
        if not isinstance(losses, list) or not losses:
            raise ProgramValidationError("OBJ_BAD_LOSSES", "losses must be non-empty list")
        names = set()
        for loss in losses:
            if not isinstance(loss, dict):
                raise ProgramValidationError("OBJ_BAD_LOSSES", "loss entries must be dicts")
            for key in ("name", "kind"):
                if key not in loss:
                    raise ProgramValidationError("OBJ_BAD_LOSSES", f"loss missing {key}")
            if loss["kind"] not in LOSS_KINDS:
                raise ProgramValidationError("OBJ_BAD_LOSSES", f"Unsupported loss kind: {loss['kind']}")
            if loss["name"] in names:
                raise ProgramValidationError("OBJ_DUP_LOSS", f"Duplicate loss name: {loss['name']}")
            names.add(loss["name"])

        weights = spec["weights"]
        if weights.get("combine") not in LOSS_COMBINE:
            raise ProgramValidationError("OBJ_BAD_WEIGHTS", "weights.combine invalid")
        values = weights.get("values", {})
        if not isinstance(values, dict):
            raise ProgramValidationError("OBJ_BAD_WEIGHTS", "weights.values must be dict")
        if set(values.keys()) != names:
            raise ProgramValidationError(
                "OBJ_BAD_WEIGHTS",
                "weights.values keys must match loss names",
                {"losses": sorted(names), "weights": sorted(values.keys())},
            )
        if not any(v > 0 for v in values.values()):
            raise ProgramValidationError("OBJ_BAD_WEIGHTS", "At least one weight must be positive")
        if any(v < 0 for v in values.values()):
            raise ProgramValidationError("OBJ_BAD_WEIGHTS", "weights must be non-negative")

        constraints = spec.get("constraints", [])
        if not isinstance(constraints, list):
            raise ProgramValidationError("OBJ_BAD_CONSTRAINTS", "constraints must be list")
        for c in constraints:
            if not isinstance(c, dict):
                raise ProgramValidationError("OBJ_BAD_CONSTRAINTS", "constraint entries must be dicts")
            if "kind" not in c or c["kind"] not in CONSTRAINT_KINDS:
                raise ProgramValidationError("OBJ_BAD_CONSTRAINTS", f"Unknown constraint kind: {c.get('kind')}")
            if "strength" in c and (not _is_finite_number(c["strength"]) or c["strength"] < 0):
                raise ProgramValidationError("OBJ_BAD_CONSTRAINTS", "constraint strength must be non-negative number")

        targets = spec["targets"]
        if "head" not in targets or not isinstance(targets["head"], str):
            raise ProgramValidationError("OBJ_BAD_TARGETS", "targets.head required")
        if targets.get("label_space") not in TARGET_LABEL_SPACES:
            raise ProgramValidationError("OBJ_BAD_TARGETS", "targets.label_space invalid")
    def _validate_curr(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("type", "family_weights", "state", "update_rule"):
            if key not in spec:
                raise ProgramValidationError("CURR_MISSING_FIELD", f"CURR missing {key}")
        if spec["type"] not in CURR_TYPES:
            raise ProgramValidationError("CURR_BAD_TYPE", "CURR.type invalid")
        family_weights = spec["family_weights"]
        if not isinstance(family_weights, dict) or not family_weights:
            raise ProgramValidationError("CURR_BAD_WEIGHTS", "family_weights must be non-empty dict")
        if not any(v > 0 for v in family_weights.values()):
            raise ProgramValidationError("CURR_BAD_WEIGHTS", "At least one family weight must be positive")
        if any(v < 0 for v in family_weights.values()):
            raise ProgramValidationError("CURR_BAD_WEIGHTS", "family weights must be non-negative")

        state = spec["state"]
        if not isinstance(state, dict):
            raise ProgramValidationError("CURR_BAD_STATE", "state must be dict")
        if state.get("memory") not in CURR_MEMORY:
            raise ProgramValidationError("CURR_BAD_STATE", "state.memory invalid")
        if state.get("memory") == "ema":
            if not _is_finite_number(state.get("ema_beta")) or not (0 <= state["ema_beta"] <= 1):
                raise ProgramValidationError("CURR_BAD_STATE", "ema_beta must be in [0,1]")
        if "window" in state and (not _is_positive_int(state["window"])):
            raise ProgramValidationError("CURR_BAD_STATE", "window must be positive int if provided")

        update_rule = spec["update_rule"]
        if spec["type"] in {"bandit", "adaptive", "self_paced"} and not update_rule:
            raise ProgramValidationError("CURR_BAD_UPDATE_RULE", "update_rule required for adaptive curricula")
        if update_rule:
            if update_rule.get("signal") not in CURR_SIGNALS:
                raise ProgramValidationError("CURR_BAD_UPDATE_RULE", "signal invalid")
            if update_rule.get("exploration") not in CURR_EXPLORATION:
                raise ProgramValidationError("CURR_BAD_UPDATE_RULE", "exploration invalid")
            if "temperature" in update_rule and not _is_finite_number(update_rule["temperature"]):
                raise ProgramValidationError("CURR_BAD_UPDATE_RULE", "temperature must be finite number")
            if "epsilon" in update_rule and not _is_finite_number(update_rule["epsilon"]):
                raise ProgramValidationError("CURR_BAD_UPDATE_RULE", "epsilon must be finite number")

    def _validate_mem(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("type", "capacity", "encoding", "retrieval", "write_policy", "decay"):
            if key not in spec:
                raise ProgramValidationError("MEM_MISSING_FIELD", f"MEM missing {key}")
        if spec["type"] not in MEM_TYPES:
            raise ProgramValidationError("MEM_BAD_TYPE", "MEM.type invalid")
        if spec["type"] == "none":
            return
        if not _is_positive_int(spec.get("capacity", 0)):
            raise ProgramValidationError("MEM_BAD_CAPACITY", "capacity must be positive int")
        encoding = spec["encoding"]
        if not isinstance(encoding, dict):
            raise ProgramValidationError("MEM_BAD_ENCODING", "encoding must be dict")
        if not _is_positive_int(encoding.get("key_dim")) or not _is_positive_int(encoding.get("value_dim")):
            raise ProgramValidationError("MEM_BAD_ENCODING", "key_dim/value_dim must be positive int")
        retrieval = spec["retrieval"]
        if retrieval.get("kind") not in RETRIEVAL_KINDS:
            raise ProgramValidationError("MEM_BAD_RETRIEVAL", "retrieval.kind invalid")
        if retrieval["kind"] == "knn":
            if not _is_positive_int(retrieval.get("k")):
                raise ProgramValidationError("MEM_BAD_RETRIEVAL", "retrieval.k must be positive int")
        if "temperature" in retrieval and not _is_finite_number(retrieval["temperature"]):
            raise ProgramValidationError("MEM_BAD_RETRIEVAL", "retrieval.temperature must be finite")
        write_policy = spec["write_policy"]
        if write_policy.get("kind") not in WRITE_KINDS:
            raise ProgramValidationError("MEM_BAD_WRITE_POLICY", "write_policy.kind invalid")
        if "threshold" in write_policy and not _is_finite_number(write_policy["threshold"]):
            raise ProgramValidationError("MEM_BAD_WRITE_POLICY", "write_policy.threshold must be finite")
        decay = spec["decay"]
        if decay.get("kind") not in DECAY_KINDS:
            raise ProgramValidationError("MEM_BAD_DECAY", "decay.kind invalid")
        if decay["kind"] == "ttl":
            if not _is_positive_int(decay.get("ttl_steps")):
                raise ProgramValidationError("MEM_BAD_DECAY", "ttl_steps must be positive int")
        if decay["kind"] == "ema":
            if not _is_finite_number(decay.get("ema_beta")) or not (0 <= decay["ema_beta"] <= 1):
                raise ProgramValidationError("MEM_BAD_DECAY", "ema_beta must be in [0,1]")

    def _validate_tools(self, node: ProgramNode) -> None:
        spec = node.spec
        if "enabled" not in spec or not isinstance(spec["enabled"], bool):
            raise ProgramValidationError("TOOLS_BAD_SPEC", "TOOLS.enabled must be boolean")
        if spec["enabled"]:
            allowed = spec.get("allowed", [])
            if not isinstance(allowed, list):
                raise ProgramValidationError("TOOLS_BAD_SPEC", "TOOLS.allowed must be list")
            unknown = [t for t in allowed if t not in TOOLS_ALLOWLIST]
            if unknown:
                raise ProgramValidationError("TOOLS_BAD_SPEC", "TOOLS.allowed contains unknown tools", {"unknown": unknown})
            limits = spec.get("limits", {})
            if not isinstance(limits, dict):
                raise ProgramValidationError("TOOLS_BAD_SPEC", "TOOLS.limits must be dict")
            for key in ("calls_per_episode", "time_budget_ms"):
                if key in limits and (not _is_positive_int(limits[key])):
                    raise ProgramValidationError("TOOLS_BAD_SPEC", f"{key} must be positive int")

    def _validate_evalc(self, node: ProgramNode) -> None:
        spec = node.spec
        for key in ("budgets", "protocol", "metrics_required", "traces_required", "determinism"):
            if key not in spec:
                raise ProgramValidationError("EVALC_MISSING_FIELD", f"EVALC missing {key}")
        budgets = spec["budgets"]
        if not isinstance(budgets, dict):
            raise ProgramValidationError("EVALC_BAD_BUDGETS", "budgets must be dict")
        inner = budgets.get("inner", {})
        if not isinstance(inner, dict):
            raise ProgramValidationError("EVALC_BAD_BUDGETS", "budgets.inner must be dict")
        if not _is_positive_int(inner.get("max_steps", 0)):
            raise ProgramValidationError("EVALC_BAD_BUDGETS", "max_steps must be positive int")
        if not _is_positive_number(inner.get("max_seconds", 0.0)):
            raise ProgramValidationError("EVALC_BAD_BUDGETS", "max_seconds must be positive")
        if not _is_positive_int(inner.get("max_memory_mb", 0)):
            raise ProgramValidationError("EVALC_BAD_BUDGETS", "max_memory_mb must be positive int")

        protocol = spec["protocol"]
        if not isinstance(protocol, dict):
            raise ProgramValidationError("EVALC_BAD_PROTOCOL", "protocol must be dict")
        for key in ("train_tasks", "shift_tasks", "unseen_tasks"):
            if not isinstance(protocol.get(key), int) or protocol[key] < 0:
                raise ProgramValidationError("EVALC_BAD_PROTOCOL", f"{key} must be non-negative int")
        if protocol.get("train_tasks", 0) + protocol.get("shift_tasks", 0) + protocol.get("unseen_tasks", 0) <= 0:
            raise ProgramValidationError("EVALC_BAD_PROTOCOL", "At least one protocol task count must be positive")

        metrics = spec["metrics_required"]
        if not isinstance(metrics, list) or not metrics:
            raise ProgramValidationError("EVALC_BAD_METRICS", "metrics_required must be non-empty list")
        traces = spec["traces_required"]
        if not isinstance(traces, list) or not traces:
            raise ProgramValidationError("EVALC_BAD_TRACES", "traces_required must be non-empty list")

        det = spec["determinism"]
        if not isinstance(det, dict):
            raise ProgramValidationError("EVALC_BAD_DETERMINISM", "determinism must be dict")
        if "seeded" not in det or "replayable" not in det:
            raise ProgramValidationError("EVALC_BAD_DETERMINISM", "determinism must include seeded and replayable")
        if not isinstance(det["seeded"], bool) or not isinstance(det["replayable"], bool):
            raise ProgramValidationError("EVALC_BAD_DETERMINISM", "determinism flags must be boolean")
    def _validate_edges(self) -> None:
        seen: set[Tuple[str, str, str]] = set()
        for src, dst, rel in self.edges:
            if src not in self.nodes or dst not in self.nodes:
                raise ProgramValidationError("EDGE_UNKNOWN_NODE", f"Edge references unknown node: {src}->{dst}")
            if rel not in ALLOWED_RELATIONS:
                raise ProgramValidationError("EDGE_BAD_RELATION", f"Unknown edge relation: {rel}")
            if src == dst and rel != "shares_state_with":
                raise ProgramValidationError("EDGE_SELF_LOOP", f"Self-loop only allowed for shares_state_with: {src}")
            key = (src, dst, rel)
            if key in seen:
                raise ProgramValidationError("EDGE_DUPLICATE", f"Duplicate edge: {src}->{dst} {rel}")
            seen.add(key)

        compat_pairs: set[Tuple[str, str]] = set()
        incompatible_pairs: set[Tuple[str, str]] = set()
        for src, dst, rel in self.edges:
            pair = (src, dst)
            if rel == "compatible_with":
                compat_pairs.add(pair)
            elif rel == "incompatible_with":
                incompatible_pairs.add(pair)
        for s, d in list(compat_pairs):
            if (s, d) in incompatible_pairs or (d, s) in incompatible_pairs:
                raise ProgramValidationError("EDGE_CONTRADICTION", f"Compatible and incompatible edges conflict: {s}<->{d}")

        required_edges = [
            ("ARCH", "OBJ", "depends_on"),
            ("LRULE", "ARCH", "compatible_with"),
            ("CURR", "OBJ", "compatible_with"),
            ("MEM", "ARCH", "compatible_with"),
            ("EVALC", "ARCH", "depends_on"),
            ("EVALC", "OBJ", "depends_on"),
            ("EVALC", "CURR", "depends_on"),
        ]
        for src_kind, dst_kind, rel in required_edges:
            src_id = self._get_single(src_kind).id
            dst_id = self._get_single(dst_kind).id
            if (src_id, dst_id, rel) not in seen:
                raise ProgramValidationError(
                    "EDGE_REQUIRED_MISSING",
                    f"Missing required edge {src_kind}->{dst_kind} ({rel})",
                )

    def _validate_compatibility(self) -> None:
        arch = self._get_single("ARCH")
        obj = self._get_single("OBJ")
        lrule = self._get_single("LRULE")
        mem = self._get_single("MEM")
        evalc = self._get_single("EVALC")

        has_policy_loss = any(loss.get("kind") == "policy_gradient" for loss in obj.spec["losses"])
        has_policy_head = any(m.get("kind") == "policy_head" for m in arch.spec.get("modules", []))
        if has_policy_loss and not has_policy_head:
            raise ProgramValidationError("COMPAT_ARCH_OBJ", "policy_gradient loss requires policy_head in ARCH")

        label_space = obj.spec.get("targets", {}).get("label_space")
        if label_space == "class":
            output_dim = arch.spec.get("io", {}).get("output_dim")
            if not _is_positive_int(output_dim) or output_dim < 2:
                raise ProgramValidationError("COMPAT_ARCH_OBJ", "Classification requires output_dim >= 2")

        mem_type = mem.spec.get("type")
        has_memory_reader = any(m.get("kind") == "memory_reader" for m in arch.spec.get("modules", []))
        if mem_type != "none" and not has_memory_reader:
            raise ProgramValidationError("COMPAT_MEM_ARCH", "Memory enabled but ARCH lacks memory_reader module")
        if mem_type == "none" and has_memory_reader:
            raise ProgramValidationError("COMPAT_MEM_ARCH", "ARCH has memory_reader but MEM.type is none")

        if lrule.spec["type"] == "learned_optimizer":
            learned = lrule.spec.get("learned", {})
            if learned.get("per_parameter", False):
                modules = arch.spec.get("modules", [])
                if not modules:
                    raise ProgramValidationError("COMPAT_LRULE_ARCH", "per_parameter optimizer requires modules present")

        sched_max = lrule.spec.get("schedule", {}).get("max_steps", 0)
        budget_max = evalc.spec.get("budgets", {}).get("inner", {}).get("max_steps", 0)
        if sched_max and budget_max and sched_max > budget_max:
            raise ProgramValidationError(
                "COMPAT_EVALC_LRULE",
                "LRULE schedule.max_steps exceeds EVALC.budgets.inner.max_steps",
            )

    def _validate_eval_contract(self) -> None:
        evalc = self._get_single("EVALC")
        lrule = self._get_single("LRULE")
        sched_max = lrule.spec.get("schedule", {}).get("max_steps", 0)
        budget_max = evalc.spec.get("budgets", {}).get("inner", {}).get("max_steps", 0)
        if sched_max and budget_max and sched_max > budget_max:
            raise ProgramValidationError("EVALC_SCHEDULE_MISMATCH", "LRULE schedule exceeds EVALC budget")
        for field in ("metrics_required", "traces_required"):
            arr = evalc.spec.get(field, [])
            if not all(isinstance(x, str) for x in arr):
                raise ProgramValidationError("EVALC_BAD_FIELD", f"{field} must contain strings")

    def _validate_basic_connectivity(self) -> None:
        connected_nodes: set[str] = set()
        for src, dst, _ in self.edges:
            connected_nodes.add(src)
            connected_nodes.add(dst)
        missing = [nid for nid in self.nodes if nid not in connected_nodes]
        if missing:
            raise ProgramValidationError("DISCONNECTED_NODE", f"Nodes have no edges: {missing}")

    def hash(self) -> str:
        return hash_graph(self)


@dataclass
class LearningProgram:
    graph: ProgramGraph
    constraints: Dict[str, Any]
    meta: Dict[str, Any]

    def hash(self) -> str:
        return self.graph.hash()

    def clone(self) -> "LearningProgram":
        return LearningProgram(
            graph=self.graph.clone(),
            constraints=copy.deepcopy(self.constraints),
            meta=copy.deepcopy(self.meta),
        )


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ProgramValidationError("NONFINITE_FLOAT", "NaN/Inf not allowed in specs")
        return obj
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    raise ProgramValidationError("NONJSON_TYPE", f"Non-JSON type in spec: {type(obj)}")


def to_canonical_json_bytes(graph: ProgramGraph) -> bytes:
    nodes_sorted = []
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        nodes_sorted.append({"id": node.id, "kind": node.kind, "spec": _canonicalize(node.spec)})

    edges_sorted = sorted(
        [{"src": s, "dst": d, "rel": r} for (s, d, r) in graph.edges],
        key=lambda e: (e["src"], e["dst"], e["rel"]),
    )

    payload = {"nodes": nodes_sorted, "edges": edges_sorted}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def hash_graph(graph: ProgramGraph) -> str:
    b = to_canonical_json_bytes(graph)
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class PathStep:
    key: str
    index: Optional[int] = None


_PATH_RE = re.compile(r"^([^\.]+)\.(spec|constraints|meta)(?:\.(.+))?$")


def parse_path(path: str) -> Tuple[str, str, List[PathStep]]:
    m = _PATH_RE.match(path)
    if not m:
        raise ValueError(f"Invalid path: {path}")
    node_id = m.group(1)
    root = m.group(2)
    rest = m.group(3)

    steps: List[PathStep] = []
    if rest:
        for part in rest.split("."):
            if "[" in part:
                if not part.endswith("]"):
                    raise ValueError(f"Invalid list syntax in path segment: {part}")
                key, idx_part = part.split("[", 1)
                idx_str = idx_part[:-1]
                if not idx_str.isdigit():
                    raise ValueError(f"Invalid list index in path segment: {part}")
                steps.append(PathStep(key, int(idx_str)))
            else:
                steps.append(PathStep(part, None))
    return node_id, root, steps


def get_by_path(program: LearningProgram, path: str) -> Any:
    node_id, root, steps = parse_path(path)
    base: Union[Dict[str, Any], Any]
    if root == "spec":
        base = program.graph.nodes[node_id].spec
    elif root == "constraints":
        base = program.constraints
    elif root == "meta":
        base = program.meta
    else:
        raise ValueError("Invalid path root")

    cur: Any = base
    for step in steps:
        if not isinstance(cur, dict):
            raise KeyError(f"Expected dict before key {step.key} in {path}")
        cur = cur[step.key]
        if step.index is not None:
            if not isinstance(cur, list):
                raise KeyError(f"Expected list at {step.key} in {path}")
            cur = cur[step.index]
    if isinstance(cur, (dict, list)):
        return copy.deepcopy(cur)
    return cur


def set_by_path(program: LearningProgram, path: str, value: Any, create_missing_dict_keys: bool = False) -> None:
    node_id, root, steps = parse_path(path)
    if root == "spec":
        base = program.graph.nodes[node_id].spec
    elif root == "constraints":
        base = program.constraints
    elif root == "meta":
        base = program.meta
    else:
        raise ValueError("Invalid path root")

    if not steps:
        raise ValueError("Cannot set root object directly")

    cur: Any = base
    for step in steps[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"Expected dict before {step.key} in {path}")
        if step.key not in cur:
            if create_missing_dict_keys:
                cur[step.key] = [] if step.index is not None else {}
            else:
                raise KeyError(f"Missing key {step.key} in {path}")
        cur = cur[step.key]
        if step.index is not None:
            if not isinstance(cur, list):
                raise KeyError(f"Expected list at {step.key} in {path}")
            if step.index < 0 or step.index >= len(cur):
                raise IndexError(f"Index out of range at {step.key}[{step.index}] in {path}")
            cur = cur[step.index]

    last = steps[-1]
    if not isinstance(cur, dict):
        raise KeyError(f"Expected dict before final key {last.key} in {path}")
    if last.key not in cur and not create_missing_dict_keys:
        raise KeyError(f"Missing final key {last.key} in {path}")

    if last.index is None:
        cur[last.key] = value
    else:
        arr = cur.get(last.key)
        if not isinstance(arr, list):
            raise KeyError(f"Expected list at final {last.key} in {path}")
        if last.index < 0 or last.index >= len(arr):
            raise IndexError(f"Index out of range at {last.key}[{last.index}] in {path}")
        arr[last.index] = value


def apply_edits_atomic(program: LearningProgram, edits: Iterable["GraphEdit"]) -> Tuple[LearningProgram, List[str]]:
    new_program = copy.deepcopy(program)
    applied: List[str] = []
    for edit in edits:
        _apply_single_edit(new_program, edit)
        applied.append(edit.meta.get("id", edit.kind) if isinstance(edit.meta, dict) else edit.kind)
    new_program.graph.validate()
    return new_program, applied


def try_apply_edits(program: LearningProgram, edits: Iterable["GraphEdit"]) -> Tuple[LearningProgram, Dict[str, Any]]:
    try:
        new_program, applied = apply_edits_atomic(program, edits)
        return new_program, {"ok": True, "applied": applied}
    except ProgramValidationError as ex:
        return program, {
            "ok": False,
            "error_code": ex.code,
            "legacy_code": ex.legacy_code,
            "error": str(ex),
            "details": ex.details,
        }


def _apply_single_edit(program: LearningProgram, edit: "GraphEdit") -> None:
    graph = program.graph
    if edit.kind == "tune_param":
        set_by_path(program, edit.target, edit.value, create_missing_dict_keys=False)
    elif edit.kind == "add_node":
        node_dict = edit.value
        nid = node_dict["id"]
        if nid in graph.nodes:
            raise ProgramValidationError("NODE_EXISTS", f"Node already exists: {nid}")
        graph.nodes[nid] = ProgramNode(id=nid, kind=node_dict["kind"], spec=node_dict["spec"])
    elif edit.kind == "remove_node":
        nid = edit.target
        if nid not in graph.nodes:
            raise ProgramValidationError("NO_SUCH_NODE", f"Cannot remove missing node: {nid}")
        kind = graph.nodes[nid].kind
        if kind in REQUIRED_SINGLETONS:
            raise ProgramValidationError("CANNOT_REMOVE_SINGLETON", f"Cannot remove required node kind: {kind}")
        del graph.nodes[nid]
        graph.edges = [(s, d, r) for (s, d, r) in graph.edges if s != nid and d != nid]
    elif edit.kind == "add_edge":
        val = edit.value
        graph.edges.append((val["src"], val["dst"], val["rel"]))
    elif edit.kind == "remove_edge":
        val = edit.value
        before = len(graph.edges)
        graph.edges = [(s, d, r) for (s, d, r) in graph.edges if not (s == val["src"] and d == val["dst"] and r == val["rel"])]
        if len(graph.edges) == before:
            raise ProgramValidationError("NO_SUCH_EDGE", f"Edge not found: {val}")
    elif edit.kind == "rewire":
        val = edit.value or {}
        for rem in val.get("remove", []):
            _apply_single_edit(program, GraphEdit(kind="remove_edge", target="", value=rem, meta={}))
        for add in val.get("add", []):
            _apply_single_edit(program, GraphEdit(kind="add_edge", target="", value=add, meta={}))
    else:
        raise ProgramValidationError("UNKNOWN_EDIT", f"Unknown edit kind: {edit.kind}")


def _is_positive_int(v: Any) -> bool:
    return isinstance(v, int) and v > 0


def _is_positive_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and v > 0 and _is_finite_number(v)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (v != v or v in (float("inf"), float("-inf")))


def _dfs(start: str, adj: Dict[str, List[str]], visited: set[str]) -> None:
    if start in visited:
        return
    visited.add(start)
    for nxt in adj.get(start, []):
        _dfs(nxt, adj, visited)


try:
    from rml.updater.meta_updater import GraphEdit  # type: ignore  # noqa: E402
except Exception:
    GraphEdit = Any  # type: ignore
