from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rml.core.engine import RMLEngine
from rml.core.factor_graph_dist import FactorGraphDistribution
from rml.core.progress import SelfImprovementTracker
from rml.core.variables import extract_assignment
from rml.eval.simple_evaluator import SimpleEvaluator
from rml.storage.artifact_store import ArtifactStore
from rml.storage.sqlite_store import SQLiteStore
from rml.updater.meta_updater import GraphEdit, UpdateAction, MetaUpdater
from rml.real_runner import RealRunner
from rml.task_family_a import FamilyATaskFamily


@dataclass
class AppConfig:
    db_path: Path
    artifact_root: Path
    runner_version: str = "dev"
    runner_kind: str = "baseline"  # baseline | real
    cache_scope: str = "step"
    taskset_mode: str = "resample"
    taskset_resample_prob: float = 0.1
    obj_prior: dict | None = None
    dist_temperature: float = 1.0
    dist_uniform_mix: float = 0.02
    dist_gibbs_sweeps: int = 3
    dist_lr: float = 0.25


class NoOpUpdater(MetaUpdater):
    def propose(self, batch) -> UpdateAction:
        return UpdateAction(program_edits={}, factor_updates={}, exploration={})

    def learn(self, history) -> None:
        return None

    def snapshot(self) -> Dict[str, Any]:
        return {}


class DummyTaskFamily:
    name = "synthetic"

    def sample_train(self, rng: int) -> Dict[str, Any]:
        return {"family": self.name, "split": "train", "seed": rng}

    def sample_shift(self, rng: int) -> Dict[str, Any]:
        return {"family": self.name, "split": "shift", "seed": rng}

    def sample_unseen(self, rng: int) -> Dict[str, Any]:
        return {"family": self.name, "split": "unseen", "seed": rng}

    def sample_transfer(self, rng: int) -> Dict[str, Any]:
        return {"family": self.name, "split": "transfer", "seed": rng}


class BaselineRunner:
    """Deterministic runner that produces plausible metrics without real training."""

    def run(self, program, task_specs: List[Any], budget: Any, rng: int, **kwargs) -> Dict[str, Any]:
        assign = extract_assignment(program)
        score = 0.5
        # Simple heuristics
        if assign.get("ARCH.type") == "transformer":
            score += 0.2
        if assign.get("LRULE.type") == "adam":
            score += 0.1
        if assign.get("LRULE.schedule.kind") == "cosine":
            score += 0.05
        width = assign.get("ARCH.core.width", 128)
        layers = assign.get("ARCH.core.n_layers", 2)
        compute_seconds = 0.01 * layers * (width / 128)
        shift_score = max(0.0, min(1.0, score - 0.05))
        unseen_score = max(0.0, min(1.0, score - 0.1))
        train_loss_final = max(0.0, 1.5 - score)
        trace = {"loss_curve": [1.5, 1.0, 0.7, train_loss_final]}
        # Demo-mode objective readiness adjustment: policy_gradient needs a policy head
        has_policy_head = any(m.get("kind") == "policy_head" for m in program.graph.nodes["ARCH:0"].spec.get("modules", []))
        obj_primary = assign.get("OBJ.primary")
        if obj_primary == "policy_gradient" and not has_policy_head:
            unseen_score -= 0.02
            shift_score -= 0.02
            train_loss_final += 0.02
        if obj_primary == "mse":
            unseen_score += 0.01
            shift_score += 0.01
            train_loss_final -= 0.01
        unseen_score = max(0.0, min(1.0, unseen_score))
        shift_score = max(0.0, min(1.0, shift_score))
        train_loss_final = max(0.0, train_loss_final)
        train_accuracy = max(0.0, min(1.0, score))
        # loss placeholders for shift/unseen
        shift_loss = max(0.0, train_loss_final + 0.02)
        unseen_loss = max(0.0, train_loss_final + 0.04)
        return {
            "program_hash": program.hash(),
            "metrics": {
                "train_loss_final": train_loss_final,
                "train_loss": train_loss_final,
                "train_accuracy": train_accuracy,
                "train_steps_to_threshold": int(50 / max(score, 0.1)),
                "shift_score": shift_score,
                "shift_accuracy": shift_score,
                "shift_loss": shift_loss,
                "unseen_score": unseen_score,
                "unseen_accuracy": unseen_score,
                "unseen_loss": unseen_loss,
                "transfer_accuracy": unseen_score,
                "transfer_loss": unseen_loss,
                "compute_seconds": compute_seconds,
                "wall_time_s": compute_seconds,
                "steps": int(max(1, layers * 10)),
                "diverged": False,
                "nan_inf": False,
            },
            "trace": trace,
        }


def build_engine(app_cfg: AppConfig) -> RMLEngine:
    store = SQLiteStore(app_cfg.db_path)
    artifact_store = ArtifactStore(app_cfg.artifact_root, db=store)
    progress = SelfImprovementTracker()
    dist = FactorGraphDistribution(
        temperature=app_cfg.dist_temperature,
        uniform_mix=app_cfg.dist_uniform_mix,
        gibbs_sweeps=app_cfg.dist_gibbs_sweeps,
        lr=app_cfg.dist_lr,
        max_retries=200,
        obj_prior=app_cfg.obj_prior,
    )
    if app_cfg.runner_kind == "real":
        runner = RealRunner()
    else:
        runner = BaselineRunner()
    eval_contract_spec = {
        "budgets": {"inner": {"max_steps": 2000, "max_seconds": 30.0, "max_memory_mb": 4096}},
        "protocol": {"train_tasks": 4, "shift_tasks": 2, "unseen_tasks": 2},
        "metrics_required": ["train_loss", "shift_score", "unseen_score", "compute_seconds", "steps_to_threshold"],
        "traces_required": ["loss_curve"],
        "determinism": {"seeded": True, "replayable": True},
    }
    budget_cfg = eval_contract_spec.get("budgets", {}).get("inner", {})
    evaluator = SimpleEvaluator(contract=eval_contract_spec, store=store, budget=budget_cfg)
    updater = NoOpUpdater()
    if app_cfg.runner_kind == "real":
        task_families = [FamilyATaskFamily()]
    else:
        task_families = [DummyTaskFamily()]

    return RMLEngine(
        dist=dist,
        runner=runner,
        evaluator=evaluator,
        updater=updater,
        task_families=task_families,
        store=store,
        artifact_store=artifact_store,
        eval_contract_spec=eval_contract_spec,
        progress_tracker=progress,
        runner_version=app_cfg.runner_version,
        cache_scope=app_cfg.cache_scope,
        taskset_mode=app_cfg.taskset_mode,
        taskset_resample_prob=app_cfg.taskset_resample_prob,
    )
