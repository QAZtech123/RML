from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rml.eval.normalize import MetricNormalizationError, normalize_metrics


@dataclass
class ScoreVector:
    gen: float
    adapt: float
    eff: float
    stab: float
    rob: float
    simp: float
    gov: float
    extra: Dict[str, float]


@dataclass
class EvalReport:
    program_hash: str
    score: ScoreVector
    constraint_violations: List[str]
    diagnostics: Dict[str, Any]


DEFAULT_CONTRACT = {
    "version": "0.1",
    "headline": "generalization_score",
    "splits": ["train", "shift", "unseen"],
    "score": {"weights": {"train": 0.2, "shift": 0.4, "unseen": 0.4}, "metric": "accuracy", "direction": "maximize"},
    "penalties": [
        {"name": "instability", "value": 1.0, "when": {"any": ["diverged=true", "nan_inf=true"]}, "mode": "subtract"},
        {"name": "overfit_gap", "value": 0.2, "when": {"all": ["train.accuracy - unseen.accuracy > 0.10"]}, "mode": "subtract"},
        {
            "name": "compute_over_budget",
            "value": 0.3,
            "when": {"any": ["wall_time_s > budget.wall_time_s", "steps > budget.steps"]},
            "mode": "subtract",
        },
    ],
    "gates": [
        {"name": "no_shift_regression", "type": "regression", "metric": "shift.accuracy", "baseline_ref": "rolling_best_20", "max_drop": 0.01},
        {"name": "no_unseen_regression", "type": "regression", "metric": "unseen.accuracy", "baseline_ref": "rolling_best_20", "max_drop": 0.01},
        {"name": "stability_required", "type": "boolean", "expr": "diverged=false AND nan_inf=false"},
    ],
}


class SimpleEvaluator:
    """Contract-driven evaluator that computes generalization_score with penalties and gates."""

    def __init__(self, constraints: Dict[str, Any] | None = None, contract: Optional[Dict[str, Any]] = None, store=None, budget: Optional[Dict[str, Any]] = None, history_window: int = 20):
        self.constraints = constraints or {}
        self.contract = contract or DEFAULT_CONTRACT
        self.store = store
        self.history_window = history_window
        self.budget = budget or {}
        self._recent = deque(maxlen=history_window)
        self.warmup_batches = int(self.contract.get("warmup_batches", 10))

    def _get_baseline(self, metric_key: str) -> Optional[float]:
        if self.store:
            try:
                val = self.store.get_rolling_best(metric_key=metric_key, window=self.history_window)
                if val is not None:
                    return float(val)
            except Exception:
                pass
        vals = [r.get(metric_key) for r in self._recent if r.get(metric_key) is not None]
        return max(vals) if vals else None

    def evaluate(self, run_result: Dict[str, Any]) -> EvalReport:
        program_hash = run_result.get("program_hash", "") if isinstance(run_result, dict) else ""
        violations: List[str] = []
        try:
            nm = normalize_metrics(run_result)
            failed_normalization = False
        except MetricNormalizationError as ex:
            nm = {
                "train_accuracy": 0.0,
                "shift_accuracy": 0.0,
                "unseen_accuracy": 0.0,
                "train_loss": 1.0,
                "shift_loss": 1.0,
                "unseen_loss": 1.0,
                "diverged": True,
                "nan_inf": True,
                "wall_time_s": 0.0,
                "steps": 0,
            }
            failed_normalization = True
            violations.append(ex.code)

        train_acc = float(nm.get("train_accuracy", 0.0))
        shift_acc = float(nm.get("shift_accuracy", 0.0))
        unseen_acc = float(nm.get("unseen_accuracy", 0.0))
        train_loss = nm.get("train_loss")
        shift_loss = nm.get("shift_loss")
        unseen_loss = nm.get("unseen_loss")
        diverged = bool(nm.get("diverged", False))
        nan_inf = bool(nm.get("nan_inf", False))
        wall_time_s = float(nm.get("wall_time_s", 0.0))
        steps = int(nm.get("steps", 0))

        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if diverged or nan_inf:
            penalties.append({"name": "instability", "value": 1.0})
            penalty_total += 1.0
        if (train_acc - unseen_acc) > 0.10:
            penalties.append({"name": "overfit_gap", "value": 0.2})
            penalty_total += 0.2
        if self.budget:
            over_budget = False
            if self.budget.get("wall_time_s") is not None and wall_time_s > self.budget["wall_time_s"]:
                over_budget = True
            if self.budget.get("steps") is not None and steps > self.budget["steps"]:
                over_budget = True
            if over_budget:
                penalties.append({"name": "compute_over_budget", "value": 0.3})
                penalty_total += 0.3
        if failed_normalization:
            penalties.append({"name": "normalization_failed", "value": 0.5})
            penalty_total += 0.5

        w = self.contract.get("score", {}).get("weights", {"train": 0.2, "shift": 0.4, "unseen": 0.4})
        base = (w.get("train", 0) * train_acc) + (w.get("shift", 0) * shift_acc) + (w.get("unseen", 0) * unseen_acc)
        generalization_score = base - penalty_total

        gates_failed: List[str] = []
        gates_skipped: List[str] = []

        if diverged or nan_inf:
            gates_failed.append("stability_required")

        enforce_regression = len(self._recent) >= self.warmup_batches

        if "stability_required" not in gates_failed:
            baseline_shift = self._get_baseline("best_shift_accuracy")
            baseline_unseen = self._get_baseline("best_unseen_accuracy")
            max_drop = 0.01
            if baseline_shift is None:
                gates_skipped.append("no_shift_regression")
            else:
                regressed = shift_acc < (baseline_shift - max_drop)
                if regressed and enforce_regression:
                    gates_failed.append("no_shift_regression")
                elif regressed:
                    gates_skipped.append("no_shift_regression")
            if baseline_unseen is None:
                gates_skipped.append("no_unseen_regression")
            else:
                regressed = unseen_acc < (baseline_unseen - max_drop)
                if regressed and enforce_regression:
                    gates_failed.append("no_unseen_regression")
                elif regressed:
                    gates_skipped.append("no_unseen_regression")

        passed = len(gates_failed) == 0

        # populate ScoreVector for compatibility; gen=unseen_acc, rob=shift_acc, stab from stability
        score_vec = ScoreVector(
            gen=unseen_acc,
            adapt=0.0,
            eff=0.0,
            stab=0.0 if (diverged or nan_inf) else 1.0,
            rob=shift_acc,
            simp=0.0,
            gov=-penalty_total,
            extra={
                "scalar": generalization_score,
                "generalization_score": generalization_score,
                "train_accuracy": train_acc,
                "shift_accuracy": shift_acc,
                "unseen_accuracy": unseen_acc,
            },
        )

        diagnostics: Dict[str, Any] = {
            "generalization_score": generalization_score,
            "base_score": base,
            "penalties_applied": penalties,
            "penalty_total": penalty_total,
            "gates_failed": gates_failed,
            "gates_skipped": gates_skipped,
            "passed": passed,
            "split_metrics": {
                "train": {"accuracy": train_acc, "loss": train_loss},
                "shift": {"accuracy": shift_acc, "loss": shift_loss},
                "unseen": {"accuracy": unseen_acc, "loss": unseen_loss},
            },
            "compute": {"wall_time_s": wall_time_s, "steps": steps},
            "stability": {"diverged": diverged, "nan_inf": nan_inf},
        }

        return EvalReport(program_hash=program_hash, score=score_vec, constraint_violations=violations, diagnostics=diagnostics)

    def update_batch_history(self, batch_meta: Dict[str, Any]) -> None:
        entry = {
            "best_shift_accuracy": batch_meta.get("best_shift_accuracy"),
            "best_unseen_accuracy": batch_meta.get("best_unseen_accuracy"),
            "best_generalization_score": batch_meta.get("best_generalization_score"),
        }
        self._recent.append(entry)
