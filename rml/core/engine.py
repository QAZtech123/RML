from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import random
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from rml.core import ids
from rml.core.program import LearningProgram, to_canonical_json_bytes, get_by_path
from rml.core.progress import SelfImprovementTracker
from rml.core.quantum import QuantumSearch, QuantumState
from rml.core.run_context import RunContext
from rml.storage.artifact_record import ArtifactRecord
from rml.storage.artifact_store import ArtifactStore
from rml.storage.checkpoint_store import CheckpointRecord, CheckpointStore
from rml.storage.sqlite_store import SQLiteStore


class InnerRunner(Protocol):
    def run(
        self,
        program: LearningProgram,
        task_specs: List[Any],
        budget: Any,
        rng: int,
        init_state_dict: Optional[Dict[str, Any]] = None,
        init_source: str = "none",
        init_signature: Optional[str] = None,
    ) -> Any: ...


class Evaluator(Protocol):
    def evaluate(self, run_result: Any) -> Any: ...


@dataclass
class ExecuteRunResult:
    run_row: Dict[str, Any]
    artifacts: List[ArtifactRecord]
    cached: bool
    run_result: Any


ELITE_FAIL_PENALTY = 0.05
ENGINE_INSTRUMENTATION_VERSION = 5


@dataclass(frozen=True)
class OverrideGene:
    gene_id: str
    lam: float
    parent_tolerance: float
    transfer_floor: float
    tie_primary_min: float


@dataclass
class OverrideGeneEpisodeSummary:
    gene_id: str
    episode_id: int
    step0: int
    step1: int
    reward: float
    reward_transfer_component: float
    reward_general_component: float
    reliable: bool
    n_trusted: int
    n_events: int
    collapse_steps: int
    transfer_proxy_mean: Optional[float]
    general_proxy_rate: Optional[float]
    rollback: bool = False
    rollback_reason: str = ""


@dataclass
class OverrideGeneMaturedSummary:
    gene_id: str
    episode_id: int
    step0: int
    step1: int
    matured_step: int
    proxy_reward: float
    delayed_reward: Optional[float]
    reward_used: float
    delayed_reliable: bool
    reliable: bool
    n_trusted: int
    n_events: int
    collapse_steps: int
    primary_survival: Optional[float]
    transfer_survival: Optional[float]


def _default_override_genes() -> List[OverrideGene]:
    tick = 0.00390625
    return [
        OverrideGene("g00_default", lam=0.50, parent_tolerance=tick, transfer_floor=0.00, tie_primary_min=0.0500),
        OverrideGene("g01_transfer_tight", lam=0.60, parent_tolerance=tick, transfer_floor=tick, tie_primary_min=0.0500),
        OverrideGene("g02_parent_flex", lam=0.50, parent_tolerance=2.0 * tick, transfer_floor=0.00, tie_primary_min=0.0500),
        OverrideGene("g03_parent_transfer_balance", lam=0.55, parent_tolerance=2.0 * tick, transfer_floor=tick, tie_primary_min=0.0500),
        OverrideGene("g04_tie_strict", lam=0.50, parent_tolerance=tick, transfer_floor=0.00, tie_primary_min=0.0625),
        OverrideGene("g05_transfer_priority", lam=0.70, parent_tolerance=tick, transfer_floor=tick, tie_primary_min=0.0550),
        OverrideGene("g06_parent_loose_transfer_floor", lam=0.55, parent_tolerance=3.0 * tick, transfer_floor=tick, tie_primary_min=0.0500),
        OverrideGene("g07_conservative", lam=0.45, parent_tolerance=tick, transfer_floor=tick, tie_primary_min=0.0625),
    ]


class OverrideGeneBandit:
    def __init__(
        self,
        genes: List[OverrideGene],
        *,
        episode_len: int = 50,
        ucb_c: float = 0.5,
        transfer_target: float = 0.01,
        primary_target: float = 0.01,
        survival_window_steps: int = 10,
        reward_blend_proxy: float = 0.3,
        reward_blend_delayed: float = 0.7,
        reliable_n_trusted: int = 10,
        reliable_n_events: int = 3,
        low_reliability_shrink: float = 0.3,
    ):
        if not genes:
            raise ValueError("OverrideGeneBandit requires at least one gene")
        self.genes = list(genes)
        self.episode_len = max(1, int(episode_len))
        self.ucb_c = float(ucb_c)
        self.transfer_target = max(1e-6, float(transfer_target))
        self.primary_target = max(1e-6, float(primary_target))
        self.survival_window_steps = max(1, int(survival_window_steps))
        self.reward_blend_proxy = float(reward_blend_proxy)
        self.reward_blend_delayed = float(reward_blend_delayed)
        self.reliable_n_trusted = max(1, int(reliable_n_trusted))
        self.reliable_n_events = max(1, int(reliable_n_events))
        self.low_reliability_shrink = float(low_reliability_shrink)

        self.counts: Dict[str, int] = {g.gene_id: 0 for g in self.genes}
        self.means: Dict[str, float] = {g.gene_id: 0.0 for g in self.genes}
        self.total_episodes = 0

        self._active_gene: Optional[OverrideGene] = None
        self._active_episode_id = -1
        self._active_episode_start = 0
        self._active_selection: Dict[str, Any] = {}
        self._pending: List[Dict[str, Any]] = []
        self._reset_accum()

    def _reset_accum(self) -> None:
        self._accum_trusted = 0
        self._accum_events = 0
        self._accum_transfer_n = 0
        self._accum_transfer_sum = 0.0
        self._accum_collapse_steps = 0
        self._accum_last_primary: Optional[float] = None
        self._accum_last_transfer: Optional[float] = None

    @staticmethod
    def _clip_unit(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def _update_arm(self, gene_id: str, reward: float) -> None:
        n_prev = self.counts.get(gene_id, 0)
        n_new = n_prev + 1
        mean_prev = self.means.get(gene_id, 0.0)
        mean_new = mean_prev + ((reward - mean_prev) / float(n_new))
        self.counts[gene_id] = n_new
        self.means[gene_id] = float(mean_new)
        self.total_episodes += 1

    def _select_gene(self) -> Tuple[OverrideGene, Dict[str, Any]]:
        # Deterministic warm start: try each unplayed gene once in declared order.
        for gene in self.genes:
            if self.counts.get(gene.gene_id, 0) == 0:
                return gene, {
                    "selected_by": "ucb1",
                    "ucb_score": None,
                    "mean_reward_before": self.means.get(gene.gene_id, 0.0),
                    "n_before": 0,
                }

        best_gene = self.genes[0]
        best_score = -float("inf")
        total_n = max(1, self.total_episodes)
        for gene in self.genes:
            gid = gene.gene_id
            n = max(1, self.counts.get(gid, 0))
            mean = self.means.get(gid, 0.0)
            score = mean + (self.ucb_c * math.sqrt(math.log(total_n + 1.0) / n))
            if score > best_score:
                best_score = score
                best_gene = gene
        return best_gene, {
            "selected_by": "ucb1",
            "ucb_score": float(best_score),
            "mean_reward_before": self.means.get(best_gene.gene_id, 0.0),
            "n_before": self.counts.get(best_gene.gene_id, 0),
        }

    def _finalize_active_episode(self, end_step: int) -> Optional[OverrideGeneEpisodeSummary]:
        if self._active_gene is None:
            return None
        transfer_mean = (
            (self._accum_transfer_sum / self._accum_transfer_n)
            if self._accum_transfer_n > 0
            else None
        )
        general_rate = (
            (self._accum_events / self._accum_trusted)
            if self._accum_trusted > 0
            else None
        )
        transfer_component = 0.0 if transfer_mean is None else self._clip_unit(transfer_mean / self.transfer_target)
        general_component = 0.0 if general_rate is None else ((general_rate - 0.5) * 2.0)
        proxy_reward = (0.7 * transfer_component) + (0.3 * general_component)
        reliable = (
            self._accum_trusted >= self.reliable_n_trusted
            and self._accum_events >= self.reliable_n_events
        )
        if not reliable:
            proxy_reward *= self.low_reliability_shrink
        if self._accum_collapse_steps > 0 and proxy_reward < 0:
            proxy_reward -= min(0.20, 0.02 * float(self._accum_collapse_steps))

        gid = self._active_gene.gene_id
        self._pending.append(
            {
                "gene_id": gid,
                "episode_id": self._active_episode_id,
                "step0": self._active_episode_start,
                "step1": end_step,
                "mature_step": end_step + self.survival_window_steps,
                "proxy_reward": float(proxy_reward),
                "proxy_general_component": float(general_component),
                "reliable": bool(reliable),
                "n_trusted": int(self._accum_trusted),
                "n_events": int(self._accum_events),
                "collapse_steps": int(self._accum_collapse_steps),
                "baseline_primary": self._accum_last_primary,
                "baseline_transfer": self._accum_last_transfer,
                "transfer_proxy_mean": transfer_mean,
                "general_proxy_rate": general_rate,
            }
        )

        return OverrideGeneEpisodeSummary(
            gene_id=gid,
            episode_id=self._active_episode_id,
            step0=self._active_episode_start,
            step1=end_step,
            reward=float(proxy_reward),
            reward_transfer_component=float(transfer_component),
            reward_general_component=float(general_component),
            reliable=bool(reliable),
            n_trusted=int(self._accum_trusted),
            n_events=int(self._accum_events),
            collapse_steps=int(self._accum_collapse_steps),
            transfer_proxy_mean=transfer_mean,
            general_proxy_rate=general_rate,
            rollback=False,
            rollback_reason="",
        )

    def _mature_pending(
        self,
        *,
        step_idx: int,
        current_primary: Optional[float],
        current_transfer: Optional[float],
    ) -> List[OverrideGeneMaturedSummary]:
        matured: List[OverrideGeneMaturedSummary] = []
        while self._pending and int(self._pending[0].get("mature_step", 0)) <= int(step_idx):
            pend = self._pending.pop(0)
            baseline_primary = pend.get("baseline_primary")
            baseline_transfer = pend.get("baseline_transfer")
            primary_survival = None
            transfer_survival = None
            if current_primary is not None and baseline_primary is not None:
                primary_survival = float(current_primary) - float(baseline_primary)
            if current_transfer is not None and baseline_transfer is not None:
                transfer_survival = float(current_transfer) - float(baseline_transfer)

            delayed_reliable = (primary_survival is not None) and (transfer_survival is not None)
            delayed_reward = None
            if delayed_reliable:
                transfer_term = self._clip_unit(float(transfer_survival) / self.transfer_target)
                primary_term = self._clip_unit(float(primary_survival) / self.primary_target)
                proxy_general_component = float(pend.get("proxy_general_component", 0.0))
                delayed_reward = (0.7 * transfer_term) + (0.2 * primary_term) + (0.1 * proxy_general_component)

            proxy_reward = float(pend.get("proxy_reward", 0.0))
            if delayed_reward is None:
                reward_used = proxy_reward
            else:
                reward_used = (
                    (self.reward_blend_proxy * proxy_reward)
                    + (self.reward_blend_delayed * float(delayed_reward))
                )

            reliable = bool(pend.get("reliable", False)) and bool(delayed_reliable)
            if not reliable:
                reward_used *= self.low_reliability_shrink
            collapse_steps = int(pend.get("collapse_steps", 0))
            if collapse_steps > 0 and reward_used < 0:
                reward_used -= min(0.20, 0.02 * float(collapse_steps))

            gene_id = str(pend.get("gene_id") or "")
            self._update_arm(gene_id, float(reward_used))

            matured.append(
                OverrideGeneMaturedSummary(
                    gene_id=gene_id,
                    episode_id=int(pend.get("episode_id", -1)),
                    step0=int(pend.get("step0", 0)),
                    step1=int(pend.get("step1", 0)),
                    matured_step=int(step_idx),
                    proxy_reward=proxy_reward,
                    delayed_reward=(float(delayed_reward) if delayed_reward is not None else None),
                    reward_used=float(reward_used),
                    delayed_reliable=bool(delayed_reliable),
                    reliable=bool(reliable),
                    n_trusted=int(pend.get("n_trusted", 0)),
                    n_events=int(pend.get("n_events", 0)),
                    collapse_steps=collapse_steps,
                    primary_survival=primary_survival,
                    transfer_survival=transfer_survival,
                )
            )
        return matured

    def begin_step(self, step_idx: int) -> Dict[str, Any]:
        finalized: Optional[OverrideGeneEpisodeSummary] = None
        if (
            self._active_gene is not None
            and step_idx > self._active_episode_start
            and step_idx % self.episode_len == 0
        ):
            finalized = self._finalize_active_episode(step_idx - 1)
            self._active_gene = None

        if self._active_gene is None:
            gene, selection = self._select_gene()
            self._active_episode_id += 1
            self._active_episode_start = step_idx
            self._active_gene = gene
            self._active_selection = selection
            self._reset_accum()

        return {
            "gene": self._active_gene,
            "episode_id": self._active_episode_id,
            "episode_start": self._active_episode_start,
            "selection": dict(self._active_selection),
            "finalized": finalized,
        }

    def observe_step(
        self,
        *,
        step_idx: int,
        trusted_override: bool,
        transfer_delta: Optional[float],
        mean_delta: Optional[float],
        block_reason: str,
        collapse_step: bool,
        current_primary: Optional[float],
        current_transfer: Optional[float],
    ) -> List[OverrideGeneMaturedSummary]:
        if current_primary is not None:
            try:
                self._accum_last_primary = float(current_primary)
            except Exception:
                self._accum_last_primary = self._accum_last_primary
        if current_transfer is not None:
            try:
                self._accum_last_transfer = float(current_transfer)
            except Exception:
                self._accum_last_transfer = self._accum_last_transfer

        if collapse_step:
            self._accum_collapse_steps += 1
        if trusted_override:
            self._accum_trusted += 1
            if transfer_delta is not None:
                self._accum_transfer_sum += float(transfer_delta)
                self._accum_transfer_n += 1
            allowed = block_reason in {"override_allowed_strict", "override_allowed_rescue"}
            if (
                allowed
                and (mean_delta is not None and float(mean_delta) > 0.0)
                and (transfer_delta is not None and float(transfer_delta) > 0.0)
            ):
                self._accum_events += 1
        return self._mature_pending(
            step_idx=step_idx,
            current_primary=current_primary,
            current_transfer=current_transfer,
        )


def execute_run_with_cache_and_artifacts(
    *,
    store: SQLiteStore,
    artifact_store: ArtifactStore,
    runner: InnerRunner,
    run_ctx: RunContext,
    program: LearningProgram,
    task_specs: List[Any],
    budget: Any,
    eval_contract: Dict[str, Any],
    rng: int,
    allow_version_mismatch: bool = False,
    runner_init_state_dict: Optional[Dict[str, Any]] = None,
    runner_init_source: str = "none",
    runner_init_signature: Optional[str] = None,
) -> ExecuteRunResult:
    """
    Executes a run with cache reuse, artifacts-first writing, and DB transaction commit.
    """
    hit = store.maybe_get_cached_run(
        run_ctx.run_id,
        require_runner_version=run_ctx.runner_version,
        allow_version_mismatch=allow_version_mismatch,
    )
    if hit is not None:
        return ExecuteRunResult(run_row=hit.loaded_run, artifacts=hit.artifacts, cached=True, run_result=hit.loaded_run)

    run_result = runner.run(
        program=program,
        task_specs=task_specs,
        budget=budget,
        rng=rng,
        init_state_dict=runner_init_state_dict,
        init_source=runner_init_source,
        init_signature=runner_init_signature,
    )

    metrics = {}
    trace = {}
    trace_summary = {}
    if isinstance(run_result, dict):
        metrics = run_result.get("metrics", {})
        trace = run_result.get("trace", {})
        trace_summary = run_result.get("trace_summary", {})
    else:
        metrics = getattr(run_result, "metrics", {}) or {}
        trace = getattr(run_result, "trace", {}) or {}
        trace_summary = getattr(run_result, "trace_summary", {}) or {}

    artifact_records: List[ArtifactRecord] = []
    if trace:
        artifact_records.append(
            artifact_store.save_json(
                run_id=run_ctx.run_id,
                kind="trace",
                name="trace_full",
                obj=trace,
                gzip_compress=True,
                register=False,
            )
        )

    run_row = {
        "run_id": run_ctx.run_id,
        "program_id": run_ctx.program_id,
        "taskset_id": run_ctx.taskset_id,
        "budget_id": run_ctx.budget_id,
        "eval_contract_id": run_ctx.eval_contract_id,
        "seed": run_ctx.seed,
        "runner_version": run_ctx.runner_version,
        "engine_step": run_ctx.engine_step,
        "task_specs": task_specs,
        "budget": budget,
        "eval_contract": eval_contract,
        "metrics": metrics,
        "traces": trace_summary,
        "artifacts": {rec.name: rec.relpath for rec in artifact_records},
        "status": "ok",
    }

    with store.transaction() as conn:
        store.upsert_program(
            program_id=run_ctx.program_id,
            graph_canonical_json=to_canonical_json_bytes(program.graph).decode("utf-8"),
            parents=program.meta.get("parents") if hasattr(program, "meta") else None,
            meta=program.meta if hasattr(program, "meta") else None,
            conn=conn,
        )
        store.insert_run(run_row, conn=conn)
        for rec in artifact_records:
            store.insert_artifact(rec, conn=conn)

    return ExecuteRunResult(run_row=run_row, artifacts=artifact_records, cached=False, run_result=run_result)


def _episode_score(ep: ProgramEpisode) -> float:
    diag = getattr(ep.eval_report, "diagnostics", {}) or {}
    if "generalization_score" in diag:
        return float(diag["generalization_score"])
    score = getattr(ep.eval_report.score, "extra", {}).get("scalar") if hasattr(ep.eval_report, "score") else None
    return float(score) if score is not None else 0.0


def select_elites_gate_aware(episodes: List[ProgramEpisode], n_elite: int) -> Tuple[List[ProgramEpisode], List[ProgramEpisode], List[ProgramEpisode]]:
    passed_eps: List[ProgramEpisode] = []
    failed_eps: List[ProgramEpisode] = []
    for ep in episodes:
        diag = getattr(ep.eval_report, "diagnostics", {}) or {}
        passed_flag = diag.get("passed")
        if passed_flag is True:
            passed_eps.append(ep)
        else:
            failed_eps.append(ep)

    def score(ep: ProgramEpisode) -> float:
        return _episode_score(ep)

    if len(passed_eps) >= n_elite:
        elites = sorted(passed_eps, key=score, reverse=True)[:n_elite]
    else:
        elites = sorted(passed_eps, key=score, reverse=True)
        remaining = n_elite - len(elites)
        failed_sorted = sorted(failed_eps, key=lambda ep: score(ep) - ELITE_FAIL_PENALTY, reverse=True)
        elites += failed_sorted[:remaining]
    return elites, passed_eps, failed_eps


def _safe_get_by_path(program: LearningProgram, path: str, default: Any = None) -> Any:
    try:
        return get_by_path(program, path)
    except Exception:
        return default


def _spec_to_dict(spec: Any) -> Dict[str, Any]:
    if is_dataclass(spec):
        return asdict(spec)
    if isinstance(spec, dict):
        return dict(spec)
    if hasattr(spec, "__dict__"):
        return dict(spec.__dict__)
    return {"value": spec}


def _bucket_int(value: Optional[int], step: int) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value) / float(step)) * step)
    except Exception:
        return None


def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_unseen_set(task_specs: List[Any]) -> Optional[str]:
    unseen_specs: List[str] = []
    for ts in task_specs:
        split = None
        if isinstance(ts, dict):
            split = ts.get("split")
        else:
            split = getattr(ts, "split", None)
        if split != "unseen":
            continue
        spec_dict = _spec_to_dict(ts)
        unseen_specs.append(json.dumps(spec_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    if not unseen_specs:
        return None
    unseen_specs.sort()
    payload = json.dumps(unseen_specs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_transfer_set(task_specs: List[Any]) -> Optional[str]:
    transfer_specs: List[str] = []
    for ts in task_specs:
        split = None
        if isinstance(ts, dict):
            split = ts.get("split")
        else:
            split = getattr(ts, "split", None)
        if split != "transfer":
            continue
        spec_dict = _spec_to_dict(ts)
        transfer_specs.append(json.dumps(spec_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    if not transfer_specs:
        return None
    transfer_specs.sort()
    payload = json.dumps(transfer_specs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EngineConfig:
    programs_per_step: int
    budget: Any
    train_tasks: int = 1
    shift_tasks: int = 0
    unseen_tasks: int = 0
    transfer_tasks: int = 0
    total_steps: Optional[int] = None
    anneal_start: float = 0.15
    anneal_end: float = 0.005
    rescue_enable: bool = False
    rescue_no_parent_rate: float = 0.66
    rescue_best_floor: float = 0.12
    rescue_inject_n: int = 1
    rescue_max_per_run: int = 10
    rescue_max_per_episode: int = 2
    rescue_low_split_n: int = 8


@dataclass
class ProgramEpisode:
    program: LearningProgram
    run_results: List[Any]
    eval_report: Any
    cached: bool = False


@dataclass
class MetaBatch:
    episodes: List[ProgramEpisode]
    dist_snapshot: Dict[str, Any]
    step: int
    rng: int
    meta: Optional[Dict[str, Any]] = None


class RMLEngine:
    def __init__(
        self,
        dist,
        runner: InnerRunner,
        evaluator: Evaluator,
        updater,
        task_families: List[Any],
        store: SQLiteStore,
        artifact_store: ArtifactStore,
        eval_contract_spec: Dict[str, Any],
        progress_tracker: Optional[SelfImprovementTracker] = None,
        runner_version: str = "dev",
        cache_scope: str = "step",
        taskset_mode: str = "resample",
        taskset_resample_prob: float = 0.1,
    ):
        self.dist = dist
        self.runner = runner
        self.evaluator = evaluator
        self.updater = updater
        self.task_families = task_families
        self.store = store
        self.artifact_store = artifact_store
        self.eval_contract_spec = eval_contract_spec
        self.runner_version = runner_version
        self.cache_scope = cache_scope
        self.taskset_mode = taskset_mode
        self.taskset_resample_prob = taskset_resample_prob
        self._fixed_tasksets: Dict[int, List[Any]] = {}
        self.progress = progress_tracker or SelfImprovementTracker()
        self.q_state = QuantumState()
        arch_domain = []
        try:
            arch_domain = list(getattr(self.dist.unaries.get("ARCH.type"), "domain", []))  # type: ignore[attr-defined]
        except Exception:
            arch_domain = []
        self.q_search = QuantumSearch(arch_domain)
        self.checkpoints = CheckpointStore(self.artifact_store.artifact_root / "checkpoints")
        self._override_gene_bandit = OverrideGeneBandit(_default_override_genes())
        self._rescue_injection_total = 0
        self._rescue_injection_episode_counts: Dict[int, int] = {}

    def _split_rng(self, base: int, *extra: Any) -> int:
        return hash((base, *extra)) & 0xFFFFFFFF

    def _seed_for(self, base_seed: int, step_idx: int, i: int) -> int:
        if self.cache_scope == "step":
            return self._split_rng(base_seed, step_idx, i)
        return self._split_rng(base_seed, i)

    def _build_task_specs(
        self,
        cfg: EngineConfig,
        rng_step: int,
        idx: int,
        program: Optional[LearningProgram] = None,
        unseen_pool_seed: Optional[int] = None,
        transfer_seed: Optional[int] = None,
    ) -> List[Any]:
        if self.taskset_mode in {"fixed", "mixed"} and idx in self._fixed_tasksets:
            # In mixed mode, optionally resample with a small probability to keep novelty.
            if self.taskset_mode == "mixed":
                rng = np.random.default_rng(self._split_rng(rng_step, idx, "mixed"))
                if rng.random() < self.taskset_resample_prob:
                    self._fixed_tasksets.pop(idx, None)
                else:
                    return self._fixed_tasksets[idx]
            else:
                return self._fixed_tasksets[idx]

        specs: List[Any] = []
        for fam in self.task_families:
            for _ in range(cfg.train_tasks):
                try:
                    specs.append(fam.sample_train(self._split_rng(rng_step, idx, "train", len(specs)), program=program))
                except TypeError:
                    specs.append(fam.sample_train(self._split_rng(rng_step, idx, "train", len(specs))))
            for _ in range(cfg.shift_tasks):
                try:
                    specs.append(fam.sample_shift(self._split_rng(rng_step, idx, "shift", len(specs)), program=program))
                except TypeError:
                    specs.append(fam.sample_shift(self._split_rng(rng_step, idx, "shift", len(specs))))
            for _ in range(cfg.unseen_tasks):
                seed = self._split_rng(unseen_pool_seed, "unseen", len(specs)) if unseen_pool_seed is not None else self._split_rng(rng_step, idx, "unseen", len(specs))
                try:
                    specs.append(fam.sample_unseen(seed, program=program))
                except TypeError:
                    specs.append(fam.sample_unseen(seed))
            for _ in range(cfg.transfer_tasks):
                transfer_rng = (
                    self._split_rng(transfer_seed, "transfer", len(specs))
                    if transfer_seed is not None
                    else self._split_rng(rng_step, idx, "transfer", len(specs))
                )
                try:
                    if hasattr(fam, "sample_transfer"):
                        spec = fam.sample_transfer(transfer_rng, program=program)
                    else:
                        spec = fam.sample_unseen(transfer_rng, program=program)
                except TypeError:
                    if hasattr(fam, "sample_transfer"):
                        spec = fam.sample_transfer(transfer_rng)
                    else:
                        spec = fam.sample_unseen(transfer_rng)
                if isinstance(spec, dict):
                    spec = dict(spec)
                    spec["split"] = "transfer"
                specs.append(spec)

        if self.taskset_mode == "fixed":
            self._fixed_tasksets[idx] = specs
        elif self.taskset_mode == "mixed":
            # store the latest to be reused until resampled
            self._fixed_tasksets[idx] = specs
        return specs

    def step(self, cfg: EngineConfig, step_idx: int, rng: int) -> MetaBatch:
        # For cache scopes beyond "step", keep task sampling stable across steps to allow reuse
        rng_step = self._split_rng(rng, step_idx if self.cache_scope == "step" else 0)
        entropy_before = getattr(self.dist, "entropy", lambda: None)()
        unseen_pool_k = 5
        unseen_pool_idx = step_idx % unseen_pool_k
        unseen_pool_seed = self._split_rng(rng, "unseen_pool", unseen_pool_idx)
        transfer_seed = self._split_rng(rng, "transfer_fixed")
        total_steps = cfg.total_steps or 0
        if total_steps and total_steps > 1:
            progress = step_idx / max(1, total_steps - 1)
        else:
            progress = 0.0
        current_planck = float(cfg.anneal_start * (1.0 - progress) + cfg.anneal_end * progress)
        self.q_state.planck_h = current_planck
        gene_state = self._override_gene_bandit.begin_step(step_idx)
        active_override_gene: OverrideGene = gene_state["gene"]
        gene_selection_meta = gene_state.get("selection") or {}
        gene_finalized = gene_state.get("finalized")
        if hasattr(self.dist, "set_arch_bias"):
            try:
                self.dist.set_arch_bias(self.q_search.probabilities())
            except Exception:
                pass
        programs = self.dist.sample(cfg.programs_per_step, rng=rng_step)

        episodes: List[ProgramEpisode] = []
        run_ids: List[str] = []
        cached_count = 0
        warm_start_meta_by_run: Dict[str, Optional[Dict[str, Any]]] = {}
        genotype_meta_by_run: Dict[str, Dict[str, Any]] = {}
        unseen_set_by_run: Dict[str, Optional[str]] = {}
        transfer_set_by_run: Dict[str, Optional[str]] = {}
        warm_start_skip_by_run: Dict[str, Dict[str, Any]] = {}
        warm_start_elite_by_run: Dict[str, Dict[str, Any]] = {}
        warm_start_relax_by_run: Dict[str, Dict[str, Any]] = {}
        warm_start_audition_by_run: Dict[str, Dict[str, Any]] = {}
        warm_start_regime_by_run: Dict[str, Dict[str, Any]] = {}
        warm_start_fallback_by_run: Dict[str, Dict[str, Any]] = {}
        scratch_prob = 0.10
        warm_start_margin = 0.002
        fallback_audition_margin = 0.001
        stale_audition_margin = 0.002
        fallback_audition_frac = 0.25
        fallback_audition_min_steps = 25
        stale_audition_seeds = 2
        checkpoint_override_gain = 0.02
        checkpoint_override_parent_gain = 0.0
        checkpoint_override_rescue_gain = 0.06
        checkpoint_override_rescue_parent_gain = 0.0
        checkpoint_override_max_regress = 0.003
        checkpoint_override_transfer_gain = float(active_override_gene.transfer_floor)
        checkpoint_override_transfer_gain_eps = max(float(active_override_gene.transfer_floor), 0.00390625)
        checkpoint_override_rescue_transfer_gain = -0.002
        checkpoint_override_rescue_transfer_gain_eps = 0.0
        # Strict-gate tradeoff controls:
        # - allow ties on transfer (>= 0) by default in strict path
        # - allow mild parent regressions when the combined score is strong
        # - for pure tie-mode acceptance, require stronger primary gain
        # - for parent-regress tolerated mode, require at least 1 transfer tick
        checkpoint_override_tradeoff_lambda = float(active_override_gene.lam)
        checkpoint_override_tradeoff_min = 0.02
        checkpoint_override_parent_tie_tolerance = float(active_override_gene.parent_tolerance)
        checkpoint_override_parent_score_tolerance = max(
            float(active_override_gene.parent_tolerance),
            float(active_override_gene.parent_tolerance) * 2.0,
        )
        checkpoint_override_transfer_tie_primary_min = float(active_override_gene.tie_primary_min)
        checkpoint_override_parent_regress_transfer_floor = float(active_override_gene.transfer_floor)
        fallback_margin = 0.002
        collapse_scalar_threshold = 0.1
        collapse_split_threshold = 0.1
        warmup_limit = getattr(self.progress, "warmup_steps", 0)
        baseline_unseen = None
        try:
            baseline_unseen = self.progress._baseline("unseen_accuracy")  # type: ignore[attr-defined]
        except Exception:
            baseline_unseen = None
        ab_probe_done = False
        ab_probe_limit = 30
        ab_probe_count = getattr(self, "_ab_probe_count", 0)
        ab_probe_rate = 0.25 if ab_probe_count < ab_probe_limit else 0.05
        ab_probe_meta: Dict[str, Any] = {"inherit_ab_used": False}
        audition_considered_count = 0
        audition_used_count = 0
        audition_win_count = 0
        candidate_exec_contexts: List[Dict[str, Any]] = []
        if not hasattr(self, "_pool_elite_cache"):
            self._pool_elite_cache = {}
        if not hasattr(self, "_pool_elite_cache_by_set"):
            self._pool_elite_cache_by_set = {}

        for i, program in enumerate(programs):
            task_specs = self._build_task_specs(
                cfg,
                rng_step,
                i,
                program=program,
                unseen_pool_seed=unseen_pool_seed,
                transfer_seed=transfer_seed,
            )
            current_unseen_set_id = _hash_unseen_set(task_specs)
            current_transfer_set_id = _hash_transfer_set(task_specs)
            # Allow program-specific budget override (e.g., BUDGET.steps variable)
            budget_local = dict(cfg.budget) if isinstance(cfg.budget, dict) else {"max_steps": cfg.budget}
            try:
                if hasattr(program, "meta") and isinstance(program.meta, dict) and program.meta.get("budget_steps"):
                    budget_local["max_steps"] = int(program.meta["budget_steps"])
            except Exception:
                pass
            arch_type = "unknown"
            try:
                arch_type = str(program.graph.nodes["ARCH:0"].spec.get("type", "unknown"))
            except Exception:
                arch_type = "unknown"
            signature_hint = None
            effective_arch_type = arch_type
            if hasattr(self.runner, "signature_for"):
                try:
                    if hasattr(self.runner, "signature_info"):
                        sig_info = self.runner.signature_info(
                            program=program, task_specs=task_specs, budget=budget_local
                        )
                        if isinstance(sig_info, dict):
                            signature_hint = sig_info.get("signature")
                            effective_arch_type = sig_info.get("effective_arch_type") or arch_type
                        elif isinstance(sig_info, tuple) and len(sig_info) >= 2:
                            signature_hint = sig_info[0]
                            effective_arch_type = sig_info[1] or arch_type
                    else:
                        signature_hint = self.runner.signature_for(
                            program=program, task_specs=task_specs, budget=budget_local
                        )
                except Exception:
                    signature_hint = None
            opt_type = _safe_get_by_path(program, "LRULE:0.spec.type", None)
            obj_primary = _safe_get_by_path(program, "OBJ:0.spec.losses[0].kind", None)
            lr_bin = _safe_get_by_path(program, "LRULE:0.spec.hyper.base_lr", None)
            curriculum = _safe_get_by_path(program, "CURR:0.spec.curriculum", None)
            if not isinstance(curriculum, dict):
                curriculum = {}
            budget_steps = (budget_local or {}).get("max_steps") or (budget_local or {}).get("steps")
            try:
                budget_steps = int(budget_steps) if budget_steps is not None else None
            except Exception:
                budget_steps = None
            regime_payload = {
                "obj_primary": obj_primary,
                "obj_spec": _safe_get_by_path(program, "OBJ:0.spec", None),
                "opt_type": opt_type,
                "opt_hyper": _safe_get_by_path(program, "LRULE:0.spec.hyper", None),
                "curriculum": curriculum,
                "budget_steps": budget_steps,
                "batch_size": (budget_local or {}).get("batch_size"),
            }
            try:
                regime_id = _hash_payload(regime_payload)
            except Exception:
                regime_id = "unknown"
            family_payload = {
                "obj_primary": str(obj_primary) if obj_primary is not None else "unknown",
                "opt_type": str(opt_type) if opt_type is not None else "unknown",
                "budget_steps": _bucket_int(budget_steps, step=400),
                "max_len_train": _bucket_int(curriculum.get("max_len_train"), step=32),
                "min_len_train": _bucket_int(curriculum.get("min_len_train"), step=32),
                "max_len_shift": _bucket_int(curriculum.get("max_len_shift"), step=32),
                "min_len_shift": _bucket_int(curriculum.get("min_len_shift"), step=32),
            }
            try:
                regime_family_id = _hash_payload(family_payload)
            except Exception:
                regime_family_id = "unknown"
            checkpoint_filter: Dict[str, Any] = {
                "opt_type": str(opt_type) if opt_type is not None else "unknown",
                "obj_primary": str(obj_primary) if obj_primary is not None else "unknown",
            }
            checkpoint_filter["unseen_pool_idx"] = int(unseen_pool_idx)
            checkpoint_filter["regime_id"] = regime_id
            checkpoint_filter["regime_family_id"] = regime_family_id
            if lr_bin is not None:
                try:
                    checkpoint_filter["lr_bin"] = float(lr_bin)
                except Exception:
                    checkpoint_filter["lr_bin"] = str(lr_bin)
            else:
                checkpoint_filter["lr_bin"] = "unknown"
            warm_start_key = None
            init_state = None
            init_source = "none"
            init_signature = None
            init_meta = None
            warm_start_skip = {"skipped": False, "reason": "", "key": ""}
            warm_start_elite = {"found": False, "score": None, "same_set": False}
            warm_start_relaxed = {
                "enabled": False,
                "margin": None,
                "used": False,
            }
            warm_start_audition = {
                "considered": False,
                "used": False,
                "win": None,
                "eligibility": "",
                "required_for": "",
                "ineligible_reason": "",
                "missing_keys": "",
                "delta_unseen": None,
                "delta_unseen_b": None,
                "mean_delta": None,
                "min_delta": None,
                "fallback_unseen": None,
                "fallback_unseen_b": None,
                "scratch_unseen": None,
                "scratch_unseen_b": None,
                "mode": "",
                "candidate_type": "",
                "margin": fallback_audition_margin,
                "steps": None,
                "block_reason": "",
                "unseen_set_match": None,
                "checkpoint_override": False,
                "checkpoint_override_gain": checkpoint_override_gain,
                "checkpoint_override_parent_gain": checkpoint_override_parent_gain,
                "checkpoint_override_rescue_gain": checkpoint_override_rescue_gain,
                "checkpoint_override_rescue_parent_gain": checkpoint_override_rescue_parent_gain,
                "checkpoint_override_max_regress": checkpoint_override_max_regress,
                "checkpoint_override_transfer_gain": checkpoint_override_transfer_gain,
                "checkpoint_override_transfer_gain_eps": checkpoint_override_transfer_gain_eps,
                "checkpoint_override_rescue_transfer_gain": checkpoint_override_rescue_transfer_gain,
                "checkpoint_override_rescue_transfer_gain_eps": checkpoint_override_rescue_transfer_gain_eps,
                "checkpoint_override_tradeoff_lambda": checkpoint_override_tradeoff_lambda,
                "checkpoint_override_tradeoff_min": checkpoint_override_tradeoff_min,
                "checkpoint_override_parent_tie_tolerance": checkpoint_override_parent_tie_tolerance,
                "checkpoint_override_parent_score_tolerance": checkpoint_override_parent_score_tolerance,
                "checkpoint_override_transfer_tie_primary_min": checkpoint_override_transfer_tie_primary_min,
                "checkpoint_override_parent_regress_transfer_floor": checkpoint_override_parent_regress_transfer_floor,
                "strict_allow_reason": "",
                "accept_mode": "",
                "strict_score": None,
                "transfer_gate_pass": None,
                "transfer_gate_tier": "none",
                "checkpoint_override_tier": "",
                "parent_unseen": None,
                "parent_delta": None,
                "probe_unseen": None,
                "error_code": "",
                "error_msg": "",
                "error_where": "",
            }
            warm_start_regime = {"fallback": False, "match": None, "family_match": None}
            warm_start_fallback = {
                "considered": False,
                "used": False,
                "blocked_reason": "",
                "parent_unseen": None,
                "pool_baseline": None,
                "delta": None,
            }
            stale_candidate = False
            if step_idx == 0:
                warm_start_skip = {"skipped": True, "reason": "forced_scratch_step0", "key": ""}
                init_source = "forced_scratch_step0"
            else:
                try:
                    sig_key = str(signature_hint) if signature_hint is not None else "unknown"
                    cache_key = (effective_arch_type, int(unseen_pool_idx), str(regime_id), sig_key)
                    cache_by_set = getattr(self, "_pool_elite_cache_by_set", {})
                    cache = getattr(self, "_pool_elite_cache", {})
                    rec_pool_strict = None
                    rec_same_strict = None
                    rec_pool_loose = None
                    rec_same_loose = None
                    rec = None
                    rec_meta: Dict[str, Any] = {}
                    rec_key = ""
                    rec_unseen = None
                    if current_unseen_set_id is not None:
                        cached = cache_by_set.get((cache_key, current_unseen_set_id))
                        if cached is not None:
                            rec_same_strict = CheckpointRecord(
                                path=Path(cached["path"]),
                                meta=cached["meta"],
                                state_dict=cached["state_dict"],
                            )
                            rec_pool_strict = rec_same_strict
                    if rec_pool_strict is None:
                        cached = cache.get(cache_key)
                        if cached is not None:
                            rec_pool_strict = CheckpointRecord(
                                path=Path(cached["path"]),
                                meta=cached["meta"],
                                state_dict=cached["state_dict"],
                            )
                    rec = rec_same_strict or rec_pool_strict
                    if rec is None:
                        rec_pool_strict = self.checkpoints.load_best(
                            effective_arch_type, signature=signature_hint, meta_filter=checkpoint_filter
                        )
                        rec_same_strict = None
                        if current_unseen_set_id is not None:
                            meta_filter_same = dict(checkpoint_filter)
                            meta_filter_same["unseen_set_id"] = current_unseen_set_id
                            rec_same_strict = self.checkpoints.load_best(
                                effective_arch_type, signature=signature_hint, meta_filter=meta_filter_same
                            )
                        rec = rec_same_strict or rec_pool_strict
                    if rec is None:
                        warm_start_fallback["considered"] = True
                        meta_filter_loose = dict(checkpoint_filter)
                        meta_filter_loose.pop("regime_id", None)
                        meta_filter_loose.pop("lr_bin", None)
                        rec_pool_loose = self.checkpoints.load_best(
                            effective_arch_type, signature=signature_hint, meta_filter=meta_filter_loose
                        )
                        if current_unseen_set_id is not None:
                            meta_filter_same_loose = dict(meta_filter_loose)
                            meta_filter_same_loose["unseen_set_id"] = current_unseen_set_id
                            rec_same_loose = self.checkpoints.load_best(
                                effective_arch_type, signature=signature_hint, meta_filter=meta_filter_same_loose
                            )
                        rec = rec_same_loose or rec_pool_loose
                        rec_transfer = None
                        if rec is not None:
                            warm_start_regime["fallback"] = True
                        else:
                            warm_start_fallback["blocked_reason"] = "no_parent"
                    baseline_pool = None
                    try:
                        baseline_pool = getattr(self, "_baseline_unseen_by_pool", {}).get(int(unseen_pool_idx))
                    except Exception:
                        baseline_pool = None
                    warm_start_fallback["pool_baseline"] = baseline_pool
                    if rec is None and not warm_start_skip["skipped"]:
                        warm_start_skip = {
                            "skipped": True,
                            "reason": warm_start_fallback.get("blocked_reason") or "no_parent",
                            "key": "",
                        }
                        init_state = None
                        init_meta = None
                        init_signature = None
                        warm_start_key = None
                        init_source = "no_parent"
                    if rec is not None:
                        warm_start_elite["found"] = True
                        warm_start_elite["same_set"] = bool(rec_same_strict is not None or rec_same_loose is not None)
                        rec_meta = rec.meta if isinstance(rec.meta, dict) else {}
                        rec_key = rec_meta.get("checkpoint_id") if isinstance(rec_meta, dict) else None
                        rec_key = rec_key or rec.path.name
                        rec_unseen = rec_meta.get("unseen_score") if isinstance(rec_meta, dict) else None
                        rec_transfer = rec_meta.get("transfer_unseen_accuracy") if isinstance(rec_meta, dict) else None
                        warm_start_elite["score"] = rec_unseen
                        warm_start_fallback["parent_unseen"] = rec_unseen
                        rec_regime = rec_meta.get("regime_id") if isinstance(rec_meta, dict) else None
                        if rec_regime is not None:
                            warm_start_regime["match"] = bool(rec_regime == regime_id)
                        else:
                            warm_start_regime["match"] = False
                        rec_family = rec_meta.get("regime_family_id") if isinstance(rec_meta, dict) else None
                        if rec_family is not None:
                            warm_start_regime["family_match"] = bool(rec_family == regime_family_id)
                        else:
                            warm_start_regime["family_match"] = False
                        if baseline_pool is not None and rec_unseen is not None:
                            try:
                                warm_start_fallback["delta"] = float(rec_unseen) - float(baseline_pool)
                            except Exception:
                                warm_start_fallback["delta"] = None
                        if warm_start_regime["fallback"] and not warm_start_regime.get("family_match"):
                            warm_start_fallback["blocked_reason"] = "family_mismatch"
                            warm_start_skip = {
                                "skipped": True,
                                "reason": "fallback_family_mismatch",
                                "key": str(rec_key),
                            }
                            init_state = None
                            init_meta = None
                            init_signature = None
                            warm_start_key = None
                            init_source = "stale_checkpoint"
                        if (
                            warm_start_regime["fallback"]
                            and not warm_start_elite["same_set"]
                            and not warm_start_skip["skipped"]
                        ):
                            if baseline_pool is None or rec_unseen is None:
                                warm_start_fallback["blocked_reason"] = "fallback_no_baseline"
                                warm_start_skip = {
                                    "skipped": True,
                                    "reason": "fallback_no_baseline",
                                    "key": str(rec_key),
                                }
                                init_state = None
                                init_meta = None
                                init_signature = None
                                warm_start_key = None
                                init_source = "stale_checkpoint"
                            else:
                                if float(rec_unseen) < float(baseline_pool) + fallback_margin:
                                    warm_start_fallback["blocked_reason"] = "fallback_low_delta"
                                    warm_start_skip = {
                                        "skipped": True,
                                        "reason": "fallback_low_delta",
                                        "key": str(rec_key),
                                    }
                                    init_state = None
                                    init_meta = None
                                    init_signature = None
                                    warm_start_key = None
                                    init_source = "stale_checkpoint"
                        stale_by_margin = (
                            step_idx >= warmup_limit
                            and baseline_pool is not None
                            and rec_unseen is not None
                            and float(rec_unseen) < float(baseline_pool) - warm_start_margin
                        )
                        if stale_by_margin:
                            warm_start_relaxed["enabled"] = True
                            warm_start_relaxed["margin"] = 0.0
                            stale_candidate = True
                            if not warm_start_elite["same_set"]:
                                warm_start_skip = {
                                    "skipped": True,
                                    "reason": "stale_same_set_required",
                                    "key": str(rec_key),
                                }
                                init_state = None
                                init_meta = None
                                init_signature = None
                                warm_start_key = None
                                init_source = "stale_same_set_required"
                    if not warm_start_skip["skipped"]:
                        rng_scratch = random.Random(self._split_rng(rng_step, step_idx, i, "forced_scratch"))
                        if rng_scratch.random() < scratch_prob:
                            init_state = None
                            init_meta = None
                            init_signature = None
                            warm_start_key = None
                            init_source = "forced_scratch"
                        else:
                            use_checkpoint = True
                            is_fallback = bool(warm_start_regime.get("fallback"))
                            audition_mode = "stale" if stale_candidate else ("fallback" if is_fallback else "checkpoint")
                            audition_margin = stale_audition_margin if stale_candidate else fallback_audition_margin
                            audition_seeds = stale_audition_seeds if stale_candidate else 1
                            warm_start_audition["considered"] = True
                            warm_start_audition["mode"] = audition_mode
                            warm_start_audition["candidate_type"] = audition_mode
                            warm_start_audition["margin"] = audition_margin
                            warm_start_audition["eligibility"] = ""
                            warm_start_audition["required_for"] = ""
                            warm_start_audition["ineligible_reason"] = ""
                            warm_start_audition["missing_keys"] = ""
                            audition_considered_count += 1
                            audition_allowed = True
                            audition_same_set = bool(warm_start_elite.get("same_set"))
                            warm_start_audition["unseen_set_match"] = audition_same_set
                            if audition_mode in {"stale", "fallback"} and not audition_same_set:
                                audition_allowed = False
                                warm_start_audition["block_reason"] = "audition_unseen_set_mismatch"
                                warm_start_audition["required_for"] = "unseen_set_match"
                                warm_start_audition["ineligible_reason"] = "unseen_set_mismatch"
                                warm_start_audition["eligibility"] = "none"
                            try:
                                full_steps = budget_local.get("max_steps", budget_local.get("steps", 200))
                                full_steps = int(full_steps)
                            except Exception:
                                full_steps = 200
                            probe_steps = max(
                                fallback_audition_min_steps,
                                min(full_steps, int(full_steps * fallback_audition_frac)),
                            )
                            warm_start_audition["steps"] = probe_steps
                            budget_probe = dict(budget_local)
                            budget_probe["max_steps"] = probe_steps
                            budget_probe["steps"] = probe_steps
                            audition_signature = signature_hint
                            if audition_signature is None and isinstance(rec_meta, dict):
                                audition_signature = rec_meta.get("model_signature")
                            try:
                                audition_error = False
                                if not audition_allowed:
                                    warm_start_audition["win"] = None
                                else:
                                    deltas = []
                                    fallback_vals = []
                                    scratch_vals = []
                                    fallback_transfer_vals = []
                                    scratch_transfer_vals = []
                                    missing_keys = set()
                                    fallback_unseen_numeric = []
                                    fallback_transfer_numeric = []
                                    def _audition_unseen(eval_report, result):
                                        diag = getattr(eval_report, "diagnostics", {}) or {}
                                        split = diag.get("split_metrics") or {}
                                        unseen_val = (split.get("unseen") or {}).get("accuracy")
                                        if unseen_val is None and isinstance(result, dict):
                                            unseen_val = (result.get("metrics") or {}).get("unseen_accuracy")
                                        return unseen_val
                                    def _audition_transfer(result):
                                        if isinstance(result, dict):
                                            return (result.get("metrics") or {}).get("transfer_accuracy")
                                        return None
                                    for seed_idx in range(audition_seeds):
                                        seed_probe = self._split_rng(
                                            rng_step, step_idx, i, "fallback_audition", seed_idx
                                        )
                                        fallback_key = f"{rec_key}:audition:{step_idx}:{i}:{audition_mode}:{seed_idx}:fallback"
                                        scratch_key = f"{rec_key}:audition:{step_idx}:{i}:{audition_mode}:{seed_idx}:scratch"
                                        fallback_ctx = RunContext.from_parts(
                                            program=program,
                                            task_specs=task_specs,
                                            budget=budget_probe,
                                            eval_contract=self.eval_contract_spec,
                                            seed=seed_probe,
                                            runner_version=self.runner_version,
                                            engine_step=step_idx,
                                            cache_scope=self.cache_scope,
                                            warm_start_key=fallback_key,
                                        )
                                        fallback_res = execute_run_with_cache_and_artifacts(
                                            store=self.store,
                                            artifact_store=self.artifact_store,
                                            runner=self.runner,
                                            run_ctx=fallback_ctx,
                                            program=program,
                                            task_specs=task_specs,
                                            budget=budget_probe,
                                            eval_contract=self.eval_contract_spec,
                                            rng=seed_probe,
                                            runner_init_state_dict=rec.state_dict,
                                            runner_init_source=f"{audition_mode}_audition",
                                            runner_init_signature=audition_signature,
                                        )
                                        scratch_ctx = RunContext.from_parts(
                                            program=program,
                                            task_specs=task_specs,
                                            budget=budget_probe,
                                            eval_contract=self.eval_contract_spec,
                                            seed=seed_probe,
                                            runner_version=self.runner_version,
                                            engine_step=step_idx,
                                            cache_scope=self.cache_scope,
                                            warm_start_key=scratch_key,
                                        )
                                        scratch_res = execute_run_with_cache_and_artifacts(
                                            store=self.store,
                                            artifact_store=self.artifact_store,
                                            runner=self.runner,
                                            run_ctx=scratch_ctx,
                                            program=program,
                                            task_specs=task_specs,
                                            budget=budget_probe,
                                            eval_contract=self.eval_contract_spec,
                                            rng=seed_probe,
                                            runner_init_state_dict=None,
                                            runner_init_source=f"{audition_mode}_audition_scratch",
                                            runner_init_signature=None,
                                        )
                                        fallback_eval = self.evaluator.evaluate(fallback_res.run_result)
                                        scratch_eval = self.evaluator.evaluate(scratch_res.run_result)
                                        fallback_unseen = _audition_unseen(fallback_eval, fallback_res.run_result)
                                        scratch_unseen = _audition_unseen(scratch_eval, scratch_res.run_result)
                                        fallback_transfer = _audition_transfer(fallback_res.run_result)
                                        scratch_transfer = _audition_transfer(scratch_res.run_result)
                                        fallback_vals.append(fallback_unseen)
                                        scratch_vals.append(scratch_unseen)
                                        fallback_transfer_vals.append(fallback_transfer)
                                        scratch_transfer_vals.append(scratch_transfer)
                                        if fallback_unseen is None:
                                            missing_keys.add("fallback_unseen")
                                        if scratch_unseen is None:
                                            missing_keys.add("scratch_unseen")
                                        if fallback_transfer is None:
                                            missing_keys.add("fallback_transfer")
                                        if scratch_transfer is None:
                                            missing_keys.add("scratch_transfer")
                                        if fallback_unseen is None or scratch_unseen is None:
                                            continue
                                        delta = float(fallback_unseen) - float(scratch_unseen)
                                        deltas.append(delta)
                                        fallback_unseen_numeric.append(float(fallback_unseen))
                                        if fallback_transfer is not None:
                                            fallback_transfer_numeric.append(float(fallback_transfer))
                                    if missing_keys:
                                        warm_start_audition["missing_keys"] = ";".join(sorted(missing_keys))
                                    if not deltas:
                                        if not warm_start_audition.get("block_reason"):
                                            warm_start_audition["block_reason"] = "audition_missing_metrics"
                                            warm_start_audition["error_code"] = "missing_metrics"
                                            warm_start_audition["error_msg"] = "missing_metrics"
                                        warm_start_audition["required_for"] = "paired_gain"
                                        warm_start_audition["ineligible_reason"] = "missing_metrics:paired_gain"
                                        warm_start_audition["eligibility"] = "none"
                                        warm_start_audition["win"] = False
                                    else:
                                        mean_delta = sum(deltas) / len(deltas)
                                        min_delta = min(deltas)
                                        mean_fallback = None
                                        try:
                                            if fallback_unseen_numeric:
                                                mean_fallback = sum(fallback_unseen_numeric) / len(fallback_unseen_numeric)
                                        except Exception:
                                            mean_fallback = None
                                        mean_fallback_transfer = None
                                        try:
                                            if fallback_transfer_numeric:
                                                mean_fallback_transfer = sum(fallback_transfer_numeric) / len(fallback_transfer_numeric)
                                        except Exception:
                                            mean_fallback_transfer = None
                                        warm_start_audition["parent_unseen"] = rec_unseen
                                        if mean_fallback is not None and rec_unseen is not None:
                                            try:
                                                warm_start_audition["parent_delta"] = (
                                                    float(mean_fallback) - float(rec_unseen)
                                                )
                                            except Exception:
                                                warm_start_audition["parent_delta"] = None
                                        warm_start_audition["transfer_parent_unseen"] = rec_transfer
                                        if mean_fallback_transfer is not None:
                                            warm_start_audition["transfer_probe_unseen"] = mean_fallback_transfer
                                        if mean_fallback_transfer is not None and rec_transfer is not None:
                                            try:
                                                warm_start_audition["transfer_delta"] = (
                                                    float(mean_fallback_transfer) - float(rec_transfer)
                                                )
                                            except Exception:
                                                warm_start_audition["transfer_delta"] = None
                                        if mean_fallback is not None:
                                            warm_start_audition["probe_unseen"] = mean_fallback
                                        elif fallback_vals:
                                            warm_start_audition["probe_unseen"] = fallback_vals[0]
                                        warm_start_audition["delta_unseen"] = deltas[0] if deltas else None
                                        warm_start_audition["fallback_unseen"] = (
                                            fallback_vals[0] if len(fallback_vals) > 0 else None
                                        )
                                        warm_start_audition["scratch_unseen"] = (
                                            scratch_vals[0] if len(scratch_vals) > 0 else None
                                        )
                                        if audition_seeds > 1:
                                            warm_start_audition["delta_unseen_b"] = (
                                                deltas[1] if len(deltas) > 1 else None
                                            )
                                            warm_start_audition["fallback_unseen_b"] = (
                                                fallback_vals[1] if len(fallback_vals) > 1 else None
                                            )
                                            warm_start_audition["scratch_unseen_b"] = (
                                                scratch_vals[1] if len(scratch_vals) > 1 else None
                                            )
                                        warm_start_audition["mean_delta"] = mean_delta
                                        warm_start_audition["min_delta"] = min_delta
                                        primary_available = True
                                        transfer_available = mean_fallback_transfer is not None and rec_transfer is not None
                                        if primary_available and transfer_available:
                                            warm_start_audition["eligibility"] = "full"
                                        elif primary_available:
                                            warm_start_audition["eligibility"] = "primary_only"
                                        elif transfer_available:
                                            warm_start_audition["eligibility"] = "transfer_only"
                                        else:
                                            warm_start_audition["eligibility"] = "none"
                                        if stale_candidate:
                                            warm_start_audition["win"] = bool(
                                                min_delta >= 0.0 and mean_delta >= audition_margin
                                            )
                                        else:
                                            warm_start_audition["win"] = bool(mean_delta >= audition_margin)
                            except Exception as ex:
                                warm_start_audition["win"] = False
                                warm_start_audition["block_reason"] = "audition_error"
                                warm_start_audition["error_code"] = type(ex).__name__
                                warm_start_audition["error_msg"] = str(ex)
                                warm_start_audition["required_for"] = "runtime_error"
                                warm_start_audition["ineligible_reason"] = "audition_error"
                                if not warm_start_audition.get("eligibility"):
                                    warm_start_audition["eligibility"] = "none"
                                tb = traceback.extract_tb(ex.__traceback__)
                                if tb:
                                    last = tb[-1]
                                    warm_start_audition["error_where"] = f"{last.filename}:{last.lineno}:{last.name}"
                            if warm_start_audition.get("block_reason") == "audition_missing_metrics":
                                warm_start_audition["block_reason"] = "audition_ineligible_missing_metrics"
                                if not warm_start_audition.get("required_for"):
                                    warm_start_audition["required_for"] = "paired_gain"
                                if not warm_start_audition.get("ineligible_reason"):
                                    warm_start_audition["ineligible_reason"] = "missing_metrics:paired_gain"
                                if not warm_start_audition.get("eligibility"):
                                    warm_start_audition["eligibility"] = "none"
                                if warm_start_audition.get("considered"):
                                    warm_start_audition["considered"] = False
                                    if audition_considered_count > 0:
                                        audition_considered_count -= 1
                                warm_start_audition["used"] = False
                                warm_start_audition["win"] = None
                                audition_allowed = False
                            audition_error = warm_start_audition.get("block_reason") in {
                                "audition_error",
                            }
                            if audition_error:
                                warm_start_audition["considered"] = False
                                warm_start_audition["used"] = False
                                warm_start_audition["win"] = None
                                if audition_considered_count > 0:
                                    audition_considered_count -= 1
                                use_checkpoint = False
                                if is_fallback:
                                    warm_start_fallback["used"] = False
                                    warm_start_fallback["blocked_reason"] = "audition_error"
                                init_state = None
                                init_meta = None
                                init_signature = None
                                warm_start_key = None
                                init_source = "audition_error_forced_scratch"
                            elif not audition_allowed:
                                use_checkpoint = False
                                if is_fallback:
                                    warm_start_fallback["used"] = False
                                    warm_start_fallback["blocked_reason"] = warm_start_audition.get("block_reason")
                            elif not warm_start_audition.get("win"):
                                use_checkpoint = False
                                if is_fallback:
                                    warm_start_fallback["used"] = False
                                    warm_start_fallback["blocked_reason"] = (
                                        warm_start_audition.get("block_reason") or "audition_failed"
                                    )
                                warm_start_audition["block_reason"] = (
                                    warm_start_audition.get("block_reason")
                                    or f"{audition_mode}_audition_failed"
                                )
                            else:
                                audition_win_count += 1
                                if audition_mode == "checkpoint":
                                    mean_delta = warm_start_audition.get("mean_delta")
                                    parent_delta = warm_start_audition.get("parent_delta")
                                    transfer_delta = warm_start_audition.get("transfer_delta")
                                    allow_override = False
                                    override_tier = ""
                                    transfer_gate_pass = None
                                    transfer_gate_tier = "none"
                                    if not audition_same_set:
                                        warm_start_audition["block_reason"] = "override_unseen_set_mismatch"
                                        warm_start_audition["required_for"] = "unseen_set_match"
                                    elif mean_delta is None or parent_delta is None:
                                        warm_start_audition["block_reason"] = "override_missing_metrics"
                                        warm_start_audition["required_for"] = "parent_delta"
                                        if not warm_start_audition.get("ineligible_reason"):
                                            warm_start_audition["ineligible_reason"] = "missing_metrics:parent_delta"
                                    else:
                                        warm_start_audition["checkpoint_override_transfer_gain"] = (
                                            checkpoint_override_transfer_gain
                                        )
                                        warm_start_audition["checkpoint_override_transfer_gain_eps"] = (
                                            checkpoint_override_transfer_gain_eps
                                        )
                                        warm_start_audition["checkpoint_override_rescue_transfer_gain"] = (
                                            checkpoint_override_rescue_transfer_gain
                                        )
                                        warm_start_audition["checkpoint_override_rescue_transfer_gain_eps"] = (
                                            checkpoint_override_rescue_transfer_gain_eps
                                        )
                                        mean_delta_val = float(mean_delta)
                                        parent_delta_val = float(parent_delta)
                                        transfer_delta_val = (
                                            float(transfer_delta) if transfer_delta is not None else None
                                        )
                                        # Strict transfer floor:
                                        # ties should pass (>= 0.0) unless explicitly configured below 0.
                                        strict_transfer_floor = min(
                                            float(checkpoint_override_transfer_gain),
                                            float(checkpoint_override_transfer_gain_eps),
                                            0.0,
                                        )
                                        score_val = (
                                            mean_delta_val + (checkpoint_override_tradeoff_lambda * transfer_delta_val)
                                            if transfer_delta_val is not None
                                            else None
                                        )
                                        strict_gain_ok = mean_delta_val >= checkpoint_override_gain
                                        strict_parent_ok = parent_delta_val >= checkpoint_override_parent_gain
                                        strict_parent_soft_ok = (
                                            parent_delta_val >= -checkpoint_override_parent_tie_tolerance
                                        )
                                        strict_parent_score_ok = (
                                            parent_delta_val >= -checkpoint_override_parent_score_tolerance
                                            and score_val is not None
                                            and score_val >= checkpoint_override_tradeoff_min
                                        )
                                        rescue_ok = (
                                            mean_delta_val >= checkpoint_override_rescue_gain
                                            and parent_delta_val >= checkpoint_override_rescue_parent_gain
                                            and parent_delta_val >= -checkpoint_override_max_regress
                                            and parent_delta_val >= 0.0
                                        )
                                        strict_transfer_ok = (
                                            transfer_delta_val is not None
                                            and transfer_delta_val >= strict_transfer_floor
                                        )
                                        strict_parent_mode = "none"
                                        if strict_parent_ok:
                                            strict_parent_mode = "parent_ok"
                                        elif strict_parent_soft_ok:
                                            strict_parent_mode = "parent_tie_tolerance"
                                        elif strict_parent_score_ok:
                                            strict_parent_mode = "parent_score_compensated"
                                        strict_transfer_mode = "none"
                                        if transfer_delta_val is not None:
                                            strict_transfer_mode = "tie" if transfer_delta_val == 0.0 else "positive"
                                        strict_accept_mode = "original_strict"
                                        if strict_parent_mode in {"parent_tie_tolerance", "parent_score_compensated"}:
                                            strict_accept_mode = "parent_regress_tolerated"
                                        elif strict_transfer_mode == "tie":
                                            strict_accept_mode = "transfer_tie_ok"
                                        strict_transfer_tie_primary_ok = (
                                            strict_accept_mode != "transfer_tie_ok"
                                            or mean_delta_val >= checkpoint_override_transfer_tie_primary_min
                                        )
                                        strict_parent_regress_transfer_ok = (
                                            strict_accept_mode != "parent_regress_tolerated"
                                            or (
                                                transfer_delta_val is not None
                                                and transfer_delta_val >= checkpoint_override_parent_regress_transfer_floor
                                            )
                                        )
                                        warm_start_audition["strict_score"] = score_val
                                        strict_ok = (
                                            strict_gain_ok
                                            and strict_transfer_ok
                                            and strict_transfer_tie_primary_ok
                                            and strict_parent_regress_transfer_ok
                                            and strict_parent_mode != "none"
                                        )
                                        if strict_ok:
                                            transfer_gate_pass = True
                                            transfer_gate_tier = "strict"
                                            allow_override = True
                                            override_tier = "strict"
                                            warm_start_audition["block_reason"] = "override_allowed_strict"
                                            warm_start_audition["accept_mode"] = strict_accept_mode
                                            warm_start_audition["strict_allow_reason"] = (
                                                f"{strict_parent_mode}+transfer_{strict_transfer_mode}"
                                            )
                                        elif strict_gain_ok and transfer_delta_val is None:
                                            warm_start_audition["block_reason"] = "override_missing_metrics"
                                            warm_start_audition["required_for"] = "transfer_delta"
                                            if not warm_start_audition.get("ineligible_reason"):
                                                warm_start_audition["ineligible_reason"] = "missing_metrics:transfer_delta"
                                            if not warm_start_audition.get("eligibility"):
                                                warm_start_audition["eligibility"] = "primary_only"
                                        elif strict_gain_ok and transfer_delta_val < strict_transfer_floor:
                                            warm_start_audition["block_reason"] = (
                                                "checkpoint_override_transfer_gate_failed_strict"
                                            )
                                        elif strict_gain_ok and not strict_transfer_tie_primary_ok:
                                            warm_start_audition["block_reason"] = (
                                                "override_strict_transfer_tie_primary_low"
                                            )
                                        elif strict_gain_ok and not strict_parent_regress_transfer_ok:
                                            warm_start_audition["block_reason"] = (
                                                "override_strict_parent_regress_transfer_floor"
                                            )
                                        elif strict_gain_ok and not (
                                            strict_parent_ok or strict_parent_soft_ok or strict_parent_score_ok
                                        ):
                                            if parent_delta_val < -checkpoint_override_parent_score_tolerance:
                                                warm_start_audition["block_reason"] = "override_strict_parent_regress_hard"
                                            else:
                                                warm_start_audition["block_reason"] = "override_strict_parent_regress_score_low"
                                        elif rescue_ok:
                                            if transfer_delta_val is None:
                                                warm_start_audition["block_reason"] = "override_missing_metrics"
                                                warm_start_audition["required_for"] = "transfer_delta"
                                                if not warm_start_audition.get("ineligible_reason"):
                                                    warm_start_audition["ineligible_reason"] = "missing_metrics:transfer_delta"
                                                if not warm_start_audition.get("eligibility"):
                                                    warm_start_audition["eligibility"] = "primary_only"
                                            elif transfer_delta_val < checkpoint_override_rescue_transfer_gain:
                                                warm_start_audition["block_reason"] = (
                                                    "checkpoint_override_transfer_gate_failed_rescue"
                                                )
                                            elif transfer_delta_val < checkpoint_override_rescue_transfer_gain_eps:
                                                warm_start_audition["block_reason"] = (
                                                    "override_rescue_failed_transfer_gain"
                                                )
                                            else:
                                                transfer_gate_pass = True
                                                transfer_gate_tier = "rescue"
                                                allow_override = True
                                                override_tier = "rescue"
                                                warm_start_audition["block_reason"] = "override_allowed_rescue"
                                                warm_start_audition["accept_mode"] = "rescue"
                                        else:
                                            if mean_delta_val < checkpoint_override_gain:
                                                warm_start_audition["block_reason"] = "override_strict_fail_mean_eps"
                                            elif parent_delta_val < -checkpoint_override_parent_score_tolerance:
                                                warm_start_audition["block_reason"] = "override_strict_parent_regress_hard"
                                            elif parent_delta_val < -checkpoint_override_parent_tie_tolerance:
                                                warm_start_audition["block_reason"] = "override_strict_parent_regress_score_low"
                                            elif transfer_delta_val is not None and transfer_delta_val < strict_transfer_floor:
                                                warm_start_audition["block_reason"] = "override_strict_failed_transfer_gain"
                                            elif mean_delta_val < checkpoint_override_rescue_gain:
                                                warm_start_audition["block_reason"] = "override_rescue_fail_mean_eps"
                                            elif parent_delta_val < 0.0:
                                                warm_start_audition["block_reason"] = "override_rescue_parent_regress"
                                            elif parent_delta_val < -checkpoint_override_max_regress:
                                                warm_start_audition["block_reason"] = "override_max_regress_exceeded"
                                            else:
                                                warm_start_audition["block_reason"] = "override_failed"
                                        warm_start_audition["transfer_gate_pass"] = transfer_gate_pass
                                        warm_start_audition["transfer_gate_tier"] = transfer_gate_tier
                                    if allow_override:
                                        warm_start_audition["checkpoint_override"] = True
                                        warm_start_audition["checkpoint_override_tier"] = override_tier
                                        warm_start_audition["checkpoint_override_transfer_gain"] = (
                                            checkpoint_override_transfer_gain
                                        )
                                        warm_start_audition["checkpoint_override_transfer_gain_eps"] = (
                                            checkpoint_override_transfer_gain_eps
                                        )
                                        warm_start_audition["checkpoint_override_rescue_transfer_gain"] = (
                                            checkpoint_override_rescue_transfer_gain
                                        )
                                        warm_start_audition["used"] = True
                                        audition_used_count += 1
                                    else:
                                        use_checkpoint = False
                                        warm_start_audition["used"] = False
                                        if not warm_start_audition.get("block_reason"):
                                            warm_start_audition["block_reason"] = "fresh_checkpoint_disabled"
                                else:
                                    warm_start_audition["used"] = True
                                    audition_used_count += 1
                            if use_checkpoint:
                                init_state = rec.state_dict
                                init_meta = rec_meta
                                if signature_hint is not None:
                                    init_signature = signature_hint
                                else:
                                    init_signature = init_meta.get("model_signature") if isinstance(init_meta, dict) else None
                                warm_start_key = rec_key
                                if stale_candidate:
                                    init_source = "stale_checkpoint"
                                elif is_fallback:
                                    init_source = "fallback_checkpoint"
                                else:
                                    init_source = (
                                        "checkpoint_override"
                                        if warm_start_audition.get("checkpoint_override")
                                        else "checkpoint"
                                    )
                                if is_fallback:
                                    warm_start_fallback["used"] = True
                            else:
                                if init_source != "audition_error_forced_scratch":
                                    init_state = None
                                    init_meta = None
                                    init_signature = None
                                    warm_start_key = None
                                    init_source = (
                                        f"{audition_mode}_audition_ineligible"
                                        if not audition_allowed
                                        else f"{audition_mode}_audition_reject"
                                    )
                                    if is_fallback:
                                        warm_start_fallback["used"] = False
                except Exception:
                    init_state = None
            seed = self._seed_for(rng, step_idx, i)
            candidate_exec_contexts.append(
                {
                    "index": i,
                    "seed": seed,
                    "program": program,
                    "task_specs": task_specs,
                    "budget": dict(budget_local) if isinstance(budget_local, dict) else budget_local,
                    "effective_arch_type": effective_arch_type,
                    "signature_hint": signature_hint,
                    "checkpoint_filter": dict(checkpoint_filter),
                    "current_unseen_set_id": current_unseen_set_id,
                    "current_transfer_set_id": current_transfer_set_id,
                }
            )
            run_ctx = RunContext.from_parts(
                program=program,
                task_specs=task_specs,
                budget=budget_local,
                eval_contract=self.eval_contract_spec,
                seed=seed,
                runner_version=self.runner_version,
                engine_step=step_idx,
                cache_scope=self.cache_scope,
                warm_start_key=warm_start_key,
            )
            warm_start_meta_by_run[run_ctx.run_id] = dict(init_meta) if isinstance(init_meta, dict) else None
            genotype_meta_by_run[run_ctx.run_id] = dict(checkpoint_filter)
            unseen_set_by_run[run_ctx.run_id] = current_unseen_set_id
            transfer_set_by_run[run_ctx.run_id] = current_transfer_set_id
            if warm_start_skip["skipped"]:
                warm_start_skip_by_run[run_ctx.run_id] = dict(warm_start_skip)
            warm_start_elite_by_run[run_ctx.run_id] = dict(warm_start_elite)
            warm_start_relax_by_run[run_ctx.run_id] = dict(warm_start_relaxed)
            warm_start_audition_by_run[run_ctx.run_id] = dict(warm_start_audition)
            warm_start_regime_by_run[run_ctx.run_id] = dict(warm_start_regime)
            warm_start_fallback_by_run[run_ctx.run_id] = dict(warm_start_fallback)

            exec_result = execute_run_with_cache_and_artifacts(
                store=self.store,
                artifact_store=self.artifact_store,
                runner=self.runner,
                run_ctx=run_ctx,
                program=program,
                task_specs=task_specs,
                budget=budget_local,
                eval_contract=self.eval_contract_spec,
                rng=seed,
                runner_init_state_dict=init_state,
                runner_init_source=init_source,
                runner_init_signature=init_signature,
            )

            cached_count += int(exec_result.cached)
            run_ids.append(run_ctx.run_id)
            eval_report = self.evaluator.evaluate(exec_result.run_result)

            run_metrics = {}
            if isinstance(exec_result.run_result, dict):
                run_metrics = exec_result.run_result.get("metrics", {}) or {}
            warm_used = bool(run_metrics.get("warm_start_used"))
            if (not ab_probe_done) and warm_used:
                rng_ab = random.Random(self._split_rng(rng_step, step_idx, i, "ab_probe"))
                if rng_ab.random() < ab_probe_rate:
                    scratch_ctx = RunContext.from_parts(
                        program=program,
                        task_specs=task_specs,
                        budget=budget_local,
                        eval_contract=self.eval_contract_spec,
                        seed=seed,
                        runner_version=self.runner_version,
                        engine_step=step_idx,
                        cache_scope=self.cache_scope,
                        warm_start_key="scratch_probe",
                    )
                    scratch_result = execute_run_with_cache_and_artifacts(
                        store=self.store,
                        artifact_store=self.artifact_store,
                        runner=self.runner,
                        run_ctx=scratch_ctx,
                        program=program,
                        task_specs=task_specs,
                        budget=budget_local,
                        eval_contract=self.eval_contract_spec,
                        rng=seed,
                        runner_init_state_dict=None,
                        runner_init_source="scratch_probe",
                        runner_init_signature=None,
                    )
                    scratch_metrics = {}
                    if isinstance(scratch_result.run_result, dict):
                        scratch_metrics = scratch_result.run_result.get("metrics", {}) or {}
                    warm_unseen = run_metrics.get("unseen_accuracy")
                    warm_shift = run_metrics.get("shift_accuracy")
                    scratch_unseen = scratch_metrics.get("unseen_accuracy")
                    scratch_shift = scratch_metrics.get("shift_accuracy")
                    delta_unseen = None
                    delta_shift = None
                    if warm_unseen is not None and scratch_unseen is not None:
                        delta_unseen = float(warm_unseen) - float(scratch_unseen)
                    if warm_shift is not None and scratch_shift is not None:
                        delta_shift = float(warm_shift) - float(scratch_shift)
                    ab_probe_meta = {
                        "inherit_ab_used": True,
                        "inherit_ab_arch": arch_type,
                        "inherit_ab_warm_unseen": warm_unseen,
                        "inherit_ab_scratch_unseen": scratch_unseen,
                        "inherit_ab_delta_unseen": delta_unseen,
                        "inherit_ab_warm_shift": warm_shift,
                        "inherit_ab_scratch_shift": scratch_shift,
                        "inherit_ab_delta_shift": delta_shift,
                    }
                    try:
                        self._ab_probe_count = getattr(self, "_ab_probe_count", 0) + 1
                    except Exception:
                        pass
                    ab_probe_done = True

            episodes.append(
                ProgramEpisode(
                    program=program,
                    run_results=[exec_result.run_result],
                    eval_report=eval_report,
                    cached=exec_result.cached,
                )
            )

        rescue_injected = False
        rescue_triggered = False
        rescue_injected_n = 0
        rescue_reason = ""
        rescue_source_used: List[str] = []
        rescue_candidate_ids: List[str] = []
        rescue_supply_status = "not_triggered"
        rescue_supply_fail_reason = ""
        rescue_supply_candidates_seen_n = 0
        rescue_supply_attempts_n = 0
        rescue_supply_source_selected: List[str] = []
        rescue_no_parent_rate_observed = None
        rescue_candidate_n_observed = len(episodes)
        rescue_best_observed = None
        rescue_collapse_observed = None
        rescue_low_split_observed = None
        if bool(getattr(cfg, "rescue_enable", False)):
            provisional_scalars: List[float] = []
            provisional_no_parent_indices: List[int] = []
            provisional_low_unseen_n = 0
            provisional_low_shift_n = 0
            scalar_by_idx: Dict[int, float] = {}

            for idx, ep in enumerate(episodes):
                diag = getattr(ep.eval_report, "diagnostics", {}) or {}
                gen_score = diag.get("generalization_score") or (
                    getattr(ep.eval_report.score, "extra", {}).get("scalar")
                    if hasattr(ep.eval_report, "score")
                    else None
                )
                try:
                    if gen_score is not None:
                        sval = float(gen_score)
                        provisional_scalars.append(sval)
                        scalar_by_idx[idx] = sval
                except Exception:
                    pass
                split = diag.get("split_metrics") or {}
                run_metrics = {}
                if ep.run_results and isinstance(ep.run_results[0], dict):
                    run_metrics = ep.run_results[0].get("metrics", {}) or {}
                source = str(run_metrics.get("warm_start_source") or "").strip()
                if source == "no_parent":
                    provisional_no_parent_indices.append(idx)
                unseen_acc = (split.get("unseen") or {}).get("accuracy")
                shift_acc = (split.get("shift") or {}).get("accuracy")
                if unseen_acc is None:
                    unseen_acc = run_metrics.get("unseen_accuracy")
                if shift_acc is None:
                    shift_acc = run_metrics.get("shift_accuracy")
                try:
                    if unseen_acc is not None and float(unseen_acc) < collapse_split_threshold:
                        provisional_low_unseen_n += 1
                except Exception:
                    pass
                try:
                    if shift_acc is not None and float(shift_acc) < collapse_split_threshold:
                        provisional_low_shift_n += 1
                except Exception:
                    pass

            provisional_candidate_n = len(episodes)
            provisional_no_parent_n = len(provisional_no_parent_indices)
            provisional_no_parent_rate = (
                provisional_no_parent_n / max(1, provisional_candidate_n)
            )
            rescue_no_parent_rate_observed = provisional_no_parent_rate
            rescue_candidate_n_observed = provisional_candidate_n
            provisional_best = max(provisional_scalars) if provisional_scalars else None
            provisional_median = float(np.median(provisional_scalars)) if provisional_scalars else None
            provisional_collapse_flag = bool(
                provisional_median is not None
                and provisional_median < collapse_scalar_threshold
            )
            rescue_best_observed = provisional_best
            rescue_collapse_observed = provisional_collapse_flag
            rescue_low_split_observed = provisional_low_unseen_n + provisional_low_shift_n

            quality_reasons: List[str] = []
            try:
                if provisional_best is not None and float(provisional_best) < float(cfg.rescue_best_floor):
                    quality_reasons.append("best_low")
            except Exception:
                pass
            if provisional_collapse_flag:
                quality_reasons.append("collapse_flag")
            if (provisional_low_unseen_n + provisional_low_shift_n) >= max(1, int(cfg.rescue_low_split_n)):
                quality_reasons.append("low_split")

            no_parent_high = provisional_no_parent_rate >= float(cfg.rescue_no_parent_rate)
            candidate_dense = provisional_candidate_n >= max(1, int(cfg.programs_per_step))
            step_ok = step_idx > 0
            run_remaining = max(0, int(cfg.rescue_max_per_run) - int(getattr(self, "_rescue_injection_total", 0)))
            episode_id_for_limit = gene_state.get("episode_id")
            episode_used = int(getattr(self, "_rescue_injection_episode_counts", {}).get(episode_id_for_limit, 0))
            episode_remaining = max(0, int(cfg.rescue_max_per_episode) - episode_used)
            inject_budget = min(max(0, int(cfg.rescue_inject_n)), run_remaining, episode_remaining)

            rescue_triggered = (
                step_ok
                and no_parent_high
                and candidate_dense
                and bool(quality_reasons)
                and inject_budget > 0
            )

            rescue_supply_fail_counts: Dict[str, int] = {}

            def _select_rescue_record(ctx: Dict[str, Any]) -> Tuple[Optional[CheckpointRecord], str, Dict[str, Any]]:
                effective_arch_type = ctx.get("effective_arch_type")
                signature_hint = ctx.get("signature_hint")
                strict_filter = dict(ctx.get("checkpoint_filter") or {})
                same_set_id = ctx.get("current_unseen_set_id")
                cache_by_set = getattr(self, "_pool_elite_cache_by_set", {}) or {}
                cache = getattr(self, "_pool_elite_cache", {}) or {}

                diag = {
                    "status": "empty",
                    "fail_reason": "no_checkpoints",
                    "candidates_seen_n": 0,
                }
                if not effective_arch_type:
                    diag["status"] = "error"
                    diag["fail_reason"] = "bad_context"
                    return None, "", diag

                def _payload_to_record(payload: Any) -> Optional[CheckpointRecord]:
                    if not isinstance(payload, dict):
                        return None
                    state_dict = payload.get("state_dict")
                    if not isinstance(state_dict, dict):
                        return None
                    path_raw = payload.get("path")
                    if not path_raw:
                        return None
                    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                    return CheckpointRecord(path=Path(str(path_raw)), meta=dict(meta), state_dict=state_dict)

                def _score_payload(payload: Dict[str, Any]) -> float:
                    if not isinstance(payload, dict):
                        return -float("inf")
                    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                    try:
                        return float(meta.get("unseen_score"))
                    except Exception:
                        return -float("inf")

                def _select_best_payload(payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                    best_payload = None
                    best_score = -float("inf")
                    for payload in payloads:
                        score = _score_payload(payload)
                        if best_payload is None or score > best_score:
                            best_payload = payload
                            best_score = score
                    return best_payload

                try:
                    # Ladder stage 1: in-memory cache exact key.
                    sig_key = str(signature_hint) if signature_hint is not None else "unknown"
                    cache_key = None
                    try:
                        cache_key = (
                            str(effective_arch_type),
                            int(strict_filter.get("unseen_pool_idx")),
                            str(strict_filter.get("regime_id")),
                            sig_key,
                        )
                    except Exception:
                        cache_key = None
                    if cache_key is not None and same_set_id is not None:
                        cached = cache_by_set.get((cache_key, same_set_id))
                        rec = _payload_to_record(cached)
                        if rec is not None:
                            diag["status"] = "ok"
                            diag["candidates_seen_n"] = max(1, int(diag["candidates_seen_n"]))
                            return rec, "checkpoint", diag
                    if cache_key is not None:
                        cached = cache.get(cache_key)
                        rec = _payload_to_record(cached)
                        if rec is not None:
                            diag["status"] = "ok"
                            diag["candidates_seen_n"] = max(1, int(diag["candidates_seen_n"]))
                            return rec, "checkpoint", diag

                    # Ladder stage 2: in-memory cache relaxed by signature / regime.
                    arch_payloads: List[Dict[str, Any]] = []
                    for payload in cache.values():
                        if not isinstance(payload, dict):
                            continue
                        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                        if str(meta.get("arch_type")) != str(effective_arch_type):
                            continue
                        arch_payloads.append(payload)
                    diag["candidates_seen_n"] = max(int(diag["candidates_seen_n"]), len(arch_payloads))

                    def _meta_signature(payload: Dict[str, Any]) -> str:
                        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                        return str(meta.get("model_signature")) if meta.get("model_signature") is not None else ""

                    def _meta_set(payload: Dict[str, Any]) -> str:
                        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                        return str(meta.get("unseen_set_id")) if meta.get("unseen_set_id") is not None else ""

                    if arch_payloads and signature_hint is not None:
                        sig_payloads = [p for p in arch_payloads if _meta_signature(p) == str(signature_hint)]
                        if same_set_id is not None:
                            sig_same_payloads = [p for p in sig_payloads if _meta_set(p) == str(same_set_id)]
                            rec = _payload_to_record(_select_best_payload(sig_same_payloads))
                            if rec is not None:
                                diag["status"] = "ok"
                                return rec, "stale_checkpoint", diag
                        rec = _payload_to_record(_select_best_payload(sig_payloads))
                        if rec is not None:
                            diag["status"] = "ok"
                            return rec, "stale_checkpoint", diag

                    if arch_payloads:
                        if same_set_id is not None:
                            same_payloads = [p for p in arch_payloads if _meta_set(p) == str(same_set_id)]
                            rec = _payload_to_record(_select_best_payload(same_payloads))
                            if rec is not None:
                                diag["status"] = "ok"
                                return rec, "fallback_checkpoint", diag
                        rec = _payload_to_record(_select_best_payload(arch_payloads))
                        if rec is not None:
                            diag["status"] = "ok"
                            return rec, "fallback_checkpoint", diag

                    # Ladder stage 3: checkpoint store probes (strict -> relaxed, with and without signature).
                    loose_filter = dict(strict_filter)
                    loose_filter.pop("regime_id", None)
                    loose_filter.pop("lr_bin", None)

                    def _store_probe(meta_filter: Optional[Dict[str, Any]], signature: Optional[str], source: str):
                        rec = self.checkpoints.load_best(
                            str(effective_arch_type),
                            signature=signature,
                            meta_filter=meta_filter,
                        )
                        if rec is not None:
                            diag["status"] = "ok"
                            diag["candidates_seen_n"] = max(1, int(diag["candidates_seen_n"]))
                            return rec, source
                        return None, ""

                    probe_filters: List[Tuple[Optional[Dict[str, Any]], str]] = []
                    if same_set_id is not None:
                        same_strict = dict(strict_filter)
                        same_strict["unseen_set_id"] = same_set_id
                        probe_filters.append((same_strict, "checkpoint"))
                    probe_filters.append((dict(strict_filter), "checkpoint"))
                    if same_set_id is not None:
                        same_loose = dict(loose_filter)
                        same_loose["unseen_set_id"] = same_set_id
                        probe_filters.append((same_loose, "stale_checkpoint"))
                    probe_filters.append((dict(loose_filter), "fallback_checkpoint"))
                    if same_set_id is not None:
                        probe_filters.append(({"unseen_set_id": same_set_id}, "stale_checkpoint"))
                    probe_filters.append((None, "fallback_checkpoint"))

                    signatures: List[Optional[str]] = []
                    if signature_hint is not None:
                        signatures.append(str(signature_hint))
                    signatures.append(None)
                    for sig in signatures:
                        for mf, source in probe_filters:
                            rec, source_label = _store_probe(mf, sig, source)
                            if rec is not None:
                                return rec, source_label, diag

                    if int(diag["candidates_seen_n"]) > 0:
                        diag["status"] = "filtered"
                        diag["fail_reason"] = "all_filtered"
                    else:
                        diag["status"] = "empty"
                        diag["fail_reason"] = "no_checkpoints"
                except Exception:
                    diag["status"] = "error"
                    diag["fail_reason"] = "supply_error"
                return None, "", diag

            if rescue_triggered and provisional_no_parent_indices:
                targets = sorted(
                    provisional_no_parent_indices,
                    key=lambda idx: scalar_by_idx.get(idx, float("inf")),
                )
                for target_idx in targets:
                    if rescue_injected_n >= inject_budget:
                        break
                    if target_idx >= len(candidate_exec_contexts):
                        continue
                    ctx = candidate_exec_contexts[target_idx]
                    rescue_supply_attempts_n += 1
                    rec, source_label, supply_diag = _select_rescue_record(ctx)
                    try:
                        rescue_supply_candidates_seen_n += int(supply_diag.get("candidates_seen_n") or 0)
                    except Exception:
                        pass
                    fail_reason = str(supply_diag.get("fail_reason") or "").strip()
                    if fail_reason:
                        rescue_supply_fail_counts[fail_reason] = rescue_supply_fail_counts.get(fail_reason, 0) + 1
                    if rec is None or not source_label:
                        continue
                    rec_meta = rec.meta if isinstance(rec.meta, dict) else {}
                    rec_key = rec_meta.get("checkpoint_id") if isinstance(rec_meta, dict) else None
                    rec_key = rec_key or rec.path.name
                    rescue_seed = self._split_rng(rng_step, step_idx, target_idx, "rescue_inject", rescue_injected_n)
                    rescue_budget = ctx.get("budget")
                    if isinstance(rescue_budget, dict):
                        rescue_budget = dict(rescue_budget)
                    rescue_warm_key = f"{rec_key}:rescue:{step_idx}:{target_idx}:{rescue_injected_n}"
                    rescue_ctx = RunContext.from_parts(
                        program=ctx["program"],
                        task_specs=ctx["task_specs"],
                        budget=rescue_budget,
                        eval_contract=self.eval_contract_spec,
                        seed=rescue_seed,
                        runner_version=self.runner_version,
                        engine_step=step_idx,
                        cache_scope=self.cache_scope,
                        warm_start_key=rescue_warm_key,
                    )
                    rescue_signature = ctx.get("signature_hint")
                    if rescue_signature is None and isinstance(rec_meta, dict):
                        rescue_signature = rec_meta.get("model_signature")
                    rescue_exec = execute_run_with_cache_and_artifacts(
                        store=self.store,
                        artifact_store=self.artifact_store,
                        runner=self.runner,
                        run_ctx=rescue_ctx,
                        program=ctx["program"],
                        task_specs=ctx["task_specs"],
                        budget=rescue_budget,
                        eval_contract=self.eval_contract_spec,
                        rng=rescue_seed,
                        runner_init_state_dict=rec.state_dict,
                        runner_init_source=source_label,
                        runner_init_signature=rescue_signature,
                    )
                    rescue_eval = self.evaluator.evaluate(rescue_exec.run_result)

                    episodes.append(
                        ProgramEpisode(
                            program=ctx["program"],
                            run_results=[rescue_exec.run_result],
                            eval_report=rescue_eval,
                            cached=rescue_exec.cached,
                        )
                    )
                    run_ids.append(rescue_ctx.run_id)
                    cached_count += int(rescue_exec.cached)
                    warm_start_meta_by_run[rescue_ctx.run_id] = dict(rec_meta) if isinstance(rec_meta, dict) else None
                    genotype_meta_by_run[rescue_ctx.run_id] = dict(ctx.get("checkpoint_filter") or {})
                    unseen_set_by_run[rescue_ctx.run_id] = ctx.get("current_unseen_set_id")
                    transfer_set_by_run[rescue_ctx.run_id] = ctx.get("current_transfer_set_id")
                    warm_start_elite_by_run[rescue_ctx.run_id] = {
                        "found": True,
                        "score": rec_meta.get("unseen_score") if isinstance(rec_meta, dict) else None,
                        "same_set": bool(
                            ctx.get("current_unseen_set_id") is not None
                            and (rec_meta.get("unseen_set_id") == ctx.get("current_unseen_set_id"))
                        ),
                    }
                    warm_start_relax_by_run[rescue_ctx.run_id] = {
                        "enabled": False,
                        "margin": None,
                        "used": False,
                    }
                    warm_start_audition_by_run[rescue_ctx.run_id] = {
                        "considered": False,
                        "used": True,
                        "win": None,
                        "eligibility": "",
                        "required_for": "",
                        "ineligible_reason": "",
                        "missing_keys": "",
                        "mode": "rescue_injected",
                        "candidate_type": "rescue",
                        "block_reason": "rescue_injected",
                    }
                    warm_start_regime_by_run[rescue_ctx.run_id] = {
                        "fallback": bool(source_label != "checkpoint"),
                        "match": None,
                        "family_match": None,
                    }
                    warm_start_fallback_by_run[rescue_ctx.run_id] = {
                        "considered": False,
                        "used": bool(source_label != "checkpoint"),
                        "blocked_reason": "",
                        "parent_unseen": rec_meta.get("unseen_score") if isinstance(rec_meta, dict) else None,
                        "pool_baseline": None,
                        "delta": None,
                    }
                    rescue_source_used.append(source_label)
                    rescue_supply_source_selected.append(source_label)
                    rescue_candidate_ids.append(str(rec_key))
                    rescue_injected_n += 1

            def _top_fail_reason(default: str) -> str:
                if not rescue_supply_fail_counts:
                    return default
                best_reason = default
                best_count = -1
                for reason, count in rescue_supply_fail_counts.items():
                    if count > best_count:
                        best_reason = reason
                        best_count = count
                return best_reason

            if rescue_triggered:
                if rescue_injected_n > 0:
                    rescue_supply_status = "ok"
                    rescue_supply_fail_reason = ""
                elif rescue_supply_fail_counts.get("supply_error"):
                    rescue_supply_status = "error"
                    rescue_supply_fail_reason = _top_fail_reason("supply_error")
                elif rescue_supply_candidates_seen_n > 0:
                    rescue_supply_status = "filtered"
                    rescue_supply_fail_reason = _top_fail_reason("all_filtered")
                else:
                    rescue_supply_status = "empty"
                    rescue_supply_fail_reason = _top_fail_reason("no_checkpoints")

            reason_parts: List[str] = []
            if no_parent_high:
                reason_parts.append("no_parent_high")
            if quality_reasons:
                reason_parts.extend(quality_reasons)
            if not candidate_dense:
                reason_parts.append("candidate_n_low")
            if not step_ok:
                reason_parts.append("step0")
            if inject_budget <= 0:
                reason_parts.append("budget_exhausted")
            if rescue_triggered and rescue_injected_n == 0:
                if rescue_supply_fail_reason:
                    reason_parts.append(rescue_supply_fail_reason)
                if rescue_supply_status in {"empty", "filtered", "error"}:
                    reason_parts.append(f"supply_{rescue_supply_status}")
                reason_parts.append("no_checkpoint_found")
            if rescue_injected_n > 0:
                reason_parts.append("injected")
            rescue_reason = "+".join(reason_parts)

            if rescue_injected_n > 0:
                rescue_injected = True
                self._rescue_injection_total = int(getattr(self, "_rescue_injection_total", 0)) + int(
                    rescue_injected_n
                )
                if episode_id_for_limit is not None:
                    current_ep_used = int(
                        getattr(self, "_rescue_injection_episode_counts", {}).get(episode_id_for_limit, 0)
                    )
                    self._rescue_injection_episode_counts[episode_id_for_limit] = (
                        current_ep_used + int(rescue_injected_n)
                    )

        # Gate-aware elite selection for distribution updates
        n_elite = max(1, cfg.programs_per_step // 2)
        elites, passed_eps, failed_eps = select_elites_gate_aware(episodes, n_elite)

        batch = MetaBatch(episodes=episodes, dist_snapshot=self.dist.snapshot(), step=step_idx, rng=rng_step, meta=None)
        update_batch = MetaBatch(
            episodes=elites, dist_snapshot=batch.dist_snapshot, step=batch.step, rng=batch.rng, meta=None
        )

        # compute generalization stats from evaluator diagnostics
        scalars = []
        shift_accs = []
        unseen_accs = []
        passes = []
        gate_fail_counts: Dict[str, int] = {}
        train_accs = []
        wall_times = []
        diverged_eps = 0
        nan_inf_eps = 0
        collapse_scalar_threshold = 0.1
        collapse_split_threshold = 0.1
        candidate_debug_rows: List[Dict[str, Any]] = []
        for ep in episodes:
            diag = getattr(ep.eval_report, "diagnostics", {}) or {}
            gen_score = diag.get("generalization_score") or (getattr(ep.eval_report.score, "extra", {}).get("scalar") if hasattr(ep.eval_report, "score") else None)
            if gen_score is not None:
                scalars.append(gen_score)
            split = diag.get("split_metrics") or {}
            run_metrics = {}
            if ep.run_results and isinstance(ep.run_results[0], dict):
                run_metrics = ep.run_results[0].get("metrics", {}) or {}
            train_acc = (split.get("train") or {}).get("accuracy")
            shift_acc = (split.get("shift") or {}).get("accuracy")
            unseen_acc = (split.get("unseen") or {}).get("accuracy")
            transfer_acc = (split.get("transfer") or {}).get("accuracy")
            if train_acc is None:
                train_acc = run_metrics.get("train_accuracy")
            if shift_acc is None:
                shift_acc = run_metrics.get("shift_accuracy")
            if unseen_acc is None:
                unseen_acc = run_metrics.get("unseen_accuracy")
            if transfer_acc is None:
                transfer_acc = run_metrics.get("transfer_accuracy")
            if train_acc is not None:
                train_accs.append(train_acc)
            if shift_acc is not None:
                shift_accs.append(shift_acc)
            if unseen_acc is not None:
                unseen_accs.append(unseen_acc)
            gates_failed = diag.get("gates_failed") or []
            for gf in gates_failed:
                gate_fail_counts[gf] = gate_fail_counts.get(gf, 0) + 1
            passed = diag.get("passed")
            if passed is not None:
                passes.append(bool(passed))
            stability = diag.get("stability") or {}
            diverged_flag = bool(stability.get("diverged"))
            nan_inf_flag = bool(stability.get("nan_inf"))
            if not diverged_flag:
                diverged_flag = bool(run_metrics.get("diverged"))
            if not nan_inf_flag:
                nan_inf_flag = bool(run_metrics.get("nan_inf"))
            diverged_eps += int(diverged_flag)
            nan_inf_eps += int(nan_inf_flag)
            compute = diag.get("compute") or {}
            if "wall_time_s" in compute and compute.get("wall_time_s") is not None:
                wall_times.append(compute.get("wall_time_s"))
            elif run_metrics.get("wall_time_s") is not None:
                wall_times.append(run_metrics.get("wall_time_s"))

            scalar_val = None
            try:
                if gen_score is not None:
                    scalar_val = float(gen_score)
            except Exception:
                scalar_val = None
            candidate_debug_rows.append(
                {
                    "idx": len(candidate_debug_rows),
                    "program_id": ids.program_id(ep.program.graph),
                    "arch": run_metrics.get("effective_arch_type")
                    or _safe_get_by_path(ep.program, "graph.nodes.ARCH:0.spec.type", "unknown"),
                    "warm_start_source": run_metrics.get("warm_start_source"),
                    "scalar": scalar_val,
                    "train_accuracy": train_acc,
                    "shift_accuracy": shift_acc,
                    "unseen_accuracy": unseen_acc,
                    "transfer_accuracy": transfer_acc,
                    "passed": passed,
                    "gates_failed": gates_failed,
                    "diverged": diverged_flag,
                    "nan_inf": nan_inf_flag,
                    "steps": run_metrics.get("steps"),
                    "wall_time_s": run_metrics.get("wall_time_s"),
                }
            )

        best_scalar = max(scalars) if scalars else None
        median_scalar = float(np.median(scalars)) if scalars else None
        best_shift_acc = max(shift_accs) if shift_accs else None
        best_unseen_acc = max(unseen_accs) if unseen_accs else None
        median_shift_acc = float(np.median(shift_accs)) if shift_accs else None
        median_unseen_acc = float(np.median(unseen_accs)) if unseen_accs else None
        median_train_acc = float(np.median(train_accs)) if train_accs else None
        median_wall_time = float(np.median(wall_times)) if wall_times else None
        candidate_health_n = len(candidate_debug_rows)
        candidate_no_parent_n = sum(
            1
            for c in candidate_debug_rows
            if str(c.get("warm_start_source") or "").strip() == "no_parent"
        )
        candidate_no_parent_rate = (
            candidate_no_parent_n / max(1, candidate_health_n)
        )
        cand_family_checkpoint_n = sum(
            1
            for c in candidate_debug_rows
            if str(c.get("warm_start_source") or "").strip().startswith("checkpoint")
        )
        cand_family_stale_n = sum(
            1
            for c in candidate_debug_rows
            if str(c.get("warm_start_source") or "").strip().startswith("stale_")
        )
        cand_family_fallback_n = sum(
            1
            for c in candidate_debug_rows
            if str(c.get("warm_start_source") or "").strip().startswith("fallback_")
        )
        cand_family_scratch_n = sum(
            1
            for c in candidate_debug_rows
            if (
                (src := str(c.get("warm_start_source") or "").strip()) in {
                    "no_parent",
                    "none",
                    "forced_scratch",
                    "forced_scratch_step0",
                    "audition_error_forced_scratch",
                }
                or "scratch" in src
            )
        )
        candidate_scalar_missing_n = sum(1 for c in candidate_debug_rows if c.get("scalar") is None)
        candidate_low_scalar_n = sum(
            1 for c in candidate_debug_rows if c.get("scalar") is not None and float(c.get("scalar")) < collapse_scalar_threshold
        )
        candidate_low_unseen_n = sum(
            1
            for c in candidate_debug_rows
            if c.get("unseen_accuracy") is not None and float(c.get("unseen_accuracy")) < collapse_split_threshold
        )
        candidate_low_shift_n = sum(
            1
            for c in candidate_debug_rows
            if c.get("shift_accuracy") is not None and float(c.get("shift_accuracy")) < collapse_split_threshold
        )
        candidate_low_transfer_n = sum(
            1
            for c in candidate_debug_rows
            if c.get("transfer_accuracy") is not None and float(c.get("transfer_accuracy")) < collapse_split_threshold
        )
        candidate_diverged_n = sum(1 for c in candidate_debug_rows if c.get("diverged"))
        candidate_nan_inf_n = sum(1 for c in candidate_debug_rows if c.get("nan_inf"))
        candidate_gate_fail_any_n = sum(1 for c in candidate_debug_rows if c.get("gates_failed"))
        candidate_passed_true_n = sum(1 for c in candidate_debug_rows if c.get("passed") is True)
        candidate_passed_false_n = sum(1 for c in candidate_debug_rows if c.get("passed") is False)
        collapse_step_flag = bool(median_scalar is not None and median_scalar < collapse_scalar_threshold)
        collapse_candidates_json = (
            json.dumps(candidate_debug_rows, sort_keys=True, separators=(",", ":"))
            if collapse_step_flag
            else ""
        )
        if best_unseen_acc is not None:
            try:
                pool_baselines = getattr(self, "_baseline_unseen_by_pool", {}) or {}
                pool_key = int(unseen_pool_idx)
                prev_best = pool_baselines.get(pool_key)
                curr_best = float(best_unseen_acc)
                if prev_best is None or curr_best > float(prev_best):
                    pool_baselines[pool_key] = curr_best
                self._baseline_unseen_by_pool = pool_baselines
            except Exception:
                pass
        pass_rate = (sum(1 for p in passes if p) / max(1, len(passes))) if passes else None
        elite_pass_rate = sum(1 for ep in elites if getattr(ep.eval_report, "diagnostics", {}).get("passed")) / max(
            1, len(elites)
        )

        best_ep = None
        best_run_result = None
        best_run_id = None
        best_metrics: Dict[str, Any] = {}
        best_idx = None
        best_score = None
        for idx, ep in enumerate(episodes):
            score = _episode_score(ep)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
                best_ep = ep
        if best_ep and best_ep.run_results:
            best_run_result = best_ep.run_results[0]
            if isinstance(best_run_result, dict):
                best_metrics = best_run_result.get("metrics", {}) or {}
        if best_idx is not None and best_idx < len(run_ids):
            best_run_id = run_ids[best_idx]

        best_warm_meta = warm_start_meta_by_run.get(best_run_id) if best_run_id else None
        warm_start_key = ""
        warm_start_step = None
        warm_start_unseen = None
        warm_start_origin_unseen_set_id = None
        warm_start_origin_transfer_set_id = None
        warm_start_origin_regime_id = None
        warm_start_origin_regime_family_id = None
        if isinstance(best_warm_meta, dict):
            warm_start_key = str(best_warm_meta.get("checkpoint_id") or best_warm_meta.get("path") or "")
            warm_start_step = best_warm_meta.get("step")
            warm_start_unseen = best_warm_meta.get("unseen_score")
            warm_start_origin_unseen_set_id = best_warm_meta.get("unseen_set_id")
            warm_start_origin_transfer_set_id = best_warm_meta.get("transfer_set_id")
            warm_start_origin_regime_id = best_warm_meta.get("regime_id")
            warm_start_origin_regime_family_id = best_warm_meta.get("regime_family_id")

        warm_start_used = None
        warm_start_source = None
        warm_start_mismatch = None
        warm_start_signature = None
        if isinstance(best_metrics, dict) and best_metrics:
            warm_start_used = best_metrics.get("warm_start_used")
            warm_start_source = best_metrics.get("warm_start_source")
            warm_start_mismatch = best_metrics.get("warm_start_mismatch")
            warm_start_signature = best_metrics.get("warm_start_signature")
        best_transfer_acc = None
        if isinstance(best_metrics, dict):
            best_transfer_acc = best_metrics.get("transfer_accuracy")

        current_unseen_set_id = unseen_set_by_run.get(best_run_id) if best_run_id else None
        current_transfer_set_id = transfer_set_by_run.get(best_run_id) if best_run_id else None
        warm_start_skip = warm_start_skip_by_run.get(best_run_id, {})
        warm_start_skipped_stale = bool(warm_start_skip.get("skipped"))
        warm_start_skip_reason = warm_start_skip.get("reason") if warm_start_skip.get("skipped") else ""
        warm_start_skip_key = warm_start_skip.get("key") if warm_start_skip.get("skipped") else ""
        warm_start_elite = warm_start_elite_by_run.get(best_run_id, {}) if best_run_id else {}
        warm_start_relax = warm_start_relax_by_run.get(best_run_id, {}) if best_run_id else {}
        warm_start_audition = warm_start_audition_by_run.get(best_run_id, {}) if best_run_id else {}
        warm_start_regime = warm_start_regime_by_run.get(best_run_id, {}) if best_run_id else {}
        warm_start_fallback = warm_start_fallback_by_run.get(best_run_id, {}) if best_run_id else {}
        warm_start_elite_used = warm_start_source in {
            "checkpoint",
            "fallback_checkpoint",
            "stale_checkpoint",
        }
        warm_start_elite_score = warm_start_elite.get("score")
        warm_start_elite_same_set = bool(warm_start_elite.get("same_set"))
        warm_start_relaxed_enabled = bool(warm_start_relax.get("enabled"))
        warm_start_relaxed_margin = warm_start_relax.get("margin")
        warm_start_relaxed_used = bool(warm_start_relax.get("used"))
        warm_start_audition_considered = bool(warm_start_audition.get("considered"))
        warm_start_audition_used = bool(warm_start_audition.get("used"))
        warm_start_audition_win = warm_start_audition.get("win")
        warm_start_audition_delta_unseen = warm_start_audition.get("delta_unseen")
        warm_start_audition_fallback_unseen = warm_start_audition.get("fallback_unseen")
        warm_start_audition_scratch_unseen = warm_start_audition.get("scratch_unseen")
        warm_start_audition_margin = warm_start_audition.get("margin")
        warm_start_audition_steps = warm_start_audition.get("steps")
        warm_start_audition_block_reason = warm_start_audition.get("block_reason") or ""
        warm_start_audition_unseen_set_match = warm_start_audition.get("unseen_set_match")
        warm_start_audition_checkpoint_override = bool(warm_start_audition.get("checkpoint_override"))
        warm_start_audition_checkpoint_override_tier = warm_start_audition.get("checkpoint_override_tier") or ""
        warm_start_audition_override_mean_eps = warm_start_audition.get("checkpoint_override_gain")
        warm_start_audition_override_parent_eps = warm_start_audition.get("checkpoint_override_parent_gain")
        warm_start_audition_override_rescue_mean_eps = warm_start_audition.get("checkpoint_override_rescue_gain")
        warm_start_audition_override_rescue_parent_eps = warm_start_audition.get("checkpoint_override_rescue_parent_gain")
        warm_start_audition_override_max_regress = warm_start_audition.get("checkpoint_override_max_regress")
        warm_start_audition_override_transfer_eps = warm_start_audition.get("checkpoint_override_transfer_gain")
        warm_start_audition_override_transfer_gain_eps = warm_start_audition.get(
            "checkpoint_override_transfer_gain_eps"
        )
        warm_start_audition_override_rescue_transfer_eps = warm_start_audition.get(
            "checkpoint_override_rescue_transfer_gain"
        )
        warm_start_audition_override_rescue_transfer_gain_eps = warm_start_audition.get(
            "checkpoint_override_rescue_transfer_gain_eps"
        )
        warm_start_audition_override_tradeoff_lambda = warm_start_audition.get(
            "checkpoint_override_tradeoff_lambda"
        )
        warm_start_audition_override_tradeoff_min = warm_start_audition.get(
            "checkpoint_override_tradeoff_min"
        )
        warm_start_audition_override_parent_tie_tolerance = warm_start_audition.get(
            "checkpoint_override_parent_tie_tolerance"
        )
        warm_start_audition_override_parent_score_tolerance = warm_start_audition.get(
            "checkpoint_override_parent_score_tolerance"
        )
        warm_start_audition_override_transfer_tie_primary_min = warm_start_audition.get(
            "checkpoint_override_transfer_tie_primary_min"
        )
        warm_start_audition_override_parent_regress_transfer_floor = warm_start_audition.get(
            "checkpoint_override_parent_regress_transfer_floor"
        )
        if warm_start_audition_override_transfer_eps is None:
            warm_start_audition_override_transfer_eps = checkpoint_override_transfer_gain
        if warm_start_audition_override_transfer_gain_eps is None:
            warm_start_audition_override_transfer_gain_eps = checkpoint_override_transfer_gain_eps
        if warm_start_audition_override_rescue_transfer_eps is None:
            warm_start_audition_override_rescue_transfer_eps = checkpoint_override_rescue_transfer_gain
        if warm_start_audition_override_rescue_transfer_gain_eps is None:
            warm_start_audition_override_rescue_transfer_gain_eps = checkpoint_override_rescue_transfer_gain_eps
        if warm_start_audition_override_tradeoff_lambda is None:
            warm_start_audition_override_tradeoff_lambda = checkpoint_override_tradeoff_lambda
        if warm_start_audition_override_tradeoff_min is None:
            warm_start_audition_override_tradeoff_min = checkpoint_override_tradeoff_min
        if warm_start_audition_override_parent_tie_tolerance is None:
            warm_start_audition_override_parent_tie_tolerance = checkpoint_override_parent_tie_tolerance
        if warm_start_audition_override_parent_score_tolerance is None:
            warm_start_audition_override_parent_score_tolerance = checkpoint_override_parent_score_tolerance
        if warm_start_audition_override_transfer_tie_primary_min is None:
            warm_start_audition_override_transfer_tie_primary_min = checkpoint_override_transfer_tie_primary_min
        if warm_start_audition_override_parent_regress_transfer_floor is None:
            warm_start_audition_override_parent_regress_transfer_floor = (
                checkpoint_override_parent_regress_transfer_floor
            )
        warm_start_audition_strict_allow_reason = warm_start_audition.get("strict_allow_reason") or ""
        warm_start_audition_accept_mode = warm_start_audition.get("accept_mode") or ""
        warm_start_audition_strict_score = warm_start_audition.get("strict_score")
        warm_start_audition_parent_unseen = warm_start_audition.get("parent_unseen")
        warm_start_audition_parent_delta = warm_start_audition.get("parent_delta")
        warm_start_audition_probe_unseen = warm_start_audition.get("probe_unseen")
        warm_start_audition_transfer_gate_pass = warm_start_audition.get("transfer_gate_pass")
        warm_start_audition_transfer_gate_tier = warm_start_audition.get("transfer_gate_tier") or ""
        warm_start_audition_transfer_parent_unseen = warm_start_audition.get("transfer_parent_unseen")
        warm_start_audition_transfer_probe_unseen = warm_start_audition.get("transfer_probe_unseen")
        warm_start_audition_transfer_delta = warm_start_audition.get("transfer_delta")
        warm_start_audition_error_code = warm_start_audition.get("error_code") or ""
        warm_start_audition_error_msg = warm_start_audition.get("error_msg") or ""
        warm_start_audition_error_where = warm_start_audition.get("error_where") or ""
        warm_start_audition_eligibility = warm_start_audition.get("eligibility") or ""
        warm_start_audition_required_for = warm_start_audition.get("required_for") or ""
        warm_start_audition_ineligible_reason = warm_start_audition.get("ineligible_reason") or ""
        warm_start_audition_missing_keys = warm_start_audition.get("missing_keys") or ""
        warm_start_audition_candidate_type = warm_start_audition.get("candidate_type") or ""
        warm_start_paired_gain_mean = warm_start_audition.get("mean_delta")
        warm_start_paired_gain_min = warm_start_audition.get("min_delta")
        warm_start_paired_gain_used = bool(
            warm_start_used
            and warm_start_audition_considered
            and warm_start_audition_win
            and warm_start_audition_unseen_set_match
        )
        warm_start_paired_gain_comparable = bool(
            warm_start_audition_considered and warm_start_audition_unseen_set_match
        )
        warm_start_regime_fallback = bool(warm_start_regime.get("fallback"))
        warm_start_regime_match = warm_start_regime.get("match")
        warm_start_family_match = warm_start_regime.get("family_match")
        warm_start_fallback_considered = bool(warm_start_fallback.get("considered"))
        warm_start_fallback_used = bool(warm_start_fallback.get("used"))
        warm_start_fallback_family_gate_passed = bool(
            warm_start_regime_fallback and warm_start_family_match and warm_start_fallback_used
        )
        warm_start_fallback_block_reason = warm_start_fallback.get("blocked_reason")
        warm_start_fallback_parent_unseen = warm_start_fallback.get("parent_unseen")
        warm_start_fallback_pool_baseline = warm_start_fallback.get("pool_baseline")
        warm_start_fallback_delta = warm_start_fallback.get("delta")

        transfer_delta_obs = None
        mean_delta_obs = None
        try:
            if warm_start_audition_transfer_delta is not None:
                transfer_delta_obs = float(warm_start_audition_transfer_delta)
        except Exception:
            transfer_delta_obs = None
        try:
            if warm_start_paired_gain_mean is not None:
                mean_delta_obs = float(warm_start_paired_gain_mean)
        except Exception:
            mean_delta_obs = None
        trusted_override_obs = bool(
            warm_start_audition_checkpoint_override and warm_start_audition_unseen_set_match
        )
        gene_matured = self._override_gene_bandit.observe_step(
            step_idx=step_idx,
            trusted_override=trusted_override_obs,
            transfer_delta=transfer_delta_obs,
            mean_delta=mean_delta_obs,
            block_reason=warm_start_audition_block_reason,
            collapse_step=bool(collapse_step_flag),
            current_primary=best_unseen_acc,
            current_transfer=best_transfer_acc,
        )

        override_gene_id = active_override_gene.gene_id
        override_gene_lam = float(active_override_gene.lam)
        override_gene_parent_tol = float(active_override_gene.parent_tolerance)
        override_gene_transfer_floor = float(active_override_gene.transfer_floor)
        override_gene_tie_primary_min = float(active_override_gene.tie_primary_min)
        gene_episode_id = gene_state.get("episode_id")
        gene_episode_step0 = gene_state.get("episode_start")
        gene_selected_by = gene_selection_meta.get("selected_by")
        gene_ucb_score = gene_selection_meta.get("ucb_score")
        gene_mean_reward_before = gene_selection_meta.get("mean_reward_before")
        gene_n_before = gene_selection_meta.get("n_before")
        gene_episode_gene_id = ""
        gene_episode_step1 = None
        gene_episode_reward = None
        gene_episode_reward_transfer_component = None
        gene_episode_reward_general_component = None
        gene_episode_reliable = None
        gene_episode_n_trusted = None
        gene_episode_n_events = None
        gene_episode_collapse_steps = None
        gene_episode_transfer_proxy_mean = None
        gene_episode_general_proxy_rate = None
        gene_episode_matured_count = None
        gene_episode_matured_step = None
        gene_episode_matured_id = None
        gene_episode_matured_gene_id = ""
        gene_episode_proxy_reward = None
        gene_episode_delayed_reward = None
        gene_episode_reward_used = None
        gene_episode_delayed_reliable = None
        gene_episode_matured_reliable = None
        gene_episode_primary_survival = None
        gene_episode_transfer_survival = None
        gene_rollback = False
        gene_rollback_reason = ""
        if isinstance(gene_finalized, OverrideGeneEpisodeSummary):
            gene_episode_id = gene_finalized.episode_id
            gene_episode_step0 = gene_finalized.step0
            gene_episode_gene_id = gene_finalized.gene_id
            gene_episode_step1 = gene_finalized.step1
            gene_episode_reward = gene_finalized.reward
            gene_episode_reward_transfer_component = gene_finalized.reward_transfer_component
            gene_episode_reward_general_component = gene_finalized.reward_general_component
            gene_episode_reliable = gene_finalized.reliable
            gene_episode_n_trusted = gene_finalized.n_trusted
            gene_episode_n_events = gene_finalized.n_events
            gene_episode_collapse_steps = gene_finalized.collapse_steps
            gene_episode_transfer_proxy_mean = gene_finalized.transfer_proxy_mean
            gene_episode_general_proxy_rate = gene_finalized.general_proxy_rate
            gene_rollback = gene_finalized.rollback
            gene_rollback_reason = gene_finalized.rollback_reason
        if gene_matured:
            latest_matured = gene_matured[-1]
            gene_episode_matured_count = len(gene_matured)
            gene_episode_matured_step = latest_matured.matured_step
            gene_episode_matured_id = latest_matured.episode_id
            gene_episode_matured_gene_id = latest_matured.gene_id
            gene_episode_proxy_reward = latest_matured.proxy_reward
            gene_episode_delayed_reward = latest_matured.delayed_reward
            gene_episode_reward_used = latest_matured.reward_used
            gene_episode_delayed_reliable = latest_matured.delayed_reliable
            gene_episode_matured_reliable = latest_matured.reliable
            gene_episode_primary_survival = latest_matured.primary_survival
            gene_episode_transfer_survival = latest_matured.transfer_survival

        forced_scratch = bool(warm_start_source == "forced_scratch")
        warm_start_origin_step = warm_start_step
        warm_start_origin_unseen = warm_start_unseen
        lineage_gain = None
        lineage_gain_comparable = False
        if (
            warm_start_used
            and warm_start_origin_unseen is not None
            and best_unseen_acc is not None
            and warm_start_origin_unseen_set_id is not None
            and current_unseen_set_id is not None
            and warm_start_origin_unseen_set_id == current_unseen_set_id
        ):
            lineage_gain_comparable = True
            try:
                lineage_gain = float(best_unseen_acc) - float(warm_start_origin_unseen)
            except Exception:
                lineage_gain = None

        best_arch_type = "unknown"
        if best_ep is not None:
            try:
                best_arch_type = str(best_ep.program.graph.nodes["ARCH:0"].spec.get("type", "unknown"))
            except Exception:
                best_arch_type = "unknown"

        best_state_dict = None
        best_model_signature = None
        best_effective_arch_type = best_arch_type
        if isinstance(best_run_result, dict):
            best_model_signature = best_run_result.get("model_signature") or best_metrics.get("model_signature")
            best_effective_arch_type = (
                best_run_result.get("effective_arch_type")
                or best_metrics.get("effective_arch_type")
                or best_arch_type
            )
            state_dict = best_run_result.get("state_dict")
            if isinstance(state_dict, dict) and state_dict:
                best_state_dict = state_dict


        best_program_id = None
        if best_ep is not None:
            try:
                best_program_id = ids.program_id(best_ep.program.graph)
            except Exception:
                best_program_id = None

        best_geno = genotype_meta_by_run.get(best_run_id, {}) if best_run_id else {}
        best_opt_type = best_geno.get("opt_type")
        best_obj_primary = best_geno.get("obj_primary")
        best_lr_bin = best_geno.get("lr_bin")
        best_regime_id = best_geno.get("regime_id")
        best_regime_family_id = best_geno.get("regime_family_id")

        arch_scores: Dict[str, float] = {}
        arch_counts: Dict[str, int] = {}
        for ep in episodes:
            diag = getattr(ep.eval_report, "diagnostics", {}) or {}
            gen_score = diag.get("generalization_score") or (
                getattr(ep.eval_report.score, "extra", {}).get("scalar") if hasattr(ep.eval_report, "score") else None
            )
            if gen_score is None:
                continue
            try:
                arch = str(ep.program.graph.nodes["ARCH:0"].spec.get("type", "unknown"))
            except Exception:
                arch = "unknown"
            delta = float(gen_score) - float(median_scalar if median_scalar is not None else gen_score)
            arch_scores[arch] = arch_scores.get(arch, 0.0) + delta
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
        for arch, total in arch_scores.items():
            avg = total / max(1, arch_counts.get(arch, 1))
            self.q_search.interference_update(arch, avg)
        if best_unseen_acc is not None:
            self.q_search.apply_uncertainty(float(best_unseen_acc))
        if hasattr(self.dist, "set_arch_bias"):
            try:
                self.dist.set_arch_bias(self.q_search.probabilities())
            except Exception:
                pass
        arch_probs = self.q_search.probabilities()

        # coupling stats for elites: (max_len_train, budget_steps)
        def _get_len_steps(ep: ProgramEpisode) -> Tuple[Optional[int], Optional[int]]:
            lval = None
            sval = None
            try:
                curr_spec = ep.program.graph.nodes.get("CURR:0", None)
                if curr_spec and isinstance(curr_spec.spec, dict):
                    lval = curr_spec.spec.get("curriculum", {}).get("max_len_train")
            except Exception:
                pass
            try:
                if hasattr(ep.program, "meta") and isinstance(ep.program.meta, dict):
                    sval = ep.program.meta.get("budget_steps")
            except Exception:
                pass
            try:
                if lval is not None:
                    lval = int(lval)
                if sval is not None:
                    sval = int(sval)
            except Exception:
                lval = lval
                sval = sval
            return lval, sval

        pairs = []
        for ep in elites:
            lval, sval = _get_len_steps(ep)
            if lval is not None and sval is not None:
                pairs.append((lval, sval))
        elite_coupling_counts: Dict[str, int] = {}
        if pairs:
            for l, s in pairs:
                key = f"{l}x{s}"
                elite_coupling_counts[key] = elite_coupling_counts.get(key, 0) + 1
        # simple Pearson corr if we have enough pairs
        elite_len_steps_corr = None
        if len(pairs) >= 2:
            arr = np.array(pairs, dtype=float)
            lvec = arr[:, 0]
            svec = arr[:, 1]
            lv = lvec - lvec.mean()
            sv = svec - svec.mean()
            denom = (np.linalg.norm(lv) * np.linalg.norm(sv))
            if denom > 0:
                elite_len_steps_corr = float(lv.dot(sv) / denom)

        meta_payload = {
            "runner_version": self.runner_version,
            "engine_instrumentation_version": ENGINE_INSTRUMENTATION_VERSION,
            "programs": len(episodes),
            "cached_runs": cached_count,
            "total_runs": len(episodes),
            "cache_rate": cached_count / max(1, len(episodes)),
            # fresh_run_ratio: fraction of runs this step that were not cached
            "fresh_run_ratio": (len(episodes) - cached_count) / max(1, len(episodes)),
            "entropy_before": entropy_before,
            "rml_planck_h": current_planck,
            "best_scalar": best_scalar,
            "median_scalar": median_scalar,
            "best_generalization_score": best_scalar,
            "median_generalization_score": median_scalar,
                "best_shift_accuracy": best_shift_acc,
                "best_unseen_accuracy": best_unseen_acc,
                "transfer_unseen_accuracy": best_transfer_acc,
                "median_shift_accuracy": median_shift_acc,
                "median_unseen_accuracy": median_unseen_acc,
            "median_train_accuracy": median_train_acc,
            "median_wall_time_s": median_wall_time,
            "pass_rate": pass_rate,
            "elite_pass_rate": elite_pass_rate,
            "gate_fail_counts": gate_fail_counts,
            "elite_coupling_counts": elite_coupling_counts,
            "elite_len_steps_corr": elite_len_steps_corr,
            "invalid_retries_total": getattr(self.dist, "_last_retry_stats", {}).get("invalid_retries") if hasattr(self.dist, "_last_retry_stats") else None,
            "retry_rate": None,
            "collapse_step_flag": collapse_step_flag,
            "collapse_scalar_threshold": collapse_scalar_threshold,
            "collapse_split_threshold": collapse_split_threshold,
            "candidate_health_n": candidate_health_n,
            "candidate_no_parent_n": candidate_no_parent_n,
            "candidate_no_parent_rate": candidate_no_parent_rate,
            "cand_family_checkpoint_n": cand_family_checkpoint_n,
            "cand_family_stale_n": cand_family_stale_n,
            "cand_family_fallback_n": cand_family_fallback_n,
            "cand_family_scratch_n": cand_family_scratch_n,
            "candidate_scalar_missing_n": candidate_scalar_missing_n,
            "candidate_low_scalar_n": candidate_low_scalar_n,
            "candidate_low_unseen_n": candidate_low_unseen_n,
            "candidate_low_shift_n": candidate_low_shift_n,
            "candidate_low_transfer_n": candidate_low_transfer_n,
            "candidate_diverged_n": candidate_diverged_n,
            "candidate_nan_inf_n": candidate_nan_inf_n,
            "candidate_gate_fail_any_n": candidate_gate_fail_any_n,
            "candidate_passed_true_n": candidate_passed_true_n,
            "candidate_passed_false_n": candidate_passed_false_n,
            "rescue_enable": bool(getattr(cfg, "rescue_enable", False)),
            "rescue_injected": rescue_injected,
            "rescue_triggered": rescue_triggered,
            "rescue_reason": rescue_reason,
            "rescue_no_parent_rate_threshold": float(getattr(cfg, "rescue_no_parent_rate", 0.66)),
            "rescue_best_floor": float(getattr(cfg, "rescue_best_floor", 0.12)),
            "rescue_low_split_n": int(getattr(cfg, "rescue_low_split_n", 8)),
            "rescue_inject_n": int(getattr(cfg, "rescue_inject_n", 1)),
            "rescue_max_per_run": int(getattr(cfg, "rescue_max_per_run", 10)),
            "rescue_max_per_episode": int(getattr(cfg, "rescue_max_per_episode", 2)),
            "rescue_injected_n": int(rescue_injected_n),
            "rescue_source_used": "|".join(rescue_source_used),
            "rescue_candidate_ids": "|".join(rescue_candidate_ids),
            "rescue_supply_status": rescue_supply_status,
            "rescue_supply_fail_reason": rescue_supply_fail_reason,
            "rescue_supply_candidates_seen_n": int(rescue_supply_candidates_seen_n),
            "rescue_supply_attempts_n": int(rescue_supply_attempts_n),
            "rescue_supply_source_selected": "|".join(rescue_supply_source_selected),
            "rescue_no_parent_rate_observed": rescue_no_parent_rate_observed,
            "rescue_candidate_n_observed": rescue_candidate_n_observed,
            "rescue_best_observed": rescue_best_observed,
            "rescue_collapse_observed": rescue_collapse_observed,
            "rescue_low_split_observed": rescue_low_split_observed,
            "rescue_injection_total": int(getattr(self, "_rescue_injection_total", 0)),
            "collapse_candidates_json": collapse_candidates_json,
            "cache_scope": self.cache_scope,
            "marginals": getattr(self.dist, "top_marginals", lambda v: {})(["ARCH.type", "LRULE.type", "OBJ.primary", "CURR.curriculum.max_len_train", "BUDGET.steps"]),
            "arch_probs": arch_probs,
            "retry_reasons": getattr(self.dist, "_last_retry_stats", {}).get("reasons") if hasattr(self.dist, "_last_retry_stats") else None,
            "obj_prior": getattr(self.dist, "obj_prior", None),
            "eval_contract": self.eval_contract_spec,
            "headline_metric": "generalization_score",
            "baseline_ref": "rolling_best_20",
            "eval_contract_version": self.eval_contract_spec.get("version") if isinstance(self.eval_contract_spec, dict) else None,
            "diverged_episodes": diverged_eps,
            "nan_inf_episodes": nan_inf_eps,
        }
        meta_payload.update(ab_probe_meta)
        meta_payload.update(
            {
                "audition_any_considered": audition_considered_count > 0,
                "audition_any_used": audition_used_count > 0,
                "audition_any_win": audition_win_count > 0,
                "audition_considered_count": audition_considered_count,
                "audition_used_count": audition_used_count,
                "audition_win_count": audition_win_count,
            }
        )

        decision = self.progress.should_accept(step_idx, meta_payload)
        tunneled = False
        tunnel_prob = None
        tunnel_prob_effective = None
        inherit_strict_gate = False
        if not decision.get("accepted"):
            baseline = (decision.get("baseline") or {}).get("baseline_generalization")
            if baseline is None:
                baseline = (decision.get("baseline") or {}).get("baseline_unseen")
            if baseline is None:
                baseline = (decision.get("baseline") or {}).get("best_unseen")
            cand = best_scalar
            if baseline is not None and cand is not None:
                tunnel_prob = self.q_state.tunneling_probability(float(baseline), float(cand), barrier_width=1.0)
                tunnel_prob_effective = tunnel_prob
                if warm_start_used and decision.get("reason") == "no_generalization_gain":
                    inherit_strict_gate = True
                    tunnel_prob_effective = 0.0
                rng_local = random.Random(self._split_rng(rng_step, step_idx, "tunnel"))
                tunneled = self.q_state.observe(tunnel_prob_effective, rng=rng_local)
                if tunneled:
                    decision = dict(decision)
                    decision["accepted"] = True
                    decision["reason"] = "tunnel_accept"
                    decision["tunnel_prob"] = tunnel_prob_effective
                    self.progress.force_accept(step_idx, meta_payload)

        checkpoint_path = ""
        checkpoint_id = ""
        if decision.get("accepted"):
            if best_state_dict and best_model_signature:
                ck_meta: Dict[str, Any] = {
                    "step": step_idx,
                    "program_id": best_program_id,
                    "run_id": best_run_id,
                    "arch_type": best_effective_arch_type,
                    "model_signature": best_model_signature,
                    "unseen_score": best_unseen_acc,
                    "shift_score": best_shift_acc,
                    "transfer_unseen_accuracy": best_transfer_acc,
                    "unseen_set_id": current_unseen_set_id,
                    "transfer_set_id": current_transfer_set_id,
                    "unseen_pool_idx": unseen_pool_idx,
                }
                ck_meta["opt_type"] = best_opt_type if best_opt_type is not None else "unknown"
                ck_meta["obj_primary"] = best_obj_primary if best_obj_primary is not None else "unknown"
                ck_meta["lr_bin"] = best_lr_bin if best_lr_bin is not None else "unknown"
                ck_meta["regime_id"] = best_regime_id if best_regime_id is not None else "unknown"
                ck_meta["regime_family_id"] = best_regime_family_id if best_regime_family_id is not None else "unknown"
                if warm_start_used is not None:
                    ck_meta["warm_start_used"] = bool(warm_start_used)
                if warm_start_source:
                    ck_meta["warm_start_source"] = warm_start_source
                if warm_start_mismatch:
                    ck_meta["warm_start_mismatch"] = warm_start_mismatch
                if warm_start_signature:
                    ck_meta["warm_start_signature"] = warm_start_signature
                try:
                    ck_path = self.checkpoints.save(
                        best_effective_arch_type, str(best_model_signature), best_state_dict, ck_meta
                    )
                    checkpoint_path = str(ck_path)
                    checkpoint_id = ck_path.name
                    try:
                        pool_elites = getattr(self, "_pool_elite_cache", {}) or {}
                        pool_elites_by_set = getattr(self, "_pool_elite_cache_by_set", {}) or {}
                        sig_key = str(best_model_signature) if best_model_signature is not None else "unknown"
                        elite_key = (
                            best_effective_arch_type,
                            int(unseen_pool_idx),
                            str(best_regime_id if best_regime_id is not None else "unknown"),
                            sig_key,
                        )
                        meta_for_cache = dict(ck_meta)
                        meta_for_cache["checkpoint_id"] = checkpoint_id
                        meta_for_cache["path"] = checkpoint_path
                        payload = {
                            "path": checkpoint_path,
                            "meta": meta_for_cache,
                            "state_dict": best_state_dict,
                            "unseen_score": best_unseen_acc,
                            "unseen_set_id": current_unseen_set_id,
                        }
                        def _score(v):
                            try:
                                return float(v)
                            except Exception:
                                return None
                        new_score = _score(best_unseen_acc)
                        existing = pool_elites.get(elite_key)
                        existing_score = _score(existing.get("unseen_score")) if isinstance(existing, dict) else None
                        if existing is None or (new_score is not None and (existing_score is None or new_score > existing_score)):
                            pool_elites[elite_key] = payload
                        if current_unseen_set_id is not None:
                            set_key = (elite_key, current_unseen_set_id)
                            existing_set = pool_elites_by_set.get(set_key)
                            existing_set_score = _score(existing_set.get("unseen_score")) if isinstance(existing_set, dict) else None
                            if existing_set is None or (
                                new_score is not None and (existing_set_score is None or new_score > existing_set_score)
                            ):
                                pool_elites_by_set[set_key] = payload
                        self._pool_elite_cache = pool_elites
                        self._pool_elite_cache_by_set = pool_elites_by_set
                    except Exception:
                        pass
                except Exception:
                    checkpoint_path = ""
                    checkpoint_id = ""

        if decision.get("accepted") and hasattr(self.dist, "update"):
            self.dist.update(update_batch)
        else:
            if hasattr(self.dist, "encourage_exploration"):
                try:
                    self.dist.encourage_exploration()
                except Exception:
                    pass
        entropy_after = getattr(self.dist, "entropy", lambda: None)()

        meta_payload.update(
            {
                "entropy_after": entropy_after,
                "rml_decision": decision,
                "rml_accept": decision.get("accepted"),
                "rml_reason": decision.get("reason"),
                "rml_tunnel": bool(tunneled),
                "rml_tunnel_prob": tunnel_prob,
                "tunnel_prob_effective": tunnel_prob_effective,
                "inherit_strict_gate": inherit_strict_gate,
                "arch_requested": best_arch_type,
                "arch_effective": best_effective_arch_type,
                "unseen_pool_idx": unseen_pool_idx,
                "current_transfer_set_id": current_transfer_set_id,
                "warm_start_used": warm_start_used,
                "warm_start_source": warm_start_source,
                "warm_start_mismatch": warm_start_mismatch,
                "warm_start_signature": warm_start_signature,
                "warm_start_key": warm_start_key or "",
                "warm_start_step": warm_start_step,
                "warm_start_unseen": warm_start_unseen,
                "forced_scratch": forced_scratch,
                "warm_start_origin_step": warm_start_origin_step,
                "warm_start_origin_unseen": warm_start_origin_unseen,
                "warm_start_origin_unseen_set_id": warm_start_origin_unseen_set_id,
                "warm_start_origin_transfer_set_id": warm_start_origin_transfer_set_id,
                "current_unseen_set_id": current_unseen_set_id,
                "lineage_gain": lineage_gain,
                "lineage_gain_comparable": lineage_gain_comparable,
                "regime_id": best_regime_id,
                "regime_family_id": best_regime_family_id,
                "warm_start_origin_regime_id": warm_start_origin_regime_id,
                "warm_start_origin_regime_family_id": warm_start_origin_regime_family_id,
                "warm_start_skipped_stale": warm_start_skipped_stale,
                "warm_start_skip_reason": warm_start_skip_reason,
                "warm_start_skip_key": warm_start_skip_key,
                "warm_start_elite_used": warm_start_elite_used,
                "warm_start_elite_score": warm_start_elite_score,
                "warm_start_elite_same_set": warm_start_elite_same_set,
                "warm_start_relaxed": warm_start_relaxed_enabled,
                "warm_start_relaxed_margin": warm_start_relaxed_margin,
                "warm_start_relaxed_used": warm_start_relaxed_used,
                "warm_start_audition_considered": warm_start_audition_considered,
                "warm_start_audition_used": warm_start_audition_used,
                "warm_start_audition_win": warm_start_audition_win,
                "warm_start_audition_delta_unseen": warm_start_audition_delta_unseen,
                "warm_start_audition_delta_unseen_b": warm_start_audition.get("delta_unseen_b"),
                "warm_start_audition_mean_delta": warm_start_audition.get("mean_delta"),
                "warm_start_audition_min_delta": warm_start_audition.get("min_delta"),
            "warm_start_audition_fallback_unseen": warm_start_audition_fallback_unseen,
            "warm_start_audition_fallback_unseen_b": warm_start_audition.get("fallback_unseen_b"),
            "warm_start_audition_scratch_unseen": warm_start_audition_scratch_unseen,
            "warm_start_audition_scratch_unseen_b": warm_start_audition.get("scratch_unseen_b"),
            "warm_start_audition_mode": warm_start_audition.get("mode"),
            "warm_start_audition_margin": warm_start_audition_margin,
            "warm_start_audition_steps": warm_start_audition_steps,
            "warm_start_audition_block_reason": warm_start_audition_block_reason,
            "warm_start_audition_eligibility": warm_start_audition_eligibility,
            "warm_start_audition_required_for": warm_start_audition_required_for,
            "warm_start_audition_ineligible_reason": warm_start_audition_ineligible_reason,
            "warm_start_audition_missing_keys": warm_start_audition_missing_keys,
            "warm_start_audition_candidate_type": warm_start_audition_candidate_type,
            "warm_start_audition_unseen_set_match": warm_start_audition_unseen_set_match,
            "warm_start_audition_checkpoint_override": warm_start_audition_checkpoint_override,
            "warm_start_audition_checkpoint_override_tier": warm_start_audition_checkpoint_override_tier,
            "warm_start_audition_override_mean_eps": warm_start_audition_override_mean_eps,
                "warm_start_audition_override_parent_eps": warm_start_audition_override_parent_eps,
                "warm_start_audition_override_rescue_mean_eps": warm_start_audition_override_rescue_mean_eps,
                "warm_start_audition_override_rescue_parent_eps": warm_start_audition_override_rescue_parent_eps,
                "warm_start_audition_override_max_regress": warm_start_audition_override_max_regress,
                "warm_start_audition_override_transfer_eps": warm_start_audition_override_transfer_eps,
                "warm_start_audition_override_transfer_gain_eps": warm_start_audition_override_transfer_gain_eps,
                "warm_start_audition_override_rescue_transfer_eps": warm_start_audition_override_rescue_transfer_eps,
                "warm_start_audition_override_rescue_transfer_gain_eps": warm_start_audition_override_rescue_transfer_gain_eps,
                "warm_start_audition_override_tradeoff_lambda": warm_start_audition_override_tradeoff_lambda,
                "warm_start_audition_override_tradeoff_min": warm_start_audition_override_tradeoff_min,
                "warm_start_audition_override_parent_tie_tolerance": warm_start_audition_override_parent_tie_tolerance,
                "warm_start_audition_override_parent_score_tolerance": warm_start_audition_override_parent_score_tolerance,
                "warm_start_audition_override_transfer_tie_primary_min": warm_start_audition_override_transfer_tie_primary_min,
                "warm_start_audition_override_parent_regress_transfer_floor": warm_start_audition_override_parent_regress_transfer_floor,
                "warm_start_audition_strict_allow_reason": warm_start_audition_strict_allow_reason,
                "warm_start_audition_accept_mode": warm_start_audition_accept_mode,
                "warm_start_audition_strict_score": warm_start_audition_strict_score,
                "warm_start_audition_parent_unseen": warm_start_audition_parent_unseen,
                "warm_start_audition_parent_delta": warm_start_audition_parent_delta,
                "warm_start_audition_probe_unseen": warm_start_audition_probe_unseen,
                "warm_start_audition_transfer_gate_pass": warm_start_audition_transfer_gate_pass,
                "warm_start_audition_transfer_gate_tier": warm_start_audition_transfer_gate_tier,
                "warm_start_audition_transfer_parent_unseen": warm_start_audition_transfer_parent_unseen,
                "warm_start_audition_transfer_probe_unseen": warm_start_audition_transfer_probe_unseen,
                "warm_start_audition_transfer_delta": warm_start_audition_transfer_delta,
                "warm_start_audition_error_code": warm_start_audition_error_code,
                "warm_start_audition_error_msg": warm_start_audition_error_msg,
                "warm_start_audition_error_where": warm_start_audition_error_where,
            "warm_start_paired_gain_mean": warm_start_paired_gain_mean,
            "warm_start_paired_gain_min": warm_start_paired_gain_min,
            "warm_start_paired_gain_used": warm_start_paired_gain_used,
            "warm_start_paired_gain_comparable": warm_start_paired_gain_comparable,
            "warm_start_regime_fallback": warm_start_regime_fallback,
                "warm_start_regime_match": warm_start_regime_match,
                "warm_start_family_match": warm_start_family_match,
                "warm_start_fallback_family_gate_passed": warm_start_fallback_family_gate_passed,
                "warm_start_fallback_considered": warm_start_fallback_considered,
                "warm_start_fallback_used": warm_start_fallback_used,
                "warm_start_fallback_block_reason": warm_start_fallback_block_reason,
                "warm_start_fallback_parent_unseen": warm_start_fallback_parent_unseen,
                "warm_start_fallback_pool_baseline": warm_start_fallback_pool_baseline,
                "warm_start_fallback_delta": warm_start_fallback_delta,
                "override_gene_id": override_gene_id,
                "override_gene_lam": override_gene_lam,
                "override_gene_parent_tol": override_gene_parent_tol,
                "override_gene_transfer_floor": override_gene_transfer_floor,
                "override_gene_tie_primary_min": override_gene_tie_primary_min,
                "gene_episode_id": gene_episode_id,
                "gene_episode_step0": gene_episode_step0,
                "gene_episode_step1": gene_episode_step1,
                "gene_episode_gene_id": gene_episode_gene_id,
                "gene_selected_by": gene_selected_by,
                "gene_ucb_score": gene_ucb_score,
                "gene_mean_reward_before": gene_mean_reward_before,
                "gene_n_before": gene_n_before,
                "gene_episode_reward": gene_episode_reward,
                "gene_episode_reward_transfer_component": gene_episode_reward_transfer_component,
                "gene_episode_reward_general_component": gene_episode_reward_general_component,
                "gene_episode_reliable": gene_episode_reliable,
                "gene_episode_n_trusted": gene_episode_n_trusted,
                "gene_episode_n_events": gene_episode_n_events,
                "gene_episode_collapse_steps": gene_episode_collapse_steps,
                "gene_episode_transfer_proxy_mean": gene_episode_transfer_proxy_mean,
                "gene_episode_general_proxy_rate": gene_episode_general_proxy_rate,
                "gene_episode_matured_count": gene_episode_matured_count,
                "gene_episode_matured_step": gene_episode_matured_step,
                "gene_episode_matured_id": gene_episode_matured_id,
                "gene_episode_matured_gene_id": gene_episode_matured_gene_id,
                "gene_episode_proxy_reward": gene_episode_proxy_reward,
                "gene_episode_delayed_reward": gene_episode_delayed_reward,
                "gene_episode_reward_used": gene_episode_reward_used,
                "gene_episode_delayed_reliable": gene_episode_delayed_reliable,
                "gene_episode_matured_reliable": gene_episode_matured_reliable,
                "gene_episode_primary_survival": gene_episode_primary_survival,
                "gene_episode_transfer_survival": gene_episode_transfer_survival,
                "gene_rollback": gene_rollback,
                "gene_rollback_reason": gene_rollback_reason,
                "checkpoint_saved": bool(checkpoint_path),
                "checkpoint_path": checkpoint_path or "",
                "checkpoint_id": checkpoint_id or "",
            }
        )

        batch.meta = meta_payload

        batch_summary = {
            "programs": len(episodes),
            "cached_runs": cached_count,
            "cache_rate": cached_count / max(1, len(episodes)),
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "rml_planck_h": current_planck,
            "best_scalar": best_scalar,
            "median_scalar": median_scalar,
            "cache_scope": self.cache_scope,
            "marginals": batch.meta.get("marginals") if batch.meta else None,
            "best_unseen_accuracy": best_unseen_acc,
            "best_shift_accuracy": best_shift_acc,
            "pass_rate": pass_rate,
            "gate_fail_counts": gate_fail_counts,
            "elite_coupling_counts": elite_coupling_counts,
            "elite_len_steps_corr": elite_len_steps_corr,
            "headline_metric": "generalization_score",
            "baseline_ref": "rolling_best_20",
            "eval_contract_version": batch.meta.get("eval_contract", {}).get("version") if batch.meta else None,
            "rml_accept": decision.get("accepted"),
            "rml_reason": decision.get("reason"),
            "rml_tunnel": bool(tunneled),
        }

        if hasattr(self.store, "insert_batch"):
            with self.store.transaction() as conn:
                self.store.insert_batch(
                    {
                        "batch_id": ids.batch_id(step_idx, run_ids, self.dist.snapshot(), self.updater.snapshot()),
                        "engine_step": step_idx,
                        "rng": rng_step,
                        "dist_snapshot": self.dist.snapshot(),
                        "updater_snapshot": self.updater.snapshot(),
                        "run_ids": run_ids,
                        "episode_summaries": [
                            {
                                "program_id": ids.program_id(ep.program.graph),
                                "cached": ep.cached,
                                "violations": getattr(ep.eval_report, "constraint_violations", []),
                            }
                            for ep in episodes
                        ],
                        "meta": batch_summary,
                    },
                    conn=conn,
                )
        if hasattr(self.evaluator, "update_batch_history"):
            try:
                self.evaluator.update_batch_history(batch_summary)
            except Exception:
                pass

        return batch
