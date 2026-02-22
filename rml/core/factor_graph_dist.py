from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rml.core.program import LearningProgram, ProgramValidationError
from rml.core.variables import (
    Variable,
    default_variables,
    extract_assignment,
    render_program_from_assignment,
)


@dataclass
class UnaryFactor:
    var: str
    domain: List[Any]
    logits: np.ndarray


@dataclass
class PairFactor:
    var_a: str
    dom_a: List[Any]
    var_b: str
    dom_b: List[Any]
    logits: np.ndarray  # shape [len(dom_a), len(dom_b)]


class FactorGraphDistribution:
    def __init__(
        self,
        variables: Optional[List[Variable]] = None,
        temperature: float = 1.0,
        uniform_mix: float = 0.02,
        gibbs_sweeps: int = 3,
        lr: float = 0.25,
        eps: float = 1e-6,
        max_retries: int = 20,
        obj_prior: Optional[Dict[str, float]] = None,
    ):
        self.variables: List[Variable] = variables or default_variables()
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.gibbs_sweeps = gibbs_sweeps
        self.lr = lr
        self.eps = eps
        self.max_retries = max_retries
        self.obj_prior = obj_prior or {}
        self._arch_bias: Dict[str, float] = {}

        # Build factors
        self.unaries: Dict[str, UnaryFactor] = {
            v.name: UnaryFactor(v.name, v.domain, np.zeros(len(v.domain), dtype=float)) for v in self.variables
        }
        # Apply objective prior once at init
        if "OBJ.primary" in self.unaries and self.obj_prior:
            uf = self.unaries["OBJ.primary"]
            for k, bias in self.obj_prior.items():
                if k in uf.domain and isinstance(bias, (int, float)) and bias != 0:
                    idx = uf.domain.index(k)
                    uf.logits[idx] += float(bias)
        # Pairs to capture entanglement
        pairs = [
            ("ARCH.type", "LRULE.type"),
            ("ARCH.type", "OBJ.type"),
            ("LRULE.type", "OBJ.type"),
            ("MEM.type", "ARCH.memory_reader"),
            ("OBJ.primary", "ARCH.type"),
            ("CURR.curriculum.max_len_train", "BUDGET.steps"),
        ]
        self.pairs: Dict[Tuple[str, str], PairFactor] = {}
        for a, b in pairs:
            dom_a = self._domain(a)
            dom_b = self._domain(b)
            self.pairs[(a, b)] = PairFactor(a, dom_a, b, dom_b, np.zeros((len(dom_a), len(dom_b)), dtype=float))

        # Map variable -> factor references for fast conditional computation
        self.var_to_pairs: Dict[str, List[PairFactor]] = {v.name: [] for v in self.variables}
        for pf in self.pairs.values():
            self.var_to_pairs[pf.var_a].append(pf)
            self.var_to_pairs[pf.var_b].append(pf)
        self._last_retry_stats: Dict[str, Any] = {}

    def _domain(self, var: str) -> List[Any]:
        return self.unaries[var].domain

    def _index(self, var: str, value: Any) -> int:
        return self._domain(var).index(value)

    def _score_unary(self, var: str, value: Any) -> float:
        score = float(self.unaries[var].logits[self._index(var, value)])
        if var == "ARCH.type" and self._arch_bias:
            score += float(self._arch_bias.get(value, 0.0))
        return score

    def set_arch_bias(self, probs: Dict[str, float]) -> None:
        """Apply a log-probability bias for ARCH.type sampling."""
        if "ARCH.type" not in self.unaries:
            return
        domain = set(self.unaries["ARCH.type"].domain)
        self._arch_bias = {k: math.log(max(v, self.eps)) for k, v in probs.items() if k in domain}

    def _score_pair(self, pf: PairFactor, assignment: Dict[str, Any], v: Optional[str] = None, val: Optional[Any] = None) -> float:
        a = assignment[pf.var_a] if not (v == pf.var_a) else val
        b = assignment[pf.var_b] if not (v == pf.var_b) else val
        return float(pf.logits[self._index(pf.var_a, a), self._index(pf.var_b, b)])

    def _score(self, assignment: Dict[str, Any]) -> float:
        total = 0.0
        for v in self.variables:
            total += self._score_unary(v.name, assignment[v.name])
        for pf in self.pairs.values():
            total += self._score_pair(pf, assignment)
        return total

    def sample(self, n: int, rng: int) -> List[LearningProgram]:
        rng_np = np.random.default_rng(rng)
        samples: List[LearningProgram] = []
        retries = 0
        invalid_retries = 0
        retry_reasons: Dict[str, int] = {}
        while len(samples) < n and retries < self.max_retries * n:
            assignment = self._init_assignment(rng_np)
            assignment = self._gibbs_refine(assignment, rng_np)
            try:
                program = render_program_from_assignment(assignment)
                samples.append(program)
            except ProgramValidationError as ex:
                reason = getattr(ex, "code", "VALIDATION_ERROR")
                retry_reasons[reason] = retry_reasons.get(reason, 0) + 1
                repaired = self._repair_assignment(assignment)
                if repaired:
                    try:
                        program = render_program_from_assignment(repaired)
                        samples.append(program)
                        continue
                    except ProgramValidationError:
                        pass
                retries += 1
                invalid_retries += 1
                continue
        if len(samples) < n:
            self._last_retry_stats = {"invalid_retries": invalid_retries, "attempts": retries, "reasons": retry_reasons}
            raise RuntimeError(f"Failed to sample {n} valid programs after {retries} retries")
        self._last_retry_stats = {"invalid_retries": invalid_retries, "attempts": retries, "reasons": retry_reasons}
        return samples

    def _init_assignment(self, rng: np.random.Generator) -> Dict[str, Any]:
        assignment: Dict[str, Any] = {}
        for v in self.variables:
            probs = self._softmax(self.unaries[v.name].logits)
            assignment[v.name] = rng.choice(v.domain, p=probs)
        return assignment

    def _gibbs_refine(self, assignment: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        for _ in range(self.gibbs_sweeps):
            for v in self.variables:
                domain = v.domain
                logits = []
                for val in domain:
                    score = self._score_unary(v.name, val)
                    for pf in self.var_to_pairs[v.name]:
                        score += self._score_pair(pf, assignment, v=v.name, val=val)
                    logits.append(score / max(self.temperature, 1e-6))
                probs = self._softmax(np.array(logits))
                assignment[v.name] = rng.choice(domain, p=probs)
        return assignment

    def log_prob(self, program: LearningProgram) -> float:
        assignment = extract_assignment(program)
        return self._score(assignment)

    def entropy(self) -> float:
        ent = 0.0
        for uf in self.unaries.values():
            p = self._softmax(uf.logits)
            ent -= float(np.sum(p * np.log(p + self.eps)))
        return ent / max(len(self.unaries), 1)

    def top_marginals(self, vars_to_report: List[str]) -> Dict[str, List[Tuple[Any, float]]]:
        out: Dict[str, List[Tuple[Any, float]]] = {}
        for var in vars_to_report:
            if var not in self.unaries:
                continue
            probs = self._softmax(self.unaries[var].logits)
            domain = self.unaries[var].domain
            pairs = list(zip(domain, probs))
            pairs.sort(key=lambda x: x[1], reverse=True)
            out[var] = pairs[:3]
        return out

    # --- updating (basic elite-based) ---
    def update(self, batch: Any) -> None:
        """Update factors given MetaBatch; expects batch.episodes with eval_report.score (dict-like or ScoreVector)."""
        if not getattr(batch, "episodes", None):
            return
        episodes = batch.episodes
        # Build scalar score: prefer 'gen' then sum others
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ep in episodes:
            score = getattr(ep.eval_report, "score", None)
            if score is None:
                continue
            if hasattr(score, "gen"):
                scalar = float(score.gen) + 0.1 * float(getattr(score, "rob", 0.0))
            elif isinstance(score, dict):
                scalar = float(score.get("gen", 0.0))
            else:
                scalar = float(score)
            scored.append((scalar, extract_assignment(ep.program)))
        if not scored:
            return
        scored.sort(key=lambda x: x[0], reverse=True)
        elite_k = max(1, int(len(scored) * 0.2))
        elites = scored[:elite_k]
        self._update_unaries(elites)
        self._update_pairs(elites)
        self._apply_entropy_floor()

    def _update_unaries(self, elites: List[Tuple[float, Dict[str, Any]]]) -> None:
        for var, uf in self.unaries.items():
            counts = np.zeros(len(uf.domain), dtype=float)
            for _, assign in elites:
                counts[self._index(var, assign[var])] += 1.0
            probs = counts / max(counts.sum(), 1.0)
            target_logits = np.log(probs + self.eps)
            uf.logits = (1 - self.lr) * uf.logits + self.lr * target_logits

    def _update_pairs(self, elites: List[Tuple[float, Dict[str, Any]]]) -> None:
        pair_multipliers = {
            ("OBJ.primary", "ARCH.type"): 2.0,
        }
        # compute marginals
        marginal: Dict[str, np.ndarray] = {}
        for var, uf in self.unaries.items():
            m = np.zeros(len(uf.domain), dtype=float)
            for _, assign in elites:
                m[self._index(var, assign[var])] += 1.0
            marginal[var] = m / max(m.sum(), 1.0)
        for pf in self.pairs.values():
            joint = np.zeros_like(pf.logits)
            for _, assign in elites:
                i = self._index(pf.var_a, assign[pf.var_a])
                j = self._index(pf.var_b, assign[pf.var_b])
                joint[i, j] += 1.0
            joint = joint / max(joint.sum(), 1.0)
            baseline = np.outer(marginal[pf.var_a], marginal[pf.var_b])
            advantage = np.log(joint + self.eps) - np.log(baseline + self.eps)
            mult = pair_multipliers.get((pf.var_a, pf.var_b)) or pair_multipliers.get((pf.var_b, pf.var_a)) or 1.0
            pf.logits = (1 - self.lr) * pf.logits + (self.lr * mult) * advantage

    def _apply_entropy_floor(self) -> None:
        for uf in self.unaries.values():
            probs = self._softmax(uf.logits)
            mixed = (1 - self.uniform_mix) * probs + self.uniform_mix * (1.0 / len(probs))
            uf.logits = np.log(mixed + self.eps)

    def _repair_assignment(self, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply simple constraint repairs to reduce resample retries."""
        repaired = assignment.copy()
        # Repair policy head requirement: avoid policy_gradient if not supported
        if repaired.get("OBJ.primary") == "policy_gradient":
            # If memory/arch not set for policy, back off to supervised
            repaired["OBJ.primary"] = "cross_entropy"
            repaired["OBJ.type"] = "supervised"
        # Repair memory: if mem enabled but reader none, set reader
        if repaired.get("MEM.type") != "none" and repaired.get("ARCH.memory_reader") == "none":
            repaired["ARCH.memory_reader"] = "attention"
        # Repair memory capacity zero
        if repaired.get("MEM.type") != "none" and repaired.get("MEM.capacity_bin") == 0:
            # easiest fix: disable memory for this sample
            repaired["MEM.type"] = "none"
        # Schedule vs budget: if potential mismatch, prefer constant schedule
        if repaired.get("LRULE.schedule.kind") not in ["constant", "cosine", "linear_warmup_cosine"]:
            repaired["LRULE.schedule.kind"] = "constant"
        return repaired

    def snapshot(self) -> dict:
        return {
            "temperature": self.temperature,
            "uniform_mix": self.uniform_mix,
            "gibbs_sweeps": self.gibbs_sweeps,
            "lr": self.lr,
            "eps": self.eps,
            "variables": [{"name": v.name, "domain": v.domain} for v in self.variables],
            "unaries": {k: uf.logits.tolist() for k, uf in self.unaries.items()},
            "obj_prior": self.obj_prior,
            "pairs": {
                f"{pf.var_a}|{pf.var_b}": {"var_a": pf.var_a, "var_b": pf.var_b, "logits": pf.logits.tolist()}
                for pf in self.pairs.values()
            },
            "last_retry_stats": getattr(self, "_last_retry_stats", None),
        }

    def encourage_exploration(self) -> None:
        """Softens the distribution without accepting an update, keeping exploration alive."""
        self._apply_entropy_floor()

    # --- utils ---
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        ex = np.exp(z)
        return ex / ex.sum()
