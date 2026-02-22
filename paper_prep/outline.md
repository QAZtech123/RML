# Pre-arXiv Outline (Skeleton)

## 1. Abstract (draft later)
- Problem: recursive policy selection in self-improving training loops.
- Method: override-gene bandit with delayed survival reward.
- Claim target: stable improvement under trusted baselines with safety constraints.
- Status: pending readiness gates.

## 2. Introduction
- Motivation for consequence-aware selection over instant proxy optimization.
- Why observability and provenance are first-class constraints.
- Contributions (to finalize after evidence lock).

## 3. System Overview
- Engine loop.
- Audition/override pipeline.
- Strict/rescue policy geometry.
- Collapse instrumentation + diagnostics.

## 4. Policy Evolution Mechanism
- Gene parameterization.
- Episode boundaries.
- UCB selection.
- Proxy reward vs delayed survival reward blending.
- Reliability gating and penalties.

## 5. Experimental Protocol
- Runs/seeds/step budgets.
- Evaluation metrics and trust filters.
- Baseline policies and fallback handling.
- Reproducibility knobs.

## 6. Results
- Override throughput and trusted sample density.
- Gene performance table (proxy, delayed, used reward).
- Survival gains by accept mode.
- Ablations:
  - fixed vs evolved policy
  - proxy-only vs delayed-only vs blended
  - W=10 vs W=20

## 7. Failure Analysis
- Collapse states and adjacency diagnostics.
- Misalignment cases (proxy high, delayed low).
- Non-stationarity and regime confounds.

## 8. Discussion
- Limits of current task-family scope.
- Reliability thresholds and claim boundaries.
- Practical implications for recursive optimization.

## 9. Limitations and Safety
- Small-sample risk.
- Reward hacking via proxy metrics.
- Guardrails, rollback policies, and trust accounting.

## 10. Conclusion
- What is proven vs not proven.
- Next milestones required for stronger claims.

## Appendix
- Config tables.
- Full metric definitions.
- Additional per-run diagnostics.
