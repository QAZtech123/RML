# Results (Current Snapshot)

This section summarizes what is currently supported by run evidence.
Numbers below are from logged runs in `rml_lab_log.md` and lineage reports.

## 1) Established Wins
- Observability stack is operational:
  - baseline provenance and trust accounting
  - collapse state labeling (`missing/unpopulated/broken_json/active`)
  - mode-level override outcomes
  - delayed reward telemetry for gene evolution
- CSV integrity hardening reduced malformed-row failures in recent smoke validation.
- No-parent metrics and rescue diagnostics are now end-to-end in engine -> CSV -> lineage stats.

## 2) Known Bottlenecks
- Override throughput remains low in multiple long runs.
- Collapse prevalence can be high and often coincides with no-parent-heavy candidate sets.
- Delayed-reward sample size is still small in many runs, limiting stable gene ranking.
- Mode performance can flip across runs (non-stationarity/regime dependence).

## 3) Current Interpretation
- The system is no longer blocked by hidden instrumentation failures.
- The binding constraint is policy throughput and candidate composition quality, not missing metrics.
- Rescue injection is implemented, but long-run efficacy is still under test.

## 4) What Is Not Yet Proven
- Sustained compounding across multiple seeds with strong trusted sample sizes.
- Stable evolved-policy advantage over fixed strict policy under delayed outcomes.
- Broad general-capability improvement beyond current task-family scope.

## 5) Near-Term Evaluation Plan
- Paired run comparison:
  - `control` (rescue off)
  - `rescue` (rescue on)
- Primary readouts:
  - `p_collapse_overall`
  - low-best catastrophe frequency
  - override allowed rate
  - trusted survival gains
  - matured episode reward trends

## 6) Readiness Gates for Paper Claims
Use the gates in `rml_lab_log.md`:
- stable general-improvement rates across seeds
- evolved-policy advantage vs fixed baseline
- delayed reward convergence and reproducible gene preference

