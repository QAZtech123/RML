# Recursive Meta-Learning with Consequence-Aware Policy Evolution

**Author:** RML Engine Project  
**Date:** February 2026  
**Status:** Technical short paper (pre-arXiv evidence stage)

## Abstract
We present an instrumented recursive meta-learning (RML) engine that adapts its own override policy while preserving explicit safety and audit constraints. The system combines strict/rescue warm-start override logic, trusted baseline accounting, collapse diagnostics, and delayed reward for override-gene selection. The main contribution is not a claim of general intelligence, but a practical kernel for measurable self-modification under controlled failure visibility. Empirically, the framework converts previously silent failure modes into explicit states, enables consequence-aware policy updates, and provides a reproducible protocol for evaluating compounding behavior over long horizons.

## 1. Architecture

The system has four interacting layers:

1. **Candidate Generation Layer**  
   A factor-graph distribution samples candidate program configurations across architecture, optimizer, objective, curriculum, and budget variables.

2. **Execution and Evaluation Layer**  
   A deterministic runner executes candidates on train/shift/unseen/transfer splits and emits diagnostics including stability, gate failures, and split accuracies.

3. **Policy and Acceptance Layer**  
   Warm-start audition logic (checkpoint/stale/fallback/scratch) computes deltas and applies strict/rescue acceptance geometry. A progress tracker enforces durability and shift robustness. Optional tunneling is available for controlled local-minima escape.

4. **Meta-Policy Evolution Layer**  
   Override policy parameters are grouped into discrete genes. A bandit selects genes per episode, and updates are driven by blended proxy+delayed reward with reliability penalties.

## 2. Mathematical Reasoning

### 2.1 Survival Gains
For an override at step \(s\), with future same-pool window \(W\):

\[
\text{primary\_survival}(s)=\overline{u}_{s+1:s+W} - b_u
\]
\[
\text{transfer\_survival}(s)=\overline{t}_{s+1:s+W} - b_t
\]

where \(\overline{u}\) and \(\overline{t}\) are window means for unseen and transfer accuracy, and \(b_u,b_t\) are robust baselines (pool-level with global fallback).

### 2.2 General-Improve Event
An event is counted only when baseline trust conditions hold and:

\[
\text{primary\_survival} > 0,\quad \text{transfer\_survival} > 0
\]

for overrides allowed by strict/rescue policy reason codes.

### 2.3 Strict-v3 Tradeoff Form
A common composite score form for strict-mode decisions is:

\[
\text{score} = \Delta_{\text{primary}} + \lambda \Delta_{\text{transfer}}
\]

with mode-specific floors for parent regression and transfer ties.

### 2.4 Episode Reward
At episode maturity:

\[
R_{\text{used}} = w_p R_{\text{proxy}} + w_d R_{\text{delayed}}
\]

with reliability shrink when trusted support is insufficient.

## 3. Instrumentation and Failure Visibility

The engine enforces explicit observability:
- baseline source labels (`pool`, `global`, `missing`)
- collapse state labels (`missing`, `unpopulated`, `broken_json`, `active`)
- accept-mode attribution (`original_strict`, `transfer_tie_ok`, `parent_regress_tolerated`, `rescue`)
- no-parent candidate rates and bucketed collapse analysis
- rescue trigger/injection telemetry with reason strings and source mix

This instrumentation turns silent failure classes into actionable diagnostics.

## 4. Empirical Snapshot (Current)

### Confirmed
- End-to-end logging and analysis pipeline is stable.
- Delayed reward pathway is implemented and operational.
- Policy decisions are auditable at per-step and per-episode granularity.
- No-parent diagnostics and rescue trigger controls are integrated.

### Not Yet Confirmed
- Stable long-horizon compounding across multiple seeds.
- Robust evolved-policy advantage over fixed policy baseline.
- Broad generalization beyond current task family.

## 5. Discussion

The project currently demonstrates a **measurable recursive optimization kernel**, not a finished autonomous general learner. The highest-leverage open problem is increasing trusted override throughput while reducing collapse-prone candidate composition. Rescue injection is introduced as a conservative composition guardrail to address no-parent-heavy failure regimes.

## 6. Reproducibility

Example run:

```powershell
python -m rml.cli train --runner real --steps 300 --programs-per-step 6 `
  --db runs\rml_transfer_v23_control.db `
  --out runs\train_log_transfer_v23_control.csv `
  --artifact-root artifacts_transfer_v23_control --verbose
```

Analysis:

```powershell
python .\scripts\lineage_stats.py .\runs\train_log_transfer_v23_control.csv --candidate-low-k 5
```

Rescue variant:

```powershell
python -m rml.cli train --runner real --steps 300 --programs-per-step 6 `
  --rescue-enable --rescue-no-parent-rate 0.66 --rescue-best-floor 0.12 `
  --rescue-inject-n 1 --rescue-max-per-run 10 --rescue-max-per-episode 2 `
  --rescue-low-split-n 8 `
  --db runs\rml_transfer_v23_rescue.db `
  --out runs\train_log_transfer_v23_rescue.csv `
  --artifact-root artifacts_transfer_v23_rescue --verbose
```

## 7. Conclusion

This system provides a practical foundation for recursive meta-learning under rigorous observability constraints. The key result so far is architectural: self-modifying policy dynamics are implemented with explicit trust accounting and consequence-aware reward, enabling disciplined iteration toward demonstrable compounding.

