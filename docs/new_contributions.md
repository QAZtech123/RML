# New Contributions (Implemented)

This page tracks concrete, code-level contributions added during the current RML cycle.
It is intentionally operational, not aspirational.

## 1) Override Geometry And Acceptance Modes

- Added strict acceptance decomposition into explicit modes:
  - `original_strict`
  - `transfer_tie_ok`
  - `parent_regress_tolerated`
  - `rescue`
- Added explicit strict score telemetry:
  - `strict_score = mean_delta + lambda * transfer_delta`
- Added mode-specific guardrails:
  - transfer-tie minimum primary delta
  - parent-regress transfer floor
  - parent tie and score-compensation tolerances

## 2) Baseline Provenance + Trust Accounting

- Added baseline source labeling: `pool`, `global`, `missing`
- Added global baseline architecture decomposition:
  - `all`, `mlp`, `transformer`, `unknown`
- Added `b_arch_src` in per-override lines for reproducible interpretation.
- Added trusted denominator accounting for general-improve metrics:
  - `n_trusted_baseline`
  - `n_untrusted_baseline`

## 3) Kernel-Level General Improvement Metrics

- Added `general_improve_event` and related per-tier summaries.
- Added reliability gating based on trusted sample count.
- Added explicit NA handling when trusted support is zero.

## 4) Delayed Reward Policy Evolution

- Added override gene catalog and UCB1 selection.
- Added episode-level proxy reward.
- Added matured delayed reward and blended update:
  - `proxy_reward`
  - `delayed_reward`
  - `reward_used`
- Added per-gene comparison reporting in lineage stats:
  - proxy vs delayed means
  - correlation and reliability columns

## 5) Collapse Observability Subsystem

- Added collapse-state truth categories:
  - `missing`
  - `unpopulated`
  - `broken_json`
  - `active`
- Added parse-failure diagnostics and examples.
- Added adjacency diagnostics (`t-1, t, t+1`) with guardrails.
- Added candidate-density diagnostics and configurable low-candidate threshold.

## 6) No-Parent Diagnostics

- Added per-step no-parent composition metrics:
  - `candidate_no_parent_n`
  - `candidate_no_parent_rate`
  - candidate family counts
- Added lineage analysis:
  - no-parent/collapse correlations
  - no-parent bucket collapse rates
  - top no-parent steps

## 7) Rescue Injection Evolution

- Added rescue trigger telemetry:
  - `rescue_triggered`, `rescue_reason`, `rescue_injected_n`
- Added rescue supply ladder and supply-state telemetry:
  - `rescue_supply_status`
  - `rescue_supply_fail_reason`
  - `rescue_supply_candidates_seen_n`
  - `rescue_supply_attempts_n`
  - `rescue_supply_source_selected`
- Added rescue vs non-rescue comparative diagnostics in lineage reports.

## 8) GitHub Collaboration Infrastructure

- Added CI workflow (`.github/workflows/ci.yml`)
- Added issue templates (bug + experiment report)
- Added PR template
- Added code owners file

## 9) Documentation And Paper-Prep Support

- Added architecture, math, results, and publishing docs.
- Added structured lab notebook (`rml_lab_log.md`) usage conventions.
- Added short paper scaffold and build scripts in `paper_prep/`.
