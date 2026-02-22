# Mathematical Reasoning

This document describes the core equations used in gating, diagnostics, and policy evolution.

## 1) Survival Gain Definitions

For an override at step `s` and a fixed post-window `W` over same-pool future rows:

- `window_primary(s) = mean(best_unseen_accuracy over next W same-pool rows)`
- `window_transfer(s) = mean(transfer_unseen_accuracy over next W same-pool rows)`

Given baseline values `b_primary`, `b_transfer`:

- `primary_survival_gain = window_primary - b_primary`
- `transfer_survival_gain = window_transfer - b_transfer`

Baselines are robust medians with configurable trimming and sparse fallback:
- pool baseline if `n_pool >= min_n`
- otherwise global baseline (or missing, depending on policy)

## 2) Trusted Baseline Criterion

An override row is trusted when:
- required gains are computable, and
- baseline source is pool, or global with acceptable unknown-arch share.

This prevents event inflation from low-quality or ambiguous baselines.

## 3) Kernel General Improvement Event

For trusted rows:

`general_improve_event = 1` iff all conditions hold:
- `primary_survival_gain > 0`
- `transfer_survival_gain > 0`
- `block_reason in {override_allowed_strict, override_allowed_rescue}`

Else `general_improve_event = 0`.

Reported rate:
- `p_general_improve = n_events / n_trusted_baseline`

## 4) Strict-v3 Style Acceptance Geometry

Strict acceptance is not a single threshold. It combines:
- mean gain floor
- parent regression tolerance
- transfer conditions
- mode-specific constraints

A common composite score form:

`score = primary_delta + lambda * transfer_delta`

where `lambda` is gene-controlled.

Mode examples:
- `original_strict`
- `transfer_tie_ok`
- `parent_regress_tolerated`
- `rescue`

## 5) Override Gene Episode Reward

At episode end, proxy components:
- transfer component (scaled/clipped)
- general-event component (mapped from rate to [-1, 1])

At maturity step (`end + W_mature`), delayed components:
- delayed transfer survival term
- delayed primary survival term

Blended reward:

`reward_used = w_proxy * proxy_reward + w_delayed * delayed_reward`

with reliability shrink when sample support is weak.

## 6) Collapse and No-Parent Diagnostics

Step-level collapse flags come from low median scalar thresholding and candidate snapshots.

No-parent signal:

- `candidate_no_parent_rate = candidate_no_parent_n / max(candidate_n, 1)`

Analysis layers:
- correlation with collapse/best/median
- bucketed collapse rates by no-parent rate bands
- adjacency around collapse steps (`t-1, t, t+1`)

## 7) Rescue Injection Trigger

Rescue can be enabled with strict gates:
- high no-parent rate
- dense candidate batch
- quality failure signal (`best low` or collapse/low-split evidence)
- run/episode injection budgets available

Rescue is explicitly audited via:
- trigger flag
- reason string
- injected count
- source mix

## 8) Claim Boundary

These equations support a measurable kernel for recursive policy optimization.
They do not, by themselves, prove broad general intelligence improvement.
Claims should remain bounded to observed task-family performance and trusted diagnostics.

