# Mathematical And Physics Reasoning

This document records the implemented equations used by the RML engine and analysis tools.
It is intentionally implementation-facing and tied to current code paths in:
- `rml/core/engine.py`
- `rml/core/progress.py`
- `rml/core/quantum.py`
- `scripts/lineage_stats.py`

## 1) Survival Gain Equations

For override step `s`, same-pool post-window `W`, and metric streams `u_t` (primary), `v_t` (transfer):

```text
window_primary(s)  = mean_{t in W(s)} u_t
window_transfer(s) = mean_{t in W(s)} v_t
```

With baselines `b_primary`, `b_transfer`:

```text
primary_survival_gain  = window_primary(s)  - b_primary
transfer_survival_gain = window_transfer(s) - b_transfer
```

Baselines use robust median logic:
- pool baseline if `n_pool >= baseline_min_n`
- else global baseline if policy is `global`
- else missing baseline
- trimmed median used when `n >= baseline_trim_n` with `baseline_trim_frac`

## 2) Trusted Baseline + Kernel Event

For each override row:

```text
trusted_baseline =
  (b_src == "pool")
  OR (b_src == "global" AND unknown_arch_share <= general_unknown_max)
```

Kernel event (`general_improve_event`) is:

```text
event = 1{
  primary_survival_gain > 0
  AND transfer_survival_gain > 0
  AND block_reason in {override_allowed_strict, override_allowed_rescue}
  AND trusted_baseline
}
```

Main rate:

```text
p_general_improve = n_events / n_trusted_baseline
```

## 3) Strict Override Geometry (Implemented)

Let:
- `Delta_mean` = audition mean delta (`mean_delta`)
- `Delta_parent` = parent delta
- `Delta_transfer` = transfer delta

Composite strict score:

```text
strict_score = Delta_mean + lambda * Delta_transfer
```

Transfer floor in strict path:

```text
strict_transfer_floor = min(transfer_floor, transfer_gain_eps, 0.0)
```

Parent modes:
- `parent_ok` if `Delta_parent >= parent_gain_floor`
- `parent_tie_tolerance` if `Delta_parent >= -parent_tie_tolerance`
- `parent_score_compensated` if
  `Delta_parent >= -parent_score_tolerance` and `strict_score >= tradeoff_min`

Mode-dependent constraints:
- `transfer_tie_ok` requires `Delta_mean >= transfer_tie_primary_min`
- `parent_regress_tolerated` requires `Delta_transfer >= parent_regress_transfer_floor`

Strict accept condition:

```text
strict_ok =
  strict_gain_ok
  AND strict_transfer_ok
  AND strict_transfer_tie_primary_ok
  AND strict_parent_regress_transfer_ok
  AND parent_mode != none
```

## 4) Rescue Trigger Equation

Rescue trigger (step-level) is:

```text
rescue_triggered =
  (step_idx > 0)
  AND (candidate_no_parent_rate >= rescue_no_parent_rate)
  AND (candidate_n >= programs_per_step)
  AND quality_failure
  AND (inject_budget > 0)
```

Where:

```text
quality_failure =
  (best_scalar < rescue_best_floor)
  OR collapse_flag
  OR (low_unseen_n + low_shift_n >= rescue_low_split_n)
```

Rescue supply states (new structured telemetry):
- `ok`
- `empty`
- `filtered`
- `error`

With fail reason examples:
- `no_checkpoints`
- `all_filtered`
- `supply_error`

## 5) Collapse And Composition Equations

Collapse flag:

```text
collapse_step_flag = 1{median_scalar < collapse_scalar_threshold}
```

No-parent composition:

```text
candidate_no_parent_rate = candidate_no_parent_n / max(candidate_n, 1)
```

Core analysis:
- correlation of `candidate_no_parent_rate` with collapse, best, median
- bucketed collapse probabilities by no-parent-rate bands
- adjacency deltas across `t-1, t, t+1`

## 6) Physics-Inspired Quantum Equations

### 6.1 Tunneling Probability

For current score `S_c`, candidate score `S_n`, barrier width `w`, Planck-like constant `h`:

```text
if S_n >= S_c:
    P_tunnel = 1
else:
    DeltaV = S_c - S_n
    P_tunnel = exp(-(DeltaV * w) / h)
P_tunnel = min(max_tunnel_prob, P_tunnel)
```

Observation (wavefunction collapse):

```text
tunnel_accept ~ Bernoulli(P_tunnel)
```

### 6.2 QuantumSearch Amplitudes

Architecture probability from amplitudes `A_i`:

```text
p_i = max(min_amp, A_i)^2 / sum_j max(min_amp, A_j)^2
```

Interference update for selected architecture `k`:

```text
if outcome_score > 0:
    A_k <- min(1, A_k * boost)
else:
    A_k <- max(min_amp, A_k * decay)
```

Uncertainty mixing:

```text
mix = clip(accuracy - 0.55, 0, uncertainty_strength)
p_i' = (1 - mix) * p_i + mix * (1 / N_arch)
A_i <- sqrt(max(min_amp, p_i'))
```

## 7) Gene Bandit Equations

### 7.1 UCB1 Selection

For gene `i` with empirical mean `mu_i`, pulls `n_i`, total pulls `N`, exploration coefficient `c`:

```text
UCB_i = mu_i + c * sqrt(log(N + 1) / max(1, n_i))
```

### 7.2 Proxy Reward (Episode End)

```text
transfer_component = clip(transfer_proxy_mean / transfer_target, -1, 1)
general_component  = 2 * (general_proxy_rate - 0.5)
proxy_reward       = 0.7 * transfer_component + 0.3 * general_component
```

Reliability shrink:

```text
if not reliable: proxy_reward *= low_reliability_shrink
```

Collapse penalty:

```text
if collapse_steps > 0 and proxy_reward < 0:
    proxy_reward -= min(0.20, 0.02 * collapse_steps)
```

### 7.3 Delayed Reward (Matured Episode)

```text
primary_survival  = current_primary  - baseline_primary
transfer_survival = current_transfer - baseline_transfer
```

```text
delayed_reward =
    0.7 * clip(transfer_survival / transfer_target, -1, 1)
  + 0.2 * clip(primary_survival  / primary_target,  -1, 1)
  + 0.1 * proxy_general_component
```

Blended reward used for bandit update:

```text
reward_used =
    w_proxy   * proxy_reward
  + w_delayed * delayed_reward
```

Then reliability shrink + collapse penalty are re-applied before arm update.

## 8) New Contributions Implemented

New work now in this repository:
- strict mode decomposition into `original_strict`, `transfer_tie_ok`, `parent_regress_tolerated`
- explicit `accept_mode` telemetry and per-mode survival diagnostics
- trusted-baseline accounting (`pool/global/missing`, `b_arch_src`)
- kernel `general_improve_event` metrics with trusted denominator controls
- delayed gene reward maturation (proxy vs delayed vs blended)
- rescue injection diagnostics with supply-state truth (`ok/empty/filtered/error`)
- collapse instrumentation truth states (`missing/unpopulated/broken_json/active`)
- no-parent causal diagnostics (correlation, buckets, adjacency, density)

## 9) Claim Boundary

These equations define a measurable recursive optimization kernel.
They do not, by themselves, prove broad intelligence growth.
Claims should remain bounded to observed task-family evidence and trusted diagnostics.
