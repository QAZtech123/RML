# RML Lab Log

Purpose: keep a strict success/failure record for recursive meta-learning runs so decisions are reproducible and hypothesis updates are explicit.

## Usage Rules
- One entry per major run or experiment batch.
- Record facts first, interpretation second.
- Do not collapse unknowns into conclusions.
- Every entry ends with a concrete next test.

## Entry Template
```markdown
## Run <RUN_ID> - <YYYY-MM-DD>

### Header
- commit_hash:
- runner_version:
- engine_instrumentation_version:
- override_gene_version:
- seed:
- steps:
- programs_per_step:
- episode_len_E:
- maturity_window_W:
- reward_blend_proxy:
- reward_blend_delayed:
- baseline_policy:
- baseline_min_n:
- baseline_trim_n:
- baseline_trim_frac:
- candidate_low_k:
- csv_path:
- lineage_report_path:

### Observed Metrics
- n_overrides:
- n_trusted_baseline:
- n_matured_episodes:
- mean_delayed_reward:
- mean_reward_used:
- corr_proxy_delayed:
- delayed_reliability_rate:
- best_gene_by_mean_reward_used:
- collapse_status:

### What Worked
- 

### What Failed
- 

### Hypothesis Update
- Current belief:
- Evidence strength: low | medium | high
- Risks/confounds:

### Next Test
- Single highest-ROI change:
- Success criterion:
- Abort criterion:
```

## Decision Gates (Paper Readiness)
- Stay in lab-log mode unless one gate below passes.

### Gate A: General Improvement Stability
- `p_general_improve > 0.5` across at least 3 seeds and repeated long runs.

### Gate B: Evolved Policy Advantage
- Gene-evolved policy outperforms fixed strict-v3 baseline on delayed transfer survival without collapse-rate blowup.

### Gate C: Delayed Reward Convergence
- Delayed reward differentiation is stable and bandit selection converges consistently across seeds.

## Seed Entries

## Run v18 - historical summary

### Header
- commit_hash: unknown (backfill if available)
- runner_version: dev
- engine_instrumentation_version: pre-delayed-reward
- override_gene_version: strict-v2/v3 transition period
- seed: mixed historical
- steps: 300
- programs_per_step: 6
- episode_len_E: n/a
- maturity_window_W: n/a
- reward_blend_proxy: n/a
- reward_blend_delayed: n/a
- baseline_policy: global
- baseline_min_n: 5
- baseline_trim_n: 7
- baseline_trim_frac: 0.2
- candidate_low_k: 5
- csv_path: runs/train_log_transfer_v18.csv
- lineage_report_path: runs/lineage_v18_default.txt

### Observed Metrics
- n_overrides: low
- n_trusted_baseline: limited
- n_matured_episodes: n/a
- mean_delayed_reward: n/a
- mean_reward_used: n/a
- corr_proxy_delayed: n/a
- delayed_reliability_rate: n/a
- best_gene_by_mean_reward_used: n/a
- collapse_status: instrumentation schema present but unpopulated in v18 CSV path

### What Worked
- Collapse/adjacency diagnostics became explicit and non-silent.
- Baseline provenance and trust accounting were visible in report outputs.

### What Failed
- Collapse columns existed in header but were unpopulated for v18.
- Override throughput remained a bottleneck.

### Hypothesis Update
- Current belief: observability was no longer the main blocker; policy geometry and eligibility throughput were.
- Evidence strength: medium
- Risks/confounds: mixed instrumentation vintages across runs.

### Next Test
- Single highest-ROI change: verify active collapse instrumentation on new runs and continue strict policy tuning.
- Success criterion: non-empty collapse diagnostics with coherent source/arch patterns.
- Abort criterion: repeated runs with schema/emit mismatch.

## Run v20_tiny - delayed-reward wiring smoke

### Header
- commit_hash: unknown (backfill with `git rev-parse --short HEAD`)
- runner_version: dev
- engine_instrumentation_version: 5
- override_gene_version: initial UCB + delayed maturity integration
- seed: 123
- steps: tiny smoke
- programs_per_step: 2
- episode_len_E: 50
- maturity_window_W: 10
- reward_blend_proxy: 0.3
- reward_blend_delayed: 0.7
- baseline_policy: global
- baseline_min_n: 5
- baseline_trim_n: 7
- baseline_trim_frac: 0.2
- candidate_low_k: 5
- csv_path: runs/train_log_gene_delayed_smoke*.csv
- lineage_report_path: console smoke output

### Observed Metrics
- n_overrides: minimal (smoke setting)
- n_trusted_baseline: low
- n_matured_episodes: observed at expected boundaries (e.g., step 59, 109 for E=50/W=10)
- mean_delayed_reward: ~0 in smoke
- mean_reward_used: ~0 in smoke
- corr_proxy_delayed: insufficient sample
- delayed_reliability_rate: low in smoke
- best_gene_by_mean_reward_used: unstable due to tiny n
- collapse_status: active path validated

### What Worked
- Delayed maturity scheduling aligned with expected step math.
- CSV fields for matured reward/survival emitted correctly.
- `lineage_stats.py` parsed and reported delayed metrics.

### What Failed
- Smoke scale is too small for any meaningful gene ranking.

### Hypothesis Update
- Current belief: delayed reward plumbing is correct; next uncertainty is signal quality at real-run scale.
- Evidence strength: medium
- Risks/confounds: tiny-sample artifacts dominate all reward statistics.

### Next Test
- Single highest-ROI change: run full 300-step real-run experiment and evaluate proxy/delayed alignment.
- Success criterion: usable `n_matured_episodes` and nontrivial `corr_proxy_delayed`.
- Abort criterion: delayed reliability remains too low to inform bandit updates.

## Run v21 - 300-step real run (cleaned)

### Header
- commit_hash: n/a (workspace not in git context)
- runner_version: dev
- engine_instrumentation_version: 5
- override_gene_version: UCB genes (`g00_default`, `g01_transfer_tight`, `g02_parent_flex`)
- seed: n/a (run-local)
- steps: 300 (parsed rows: 288 valid, 2 malformed skipped by stats)
- programs_per_step: 6
- episode_len_E: 50
- maturity_window_W: 10
- reward_blend_proxy: 0.3
- reward_blend_delayed: 0.7
- baseline_policy: global
- baseline_min_n: 5
- baseline_trim_n: 7
- baseline_trim_frac: 0.2
- candidate_low_k: 5
- csv_path: runs/train_log_transfer_v21.csv
- lineage_report_path: runs/lineage_v21.txt

### Observed Metrics
- n_overrides: 6 (strict), 0 (rescue), 6 overall
- n_trusted_baseline: 6
- n_matured_episodes: 5
- mean_delayed_reward (overall matured): 0.2556
- mean_reward_used (overall matured): 0.0275
- corr_proxy_delayed (overall matured): weak/unstable (per-gene corr was -1.0 for g00/g01 at n=2 each)
- delayed_reliability_rate: 1.0 at episode maturity, but matured_reliable=0 due strict reliability gates
- best_gene_by_mean_reward_used: `g00_default` (0.1434), then `g01_transfer_tight` (0.0774), `g02_parent_flex` (-0.3050)
- collapse_status: active

### What Worked
- Engine remained runtime-stable (no retries/crashes), and delayed maturity scheduling produced expected boundary events.
- Post-override survival remained positive on primary (+0.0245) and slightly positive on transfer (+0.0036) for strict overrides.
- `transfer_tie_ok` and `parent_regress_tolerated` modes were positive in trusted windows on this run.

### What Failed
- Override throughput stayed low (`override_allowed_rate=0.0268` over considered auditions).
- Collapse rate remained high (`p_collapse_overall=0.3542`; 102 collapse steps).
- Collapse candidates were dominated by regression guards (`no_shift_regression`, `no_unseen_regression`) and `no_parent` origins.
- Delayed-reward sample size is still too small for reliable gene ranking; reliability gates zeroed `n_matured_rel`.
- CSV contained 2 malformed records (steps 5 and 68), requiring explicit skip handling in stats.

### Hypothesis Update
- Current belief: observability and delayed-reward plumbing are sound, but selection throughput and collapse prevalence are the binding constraints. Current gene differentiation signal is present but statistically underpowered.
- Evidence strength: medium
- Risks/confounds: small matured episode count; non-stationarity across architecture regimes; malformed CSV rows can contaminate metrics without sanitation.

### Next Test
- Single highest-ROI change: increase trusted/matured sample density before broad gene expansion (keep current 3 genes, run multi-seed 300-step set, then consider W=20 only if delayed noise remains high).
- Success criterion: >=15 matured episodes aggregate across seeds with stable gene ordering by `mean_reward_used` and non-negative mean transfer survival for winning gene.
- Abort criterion: collapse rate stays >0.30 with no improvement in override throughput across seeds.

## Run v22_smoke - CSV emission hardening validation

### Header
- commit_hash: n/a (workspace not in git context)
- runner_version: dev
- engine_instrumentation_version: 5
- override_gene_version: g00_default only in this short run
- seed: default
- steps: 90
- programs_per_step: 6
- episode_len_E: 50
- maturity_window_W: 10
- reward_blend_proxy: 0.3
- reward_blend_delayed: 0.7
- baseline_policy: global
- baseline_min_n: 5
- baseline_trim_n: 7
- baseline_trim_frac: 0.2
- candidate_low_k: 5
- csv_path: runs/train_log_transfer_v22_smoke.csv
- lineage_report_path: console output (`lineage_stats.py`)

### Observed Metrics
- malformed_rows_skipped: 0
- rows parsed: 90
- missing step blocks: 0 (0..89 all present)
- csv_sanitized_fields_n: 0 on all rows in this run (no runtime sanitization needed, but guard active)
- collapse_status: active (expected)
- override_allowed_rate: 0.1373
- post_override_primary_survival_mean (strict): -0.0083
- post_override_transfer_survival_mean (strict): -0.0155

### What Worked
- CSV writer hardening eliminated malformed records in a run that crossed earlier failure region (~step 68).
- Parser integrity checks (`row[None]` extras) stayed clean.
- Lab pipeline remained fully analyzable without row skipping.

### What Failed
- This run is not a policy win run; strict override outcomes were negative on both primary and transfer means.
- Gene differentiation remains underpowered (`n_matured=1`).

### Hypothesis Update
- Current belief: structural CSV corruption issue is addressed by quote-all + value sanitization; remaining bottlenecks are policy geometry and collapse prevalence, not log integrity.
- Evidence strength: medium
- Risks/confounds: single-seed smoke; corruption could still reappear in longer runs if another serialization path is introduced.

### Next Test
- Single highest-ROI change: execute full 300-step real run with hardened writer and confirm zero malformed rows while tracking collapse/override tradeoffs.
- Success criterion: 300 rows, no missing steps, malformed_rows_skipped=0, and stable gene table extraction.
- Abort criterion: any reappearance of `row[None]` extras or missing contiguous step ranges.

## Start-to-Now Retrospective (RML Kernel)

### Phase 1 - Baseline provenance and strict-gate deadlock

### Successes
- Added explicit baseline controls (`policy`, `min_n`, `trim_n`, `trim_frac`) and printed them in report headers.
- Removed baseline ambiguity by logging baseline source (`pool/global/missing`) and baseline size (`b_n`).
- Added global baseline architecture breakdown (`all/mlp/transformer/unknown`) with unknown-share visibility.

### Failures
- Early strict gate under quantized metrics treated ties as failures and starved overrides.
- Sparse per-pool baselines forced frequent global fallback, reducing local trust.
- Initial reports could be misread because baseline provenance was implicit.

### Phase 2 - Audition eligibility starvation

### Successes
- Reworked stale audition eligibility so missing-metric handling no longer discarded most stale candidates.
- Added eligibility taxonomy (`full/primary_only/transfer_only/none`) and block-reason accounting.
- Eliminated the main choke point where stale candidates were dropped as ineligible by default.

### Failures
- Override volume remained low even after eligibility fixes, revealing policy/selectivity constraints.
- High positive audition deltas did not always translate into downstream survival gains.

### Phase 3 - Accept-mode transparency and policy geometry

### Successes
- Added `accept_mode` and stratified override outcome tables by mode and trust.
- Exposed mode-level behavior differences (`original_strict`, `transfer_tie_ok`, `parent_regress_tolerated`, `rescue`).
- Enabled data-driven strict-gate revisions (strict-v2/v3) instead of guesswork.

### Failures
- Mode performance flipped across runs (non-stationary behavior), preventing naive conclusions.
- Baseline regime mixing confounded mode attribution in some runs.
- Throughput stayed low in some settings even when accepted overrides looked strong.

### Phase 4 - Collapse observability and CSV integrity

### Successes
- Built explicit collapse states: `missing`, `unpopulated`, `broken_json`, `active`.
- Added column precedence/resolution reporting, parse-fail counters, and bounded parse-fail examples.
- Added adjacency and candidate-density diagnostics with sample-size guardrails and `candidate_low_k`.
- Hardened CSV emission/reading path so malformed row corruption is detected or prevented.

### Failures
- Historical runs had schema/emit mismatches (collapse columns in header but empty body).
- Some long runs showed malformed CSV records before writer hardening.
- Early collapse analysis could silently fail without explicit state labeling.

### Phase 5 - Delayed reward and override-gene evolution

### Successes
- Added episode-based override gene bandit with delayed maturity rewards and proxy/delayed blending.
- Logged matured episode outcomes (`proxy`, `delayed`, `reward_used`, survival terms, reliability).
- Established the minimal loop for consequence-aware policy selection.

### Failures
- Matured sample count remained low in many runs, making ranking unstable.
- Proxy-to-delayed correlation was weak/volatile under small-N conditions.
- Reliability gates often reduced effective signal when trusted events were sparse.

### Phase 6 - Current frontier: no-parent collapse control

### Successes
- Added per-step no-parent metrics (`candidate_no_parent_n/rate`) and family composition counts.
- Added no-parent diagnostics section in `lineage_stats` (availability, correlations, buckets, top steps).
- Added conservative rescue injection mechanism with strict triggers, hard budgets, and full audit logging.

### Failures
- In smoke tests, rescue trigger fired but often found no eligible checkpoint parent (`no_checkpoint_found`).
- Long-run impact on collapse rate and override throughput is not yet measured.
- Rescue must prove it reduces collapse without creating stale-source overdominance.

## Cumulative Successes
- Built an auditable RML kernel with explicit policy geometry, baseline provenance, trust accounting, and failure-state observability.
- Converted multiple silent failure modes into explicit, queryable diagnostics.
- Reached consequence-aware policy adaptation (delayed reward) with reproducible logging.

## Cumulative Failures Still Open
- Reliable compounding remains unproven at scale due low trusted/matured sample density.
- Collapse prevalence is still high in hard runs and strongly linked to no-parent-heavy candidate sets.
- Override throughput remains the primary bottleneck in several regimes.
- Mode/gene performance can be regime-dependent and non-stationary; requires multi-seed confirmation.

## Run v23_prep - Path A no-parent + rescue injection integration

### Header
- commit_hash: n/a (workspace currently not in git repo context)
- runner_version: dev
- engine_instrumentation_version: 5
- override_gene_version: current UCB + delayed maturity + strict-v3 + rescue hooks
- seed: smoke defaults
- steps: smoke only (6/10)
- programs_per_step: 3
- episode_len_E: 50
- maturity_window_W: 10
- reward_blend_proxy: 0.3
- reward_blend_delayed: 0.7
- baseline_policy: global
- baseline_min_n: 5
- baseline_trim_n: 7
- baseline_trim_frac: 0.2
- candidate_low_k: 5
- csv_path: runs/train_log_rescue_on_smoke.csv
- lineage_report_path: console smoke output

### Observed Metrics
- no_parent metrics: present and populated (`candidate_no_parent_n/rate`).
- no-parent diagnostics: active (correlations + buckets + top-step table emitted).
- rescue_triggered_steps: 4 (smoke)
- rescue_injected_steps: 0 (smoke)
- rescue_injected_total: 0 (smoke)
- dominant rescue reason in smoke: `no_checkpoint_found`.

### What Worked
- Path A instrumentation is end-to-end: engine -> CSV -> lineage diagnostics.
- Rescue control plane is present, conservative by default, and fully auditable.
- Default behavior remains unchanged unless `--rescue-enable` is set.

### What Failed
- Smoke checkpoint inventory was insufficient for actual rescue injection.
- No evidence yet that rescue reduces collapse in long runs.

### Hypothesis Update
- Current belief: rescue mechanism is wired correctly but effectiveness depends on checkpoint availability and long-run trigger frequency.
- Evidence strength: low-to-medium
- Risks/confounds: short smoke horizon; checkpoint scarcity can mask rescue effect.

### Next Test
- Single highest-ROI change: run paired 300-step experiments (`v23_control` vs `v23_rescue`) and compare collapse + override throughput.
- Success criterion: lower `p_collapse_overall` and fewer catastrophic low-best steps in rescue run without reducing trusted override quality.
- Abort criterion: no collapse improvement or strong drift to stale/fallback dominance.

## Run docs_pack_2026-02-22 - GitHub and manuscript prep

### Header
- commit_hash: n/a (workspace currently not in git repo context)
- runner_version: n/a (documentation pass)
- engine_instrumentation_version: 5
- override_gene_version: unchanged
- seed: n/a
- steps: n/a
- programs_per_step: n/a
- episode_len_E: n/a
- maturity_window_W: n/a
- reward_blend_proxy: n/a
- reward_blend_delayed: n/a
- baseline_policy: n/a
- baseline_min_n: n/a
- baseline_trim_n: n/a
- baseline_trim_frac: n/a
- candidate_low_k: 5
- csv_path: n/a
- lineage_report_path: n/a

### Observed Metrics
- n_overrides: n/a
- n_trusted_baseline: n/a
- n_matured_episodes: n/a
- mean_delayed_reward: n/a
- mean_reward_used: n/a
- corr_proxy_delayed: n/a
- delayed_reliability_rate: n/a
- best_gene_by_mean_reward_used: n/a
- collapse_status: n/a

### What Worked
- Added public repo scaffold: `README.md`, `CONTRIBUTING.md`, `requirements.txt`, and `docs/`.
- Added structured technical docs for architecture, math reasoning, results, and publishing workflow.
- Added short-paper source and build pipeline; generated PDF artifact successfully.
- Added `.gitignore` cleanup for runtime artifacts and local probe files.

### What Failed
- Local one-off Edge probe files could not be removed due shell policy; now ignored by `.gitignore`.
- No new empirical results in this pass (documentation only).

### Hypothesis Update
- Current belief: repo is now ready for external technical review while preserving claim boundaries.
- Evidence strength: high (for documentation readiness), unchanged (for scientific claims).
- Risks/confounds: publication quality still depends on forthcoming multi-seed control vs rescue evidence.

### Next Test
- Single highest-ROI change: append `v23_control` and `v23_rescue` empirical entries immediately after runs complete.
- Success criterion: docs remain aligned with measured outcomes and no over-claim drift appears.
- Abort criterion: if control/rescue results contradict current written framing, revise docs before any public release.
