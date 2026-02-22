# Experiment Protocol (Pre-arXiv)

Use this protocol to avoid accidental drift while evaluating policy evolution.

## 1) Fixed Inputs
- Runner: `real`
- Steps: `300` per run
- Programs per step: `6`
- Seed set: choose fixed list (example: `123, 456, 789`)
- Episode length `E`: `50`
- Maturity window `W`: evaluate `10` and `20`
- Reward blending:
  - blended: proxy `0.3`, delayed `0.7`
  - ablations: proxy-only, delayed-only

## 2) Run Matrix
- Baseline strict policy (no gene evolution)
- Gene-evolved blended reward
- Gene-evolved proxy-only
- Gene-evolved delayed-only
- Optional: W comparison (`10` vs `20`)

## 3) Command Template
```powershell
python -m rml.cli train --runner real --steps 300 --programs-per-step 6 `
  --db runs\rml_transfer_<RUN_ID>.db `
  --out runs\train_log_transfer_<RUN_ID>.csv `
  --artifact-root artifacts_transfer_<RUN_ID> `
  --seed <SEED> --verbose
```

## 4) Report Generation
```powershell
python .\scripts\lineage_stats.py .\runs\train_log_transfer_<RUN_ID>.csv `
  --candidate-low-k 5 | Tee-Object .\runs\lineage_<RUN_ID>_default.txt
```

## 5) Required Logged Outputs
- `n_overrides`
- `n_trusted_baseline`
- `n_matured_episodes`
- `mean_proxy_reward`
- `mean_delayed_reward`
- `mean_reward_used`
- `corr_proxy_delayed`
- `mean_primary_surv_del`
- `mean_transfer_surv_del`
- `collapse_status`

## 6) Promotion Criteria (from notebook mode to manuscript mode)
- At least one readiness gate satisfied from `../rml_lab_log.md`.
- Evidence replicated across at least 3 seeds.
- Ablation results archived with matching config metadata.

## 7) Run Entry Checklist
- Immediately append run summary to `../rml_lab_log.md`.
- Include commit hash, instrumentation version, and all threshold knobs.
- Record one explicit hypothesis update and one next test.
