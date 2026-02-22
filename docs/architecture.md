# Architecture

## High-Level Flow
1. Sample candidate programs from a factor-graph distribution.
2. Execute candidates with warm-start policy logic (checkpoint/stale/fallback/scratch).
3. Evaluate candidates on train/shift/unseen/transfer splits.
4. Apply acceptance gate (`SelfImprovementTracker`) and optional tunneling.
5. Save accepted checkpoints and update sampling distribution.
6. Emit step-level telemetry to CSV.
7. Post-hoc analyze with `scripts/lineage_stats.py`.

## Main Components

### Engine
- File: `rml/core/engine.py`
- Responsibilities:
  - orchestrate per-step training/evaluation
  - warm-start audition and override logic
  - strict/rescue acceptance modes
  - collapse and no-parent diagnostics
  - delayed reward episode maturation for override genes

### Distribution
- File: `rml/core/factor_graph_dist.py`
- Responsibilities:
  - maintain unaries and pairwise factors over program variables
  - sample candidate programs
  - update factors from elite episodes
  - enforce entropy floor and optional architecture bias

### Progress Gate
- File: `rml/core/progress.py`
- Responsibilities:
  - enforce durable gain and shift robustness guards
  - block unstable steps (diverged/nan)
  - keep rolling baselines for acceptance decisions

### Quantum Utilities
- File: `rml/core/quantum.py`
- Responsibilities:
  - non-zero architecture probabilities
  - tunneling probability for controlled escape from local minima

### Real Runner
- File: `rml/real_runner.py`
- Responsibilities:
  - deterministic CPU execution
  - Family A algorithmic tasks
  - metrics for train/shift/unseen/transfer

### Storage
- Files: `rml/storage/*.py`
- Responsibilities:
  - SQLite run metadata
  - artifact and checkpoint persistence
  - orphan sweep and integrity verification

### CLI
- Files: `rml/cli/*.py`
- Entrypoint: `python -m rml.cli`
- Responsibilities:
  - train, replay, plot, maintenance, demo workflows
  - CSV logging and schema-safe output

### Analysis
- File: `scripts/lineage_stats.py`
- Responsibilities:
  - lineage and baseline diagnostics
  - accept-mode outcomes
  - general-improvement event tracking
  - collapse/no-parent/rescue diagnostics
  - override gene performance summaries

## Data and Telemetry

## Primary Outputs
- training CSVs in `runs/`
- artifact traces in `artifacts*/`
- lineage summaries from `scripts/lineage_stats.py`

## Observability Guarantees
- Baseline provenance labels (`pool/global/missing`)
- Collapse state labels (`missing/unpopulated/broken_json/active`)
- Candidate composition fields (including no-parent rates)
- Rescue trigger and injection telemetry
- Episode-level proxy/delayed reward metrics for policy evolution

## Current Extension Points
- Override policy geometry (`engine.py`)
- Gene catalog and bandit settings (`engine.py`)
- Baseline/trust rules (`lineage_stats.py`)
- Rescue trigger thresholds (`train CLI` -> `EngineConfig`)

