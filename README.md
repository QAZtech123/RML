# Recursive Meta-Learning (RML) Engine

[![CI](https://github.com/QAZtech123/RML/actions/workflows/ci.yml/badge.svg)](https://github.com/QAZtech123/RML/actions/workflows/ci.yml)
[![Issues](https://img.shields.io/github/issues/QAZtech123/RML)](https://github.com/QAZtech123/RML/issues)
[![License](https://img.shields.io/badge/license-unlicensed-lightgrey)](https://github.com/QAZtech123/RML)

RML is an experimental engine for consequence-aware self-improvement in training loops.
It focuses on one core question:

Can a system modify its own selection policy and improve over time without hiding failure modes?

This repository includes:
- a trainable engine with strict/rescue override policies
- delayed-reward override-gene evolution
- explicit collapse diagnostics and baseline provenance tracking
- reproducible CLI pipelines for training and lineage analysis

## Status
- Current phase: instrumented kernel with active policy-evolution experiments.
- Readiness mode: lab notebook first (`rml_lab_log.md`), manuscript after stability gates.

## Repository Structure
- `rml/`: engine, distribution, runner, storage, updater, CLI
- `scripts/lineage_stats.py`: post-run diagnostics and evaluation summaries
- `tests/`: unit tests for ids/program/storage primitives
- `paper_prep/`: manuscript scaffold, experiment protocol, short paper source
- `docs/`: architecture, math reasoning, results framing, publishing guide
- `rml_lab_log.md`: success/failure log and hypothesis updates

## Quick Start
1. Create environment and install dependencies.
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a short training experiment.
```powershell
python -m rml.cli train --runner real --steps 50 --programs-per-step 6 `
  --db runs\rml.db --out runs\train_log.csv --artifact-root artifacts --verbose
```

3. Analyze with lineage stats.
```powershell
python .\scripts\lineage_stats.py .\runs\train_log.csv --candidate-low-k 5
```

4. Compare control vs rescue composition guardrail.
```powershell
# Control
python -m rml.cli train --runner real --steps 300 --programs-per-step 6 `
  --db runs\rml_transfer_control.db `
  --out runs\train_log_transfer_control.csv `
  --artifact-root artifacts_transfer_control --verbose

# Rescue enabled
python -m rml.cli train --runner real --steps 300 --programs-per-step 6 `
  --rescue-enable --rescue-no-parent-rate 0.66 --rescue-best-floor 0.12 `
  --rescue-inject-n 1 --rescue-max-per-run 10 --rescue-max-per-episode 2 `
  --rescue-low-split-n 8 `
  --db runs\rml_transfer_rescue.db `
  --out runs\train_log_transfer_rescue.csv `
  --artifact-root artifacts_transfer_rescue --verbose
```

## Core CLI Commands
```powershell
python -m rml.cli --help
```

Subcommands:
- `train`
- `plot`
- `demo`
- `replay-run`
- `show-run`
- `verify-runs`
- `sweep-orphans`

## Documentation
- `docs/README.md`: docs index
- `docs/architecture.md`: system architecture and data flow
- `docs/mathematical_reasoning.md`: equations, gates, reward definitions
- `docs/new_contributions.md`: concrete features and instrumentation added so far
- `docs/results.md`: current empirical status and claim boundaries
- `docs/publishing.md`: clean GitHub and release checklist

## GitHub Collaboration
- CI: `.github/workflows/ci.yml`
- Issue templates: `.github/ISSUE_TEMPLATE/`
- PR template: `.github/pull_request_template.md`
- Code owners: `.github/CODEOWNERS`

## Paper-Style Summary
- Source: `paper_prep/rml_recursive_meta_learning_short_paper.md`
- Build script: `paper_prep/build_short_paper.py`
- Output: `paper_prep/rml_recursive_meta_learning_short_paper.pdf` (when local PDF tool is available)

## Reproducibility Notes
- Deterministic settings are used in `RealRunner` where possible.
- CSV logs are hardened and schema-stamped for auditability.
- `lineage_stats.py` explicitly labels instrumentation states (`missing`, `unpopulated`, `broken_json`, `active`).

## Contribution Workflow
- Keep changes measurable and reversible.
- Log every major run in `rml_lab_log.md`.
- Do not claim stable compounding without multi-seed evidence.
