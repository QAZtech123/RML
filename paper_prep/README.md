# Paper Prep (Pre-arXiv)

This directory is for preparation artifacts only. It is intentionally not a submission draft.

## Goal
Make eventual paper writing mechanical once evidence is stable.

## Readiness Policy
Do not start full manuscript drafting until at least one gate is satisfied:
- General-improvement stability across multiple long runs and seeds.
- Evolved-policy advantage over fixed strict baseline.
- Delayed-reward convergence and consistent gene preference across seeds.

See:
- `../rml_lab_log.md` for empirical run history.
- `outline.md` for manuscript structure.
- `experiment_protocol.md` for reproducible eval setup.

## Required Assets Before Drafting
- 3+ seed runs with consistent config.
- Lineage reports archived per run.
- One ablation table set:
  - fixed strict-v3
  - proxy-only bandit
  - delayed-only bandit
  - blended delayed/proxy bandit
  - maturity window comparison (W=10 vs W=20)

## Short Paper Artifact (Current Technical Summary)
- Source: `rml_recursive_meta_learning_short_paper.md`
- Builder: `build_short_paper.py`
- Output:
  - HTML: `rml_recursive_meta_learning_short_paper.html`
  - PDF: `rml_recursive_meta_learning_short_paper.pdf` (when Edge headless is available)

Build:
```powershell
python .\paper_prep\build_short_paper.py
```

## Naming Convention
- Run artifacts: `runs/train_log_transfer_vXX.csv`
- Reports: `runs/lineage_vXX_default.txt`
- Figures/tables:
  - `paper_prep/figures/F<NN>_<short_name>.png`
  - `paper_prep/tables/T<NN>_<short_name>.md`
