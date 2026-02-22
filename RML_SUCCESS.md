# THE RML (bedrock contract)

Goal: build an autonomous system that reliably improves its own general intelligence over time by modifying its own learning processes, without human steering, while preserving stability, truth-seeking, and identity continuity.

Success is a pattern over time, not a one-off score. Core criterion: across repeated self-modification cycles, the system shows sustained, statistically meaningful improvement on novel task families, without degrading prior capabilities.

Non-negotiable properties now enforced in code:
- Durable generalization gain: accepted updates must improve unseen accuracy versus a rolling baseline (within tolerance) and keep that gain.
- Positive progress over time: the system accepts any net improvement; acceleration is tracked but not required for acceptance.
- Stability under change: any diverged/nan runs or gate failures block acceptance.
- Robustness to shift: shift accuracy cannot regress beyond a small guard band.
- No human steering after init: task sampling, evaluation, gating, and distribution updates are automatic once started.
- Quantum tunneling and superposition: the search maintains non-zero architecture probabilities and can occasionally accept a worse step to escape local minima.

Where this lives:
- `rml/core/progress.py`: `SelfImprovementTracker` enforces the guards above and only allows distribution updates when they pass.
- `rml/core/quantum.py`: `QuantumState` (tunneling) and `QuantumSearch` (superposition + uncertainty) inject exploration pressure without collapsing to a single architecture.
- `rml/core/factor_graph_dist.py`: consumes quantum architecture biases during sampling.
- `rml/core/engine.py`: captures split metrics, stability counts, and wraps distribution updates with the tracker; logs the RML decision into batch metadata.
- `rml/core/factor_graph_dist.py`: adds `encourage_exploration` to soften the distribution when an update is rejected.
- `rml/cli/train.py`: CSV logs the RML gate decision (`rml_accept`, reason, unseen gain, shift delta, stability/acceleration flags) alongside the usual metrics.

How to run a minimal self-improvement loop (real tasks):
```
python -m rml.cli.train --runner real --steps 20 --programs-per-step 6 --db runs/rml.db --out runs/train_log.csv
```

Interpretation: RML succeeds when you can look at the logged decisions and truthfully say: even without knowing the next self-change, you expect it to raise general capability rather than lower it. The tracker enforces that contract automatically.

## Immediate Workflow (Lab Notebook First)

Use these files while runs are in progress:
- `rml_lab_log.md`: structured success/failure entries per major run.
- `paper_prep/README.md`: readiness policy for moving from notebook mode to manuscript mode.
- `paper_prep/outline.md`: pre-arXiv paper skeleton.
- `paper_prep/experiment_protocol.md`: fixed run matrix + command templates.

Recommended loop:
1. Run experiment (`train_log_transfer_vXX.csv`).
2. Generate lineage report (`lineage_vXX_default.txt`).
3. Append one entry to `rml_lab_log.md` immediately.
4. Update hypothesis + next test before launching the next run.
