# Contributing

## Principles
- Prefer measurable changes over broad refactors.
- Preserve observability: every new behavior should be inspectable in CSV and lineage reports.
- Keep failures explicit. Do not hide missing data behind silent defaults.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Test
```powershell
pytest -q
```

## Run + Analyze Loop
```powershell
python -m rml.cli train --runner real --steps 100 --programs-per-step 6 `
  --db runs\rml_local.db --out runs\train_log_local.csv --artifact-root artifacts_local

python .\scripts\lineage_stats.py .\runs\train_log_local.csv --candidate-low-k 5
```

## Documentation Expectations
- Update `rml_lab_log.md` for significant behavior changes.
- If you change policy geometry or reward logic, update:
  - `docs/mathematical_reasoning.md`
  - `docs/results.md` (claim boundaries section)
- If you add a new emitted metric, ensure:
  - train CSV field is documented
  - lineage stats either uses it or explicitly ignores it

## Pull Request Checklist
- [ ] Tests pass locally
- [ ] New fields are backward-compatible
- [ ] No silent behavior changes
- [ ] Run entry appended (or planned) in `rml_lab_log.md`
- [ ] Docs updated for user-facing changes

