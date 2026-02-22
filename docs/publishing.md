# Publishing Guide (Clean GitHub + Structured Docs)

## 1) Repository Hygiene
- Keep generated runtime data out of version control:
  - `artifacts*/`
  - `runs/`
  - local DB files
- Keep source-of-truth docs in repo:
  - `README.md`
  - `docs/*.md`
  - `rml_lab_log.md`
  - `paper_prep/*` source files

## 2) Suggested Public Repo Contents
- `README.md` (quick start + status + structure)
- `CONTRIBUTING.md`
- `requirements.txt`
- `docs/` technical documentation
- `paper_prep/` manuscript source and build scripts
- `tests/` baseline validation tests

## 3) Pre-Publish Checklist
- [ ] Run `pytest -q`
- [ ] Run one short training command and one lineage stats command
- [ ] Confirm docs links resolve
- [ ] Confirm no large artifact directories are staged
- [ ] Update `rml_lab_log.md` with latest run interpretation

## 4) Recommended Release Narrative
1. What the engine is designed to do.
2. What is currently demonstrated with evidence.
3. What remains open.
4. Exact experimental protocol and reproduction commands.

## 5) Avoid Overclaiming
When publishing:
- separate "implemented" from "validated"
- report confidence and sample sizes
- include failures and confounds explicitly

## 6) Paper Asset Build
Short paper source:
- `paper_prep/rml_recursive_meta_learning_short_paper.md`

Build command:
```powershell
python .\paper_prep\build_short_paper.py
```

Output:
- HTML: `paper_prep/rml_recursive_meta_learning_short_paper.html`
- PDF (if local headless browser is available):
  `paper_prep/rml_recursive_meta_learning_short_paper.pdf`

