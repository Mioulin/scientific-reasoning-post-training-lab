# Persistent experiment ledger

Primary dataset: scireason-synthetic-v2. Protocol: reports/preregistered_protocol.md.
Pilot v1 is preserved in experiments/pilot_v1 and excluded from primary comparisons.

| Milestone | Status | Evidence |
|---|---|---|
| Environment and baseline | COMPLETED | reports/environment.json; baseline test/final JSONL |
| Data and evaluation | COMPLETED | dataset_manifest.json; dataset_audit.json; tokenization_validation.json |
| SFT | COMPLETED | 2 epochs / 72 steps; checkpoint reloaded; test 49/120, final 29/120 |
| DPO | COMPLETED | 1 epoch / 36 steps; frozen SFT reference; test 60/120, final 42/120 |
| Beta sensitivity | COMPLETED | beta .5, 1 epoch; test 60/120, final 39/120 |
| Reward modelling | COMPLETED | 2 epochs / 72 steps; test 104/120, final 105/120; shortcut probes |
| Comparative report and figures | COMPLETED | reports/research_report.md; figures checked visually |
| Local tests | COMPLETED | 10 passed; Ruff passes; uv sync --offline --locked succeeds |
| Artifact verification | COMPLETED | 8 policy evaluations, 2 reward evaluations, 4 adapter hashes checked |
| Russian learning/interview guides | COMPLETED | docs/learning_guide_ru.md; docs/interview_defence_ru.md |
| PPO | NOT RUN | No PPO implementation in pinned TRL tree; online reward not sufficiently validated |
| GitHub publication | COMPLETED | Public repository and research commit verified; reports/publication.json |
| Remote CI | COMPLETED | Linux workflow 42472058525 passed, including tests and Ruff |

No paid cloud resources were used. The public repository is published; no model weights were uploaded.
GitHub connector and CLI identity verified: Mioulin. User explicitly approved publication.

## Publication evidence

Repository: https://github.com/Mioulin/scientific-reasoning-post-training-lab
Verified research commit: 0392c9150b54404c7d7005b0c680adedec2d143f.
Linux CI: https://github.com/Mioulin/scientific-reasoning-post-training-lab/actions/runs/42472058525
See reports/publication.json. Documentation-only publication-status commits may follow.
No further approval is required for the already-authorised publication; model-weight upload remains unauthorised.

## Reproduction

uv sync --locked --extra dev --python 3.12
uv run python scripts/download_model.py
uv run scireason prepare
bash scripts/run_experiments.sh
uv run pytest -q
uv run ruff check src tests scripts

See CLAUDE.md for integrity constraints. Do not tune on existing test/final results.
