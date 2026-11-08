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

## Verified computation extension, 2026-11-08

Branch: research/verified-computation. Original published v2 study remains separate.
Fresh data: 384 train, 64 validation, 144 test, 144 final; eight training families and four held-out families.
SFT seed 42: COMPLETED (96 steps, saved/reloaded final adapter).
First DPO seed 42 attempt: FAILED before optimiser creation because YAML `5e-05` became a string. Preserved failure JSON and `experiments/toolstudy/dpo_42_training_failed_yaml.log`. Corrected configs to numeric 0.00005, added explicit rate parsing and regression test. Retry IN PROGRESS.
Initial baseline/SFT evaluation used batch 1; archived in experiments/toolstudy/pilot_batch1. All primary comparisons are rerun at batch 8 for throughput, with unchanged prompts, tasks, greedy decoding and 64-token budget.
Other seed runs and comparative report: IN PROGRESS. `scripts/run_toolstudy.sh` resumes completed training and reruns evaluation. Runtime logs and checkpoints remain on disk.
Numerical value-only diagnostics were added post hoc after format failures; never substitute them for the frozen strict primary outcome.
New public push: NOT RUN, awaiting concrete final review.

### Extension completion

COMPLETED: all three SFT seeds, all three DPO seeds, 4,032 primary predictions at batch 8, comparative analysis, plots, independent regrading and 16 tests. The initial DPO type failure is historical and recovered. 28 calculator requests executed, all incorrect numeric literals and no arithmetic operations. Do not describe correct model tool use as achieved. Publication of the extension remains NOT RUN pending confirmation.
