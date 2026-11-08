# Persistent experiment ledger

Protocol: reports/preregistered_protocol.md. Dataset: scireason-synthetic-v2.
Pilot v1 is preserved in experiments/pilot_v1; exclude it from reported comparisons.

- Environment and pinned dependencies: COMPLETED (M4 Pro / 24 GB / MPS / Python 3.12.13).
- Dataset generation and isolation tests: COMPLETED (288 train, 48 validation, 120 test, 120 final).
- Baseline v2: COMPLETED, 15/120 strict test accuracy.
- SFT v2: COMPLETED, two epochs / 72 steps; checkpoint reload and test evaluation complete.
- DPO: COMPLETED, one epoch / 36 steps, 60/120 primary test accuracy; SFT reference checked frozen.
- Beta sensitivity: COMPLETED, beta .5, one epoch, 60/120 primary test accuracy.
- Reward modelling: training COMPLETED (two epochs / 72 steps); held-out evaluation currently executing.
- Final replication and analysis: NOT RUN.
- CPU tests: COMPLETED, 10 passed, including checkpoint recovery.
- PPO: NOT RUN (optional; scientific justification assessed after reward evaluation).
- Publication: BLOCKED pending final user confirmation and GitHub CLI authentication.

Active execution: scripts/run_experiments.sh, sequential offline model runs on Metal.
Check the actual experiment JSON and logs before resuming; do not rerun completed stages blindly.
Local CLI authentication can be established with gh auth login (account Mioulin).
No public repository or model-weight upload has been created.
