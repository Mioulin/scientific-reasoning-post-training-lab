# Persistent experiment ledger

- Environment inspection: COMPLETED. Apple M4 Pro, 24 GB unified RAM, macOS 15.7.4, Python 3.12.13.
- Dependency resolution: COMPLETED; runtime verified on MPS; exact versions in uv.lock and experiment JSON.
- Baseline: COMPLETED, 120 examples, strict accuracy 16/120, 30.91 seconds.
- Dataset and metric validation: COMPLETED. Eight CPU tests pass including tiny-model SFT/DPO integration.
- SFT: COMPLETED, 72 steps / 2 epochs, validation loss 0.346042; checkpoint checkpoints/sft/final. Generative evaluation in progress.
- DPO: NOT RUN.
- Reward modelling: NOT RUN.
- Analysis and reports: NOT RUN.
- PPO: NOT RUN (optional, reconsider after A–D).
- Publication: BLOCKED pending final user confirmation; CLI unauthenticated. Connected GitHub identity: Mioulin.

No metrics are estimated or simulated. Update this ledger only from saved artefacts.
