# Engineering decisions and observed problems

1. **Runtime isolation.** The system Python was 3.9.6. The project uses an isolated
   Homebrew Python 3.12.13 environment and a pinned lockfile; existing Anaconda
   environments were not modified.
2. **Sandbox capability visibility.** Sandboxed PyTorch reported MPS unavailable;
   the authorised host execution reported MPS available and ran real Metal operations.
   This was an execution-permission distinction, not lack of an Apple GPU.
3. **Network setup.** The first sandboxed PyPI attempt failed DNS resolution.
   An authorised dependency download and public model download succeeded. Training
   and evaluation use local model files, with the sequential script in offline mode.
4. **Leakage audit failure.** Pilot v1 rejected exact prompt duplicates but allowed
   repeated numerical conditions under different wording. Version 2 rejects both.
   Pilot logs are preserved, all primary runs repeated, and no pilot result is used
   in the main tables. This still does not guarantee semantic independence.
5. **Current tokenizer API.** A token-length audit initially counted the keys of
   a BatchEncoding returned by apply_chat_template, producing a spurious length 2.
   The audit was repaired by rendering then tokenising the string, and now asserts
   nontrivial token lengths. The actual trainers and generation already used the
   correct representations; this did not change model training. Maximum observed
   primary task length is 128 tokens, below the 256-token training limit.
6. **DPO reference.** The pinned TRL version clones the supplied SFT adapter into
   a reference adapter. Equality and frozen gradients are asserted; a tiny CPU
   optimisation test verifies the reference remains unchanged. Silently disabling
   the SFT adapter would instead compare with the original baseline.
7. **New reward head.** The scalar classification head is intentionally newly
   initialised, then trained and saved through PEFT modules_to_save. It is not a
   pretrained scientific truth estimator.
8. **Model configuration warning.** PEFT warns that a Hub config file is not found
   in its default cache during offline saving and assumes vocabulary unchanged.
   This project does not add tokens or resize embeddings. Checkpoint reloads and
   completed evaluations verify the saved adapters are usable.
9. **Metric confounds.** Strict JSON failures motivated a clearly labelled post-hoc
   numeric-string diagnostic. Zero false support also motivated a supported-claim
   recall diagnostic. Neither replaces or retrospectively changes the primary endpoint.
10. **Reproducibility limits.** Seeds, data hashes, model revision and dependencies
    are recorded. Bitwise equality across MPS/CUDA/CPU is not promised. Checkpoint
    recovery is integration-tested on a tiny CPU model; full-size checkpoints are
    saved and reloaded for inference.

11. **PPO availability.** Inspection of the installed TRL 1.17.1 source tree found
    no PPO implementation files. No legacy example was substituted. A separate
    compatible implementation and stronger reward validation would be needed.
