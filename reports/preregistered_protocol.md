# Protocol fixed before comparative training (2026-11-08)

Model: Qwen/Qwen2.5-0.5B-Instruct, immutable revision in configs/model.json.
This is an already instruction-tuned baseline, not a raw pretrained base model.

Original CC0 synthetic data: 288 training and 48 validation examples (six families).
Evaluation: 120 test and 120 final examples, each with 12 per family; four families
are unseen during training. Wording is separated across train/validation/test for
in-distribution families. OOD test/final share templates: these are independent
numerical draws, not independent families. No claims of broad generalisation.
The test set is the primary frozen comparison; final is a separately reserved replication.
No hyperparameter selection using either evaluation set. Train epochs and beta are
fixed in configs; final checkpoints are used, not selected by test accuracy.

SFT: two epochs, rank-8 q/v LoRA, LR 2e-4, effective batch 8.
DPO: initialise SFT adapter, frozen identical SFT reference adapter, beta .1,
one epoch, LR 5e-5. Controlled sensitivity: identical run at beta .5.
Reward model: independent scalar head plus rank-8 q/v LoRA, two epochs, LR 1e-4.
Seed 42 throughout. Full float32 and max sequence length 256 for MPS robustness.
No confidence calibration: the model is not asked for a probability.

Generation: greedy, 48 new tokens, strict JSON answer and abstain fields.
Primary endpoint: exact categorical/abstention or numerical tolerance accuracy
(abs and rel 1e-3), including malformed responses as failures.
Secondary: format validity, numeric accuracy, parseable-only MAE with coverage,
abstention precision/recall, and false support on explicitly unsupported claims.
This last metric does NOT measure arbitrary free-text hallucination.
Paired example bootstrap intervals quantify sampling variability conditional on
these templates and this one training seed; family/seed uncertainty is unmeasured.
Report both ID and OOD, errors, and negative changes.

Preferences are automatically labelled and intentionally simple. Rejected
responses differ in answer correctness while sharing JSON format. Test reward
accuracy, margins, OOD accuracy, and controlled additions of irrelevant verbosity
to chosen/rejected outputs. These probes establish sensitivity, not comprehensive
absence of reward hacking. Reward accuracy is not scientific truth.

Smoke runs are plumbing checks only and their adapters are never used downstream.
No PPO unless all required stages are validated and budget permits a meaningful run.

Protocol correction: pilot v1 shared some numeric conditions across wording splits.
Version 2 rejects duplicate (family, sorted numerical conditions) keys globally.
Pilot artefacts are retained in experiments/pilot_v1 and excluded from final comparisons.
All primary runs are repeated under v2; this change was not based on outcome optimisation.

Before RM training: add semantics-preserving JSON indentation probes and opposite-sign
numerical corruptions (training rejected numbers are always increased). These assess
format sensitivity and a directional numeric shortcut. Report them as secondary probes.

Generation metadata clarification after execution: the pinned Qwen generation config
inherits repetition_penalty=1.1 and eos_token_id=[151645,151643]. The inference code
overrides pad_token_id with tokenizer.eos_token_id=151645, do_sample=false and
max_new_tokens=48. All policies used these same settings. configs/generation.json
records their effective values, including the default num_beams=1. Sampling-only
temperature/top-p/top-k do not affect greedy decoding.
