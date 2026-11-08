# Verified computation study — measured results

Original scireason-program-v1 data; 384 train, 64 validation, 144 primary and 144 final tasks. Four families held out. Three SFT/DPO training seeds. All policies use greedy decoding, one output and at most one bounded calculator execution; 64 output tokens. No oracle expression is shown to the model.

The intervention changes both the output instruction and available executor. It is a program-aided pipeline comparison, not an isolated causal effect of a calculator with identical prompts. No multi-turn tool agent or constrained decoding is claimed. OOD final shares templates with OOD primary, with disjoint numerical conditions.

## test
| Policy | Mode | Accuracy | Numeric accuracy |
|---|---|---:|---:|
| baseline | direct | 0.062 | 0.031 |
| baseline | calculator | 0.000 | 0.000 |
| sft_42 | direct | 0.160 | 0.031 |
| sft_42 | calculator | 0.000 | 0.000 |
| dpo_42 | direct | 0.146 | 0.021 |
| dpo_42 | calculator | 0.000 | 0.000 |
| sft_43 | direct | 0.167 | 0.031 |
| sft_43 | calculator | 0.000 | 0.000 |
| dpo_43 | direct | 0.118 | 0.021 |
| dpo_43 | calculator | 0.000 | 0.000 |
| sft_44 | direct | 0.132 | 0.010 |
| sft_44 | calculator | 0.000 | 0.000 |
| dpo_44 | direct | 0.229 | 0.010 |
| dpo_44 | calculator | 0.000 | 0.000 |

## final
| Policy | Mode | Accuracy | Numeric accuracy |
|---|---|---:|---:|
| baseline | direct | 0.000 | 0.000 |
| baseline | calculator | 0.000 | 0.000 |
| sft_42 | direct | 0.167 | 0.052 |
| sft_42 | calculator | 0.000 | 0.000 |
| dpo_42 | direct | 0.097 | 0.021 |
| dpo_42 | calculator | 0.000 | 0.000 |
| sft_43 | direct | 0.174 | 0.031 |
| sft_43 | calculator | 0.000 | 0.000 |
| dpo_43 | direct | 0.118 | 0.042 |
| dpo_43 | calculator | 0.000 | 0.000 |
| sft_44 | direct | 0.146 | 0.042 |
| sft_44 | calculator | 0.000 | 0.000 |
| dpo_44 | direct | 0.146 | 0.052 |
| dpo_44 | calculator | 0.000 | 0.000 |

## Limitations
Small artificial single-step tasks; three seeds do not establish model-family generality. Correct execution does not imply correct programme construction. Base is already instruction-tuned. Numerical conditions and family-level holdouts reduce leakage, but pretraining contamination is unmeasured. All previously published v2 results remain separate. Paired example-bootstrap intervals are conditional on the task set; do not pool three seeds as independent test problems.

Reproduce: `bash scripts/run_toolstudy.sh`. Training histories are in experiments/toolstudy_*_training.json; per-example responses, programmes and tool errors are retained in experiments/toolstudy/.

## Diagnostic decomposition (post hoc)

The strict primary outcome combines schema compliance and task correctness. After observing baseline format failures, we added numerical value-only grading as a diagnostic: it ignores missing/extra keys and abstention flags, converts numeric strings, and executes calculator expressions independently of schema. This is not the preregistered primary outcome and is not evidence of reliable instruction following. All predictions are unchanged.

| Policy | Mode | Final strict | Format valid | Numeric value only | ID | OOD |
|---|---|---:|---:|---:|---:|---:|
| baseline | direct | 0.000 | 0.410 | 0.010 | 0.000 | 0.000 |
| baseline | calculator | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| sft_42 | direct | 0.167 | 0.826 | 0.052 | 0.250 | 0.000 |
| sft_42 | calculator | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| dpo_42 | direct | 0.097 | 0.472 | 0.021 | 0.146 | 0.000 |
| dpo_42 | calculator | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sft_43 | direct | 0.174 | 0.826 | 0.031 | 0.260 | 0.000 |
| sft_43 | calculator | 0.000 | 0.097 | 0.000 | 0.000 | 0.000 |
| dpo_43 | direct | 0.118 | 0.500 | 0.042 | 0.177 | 0.000 |
| dpo_43 | calculator | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sft_44 | direct | 0.146 | 0.826 | 0.042 | 0.219 | 0.000 |
| sft_44 | calculator | 0.000 | 0.062 | 0.000 | 0.000 | 0.000 |
| dpo_44 | direct | 0.146 | 0.701 | 0.052 | 0.177 | 0.083 |
| dpo_44 | calculator | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Exploratory paired model comparisons are in model_comparisons.json (no multiplicity correction; intervals do not capture family uncertainty). Training seed ranges are in seed_summary.json; representative first-in-file successes/failures are in examples.json, with no claim that these are a random qualitative sample. Compute timers and memory definitions are in compute.json. Loss curves compare different objectives and their absolute values are not directly comparable. The tool-output schema differs from SFT numerical-output training; failures may reflect instruction distribution shift. No conclusion of intrinsic tool ineffectiveness is warranted.

## Observed conclusion

Final strict accuracy: baseline 0.00%; SFT mean 16.20% (seed SD 1.45 percentage points); DPO mean 12.04% (SD 2.44 points). Compare individual seeds before generalising: the strict score includes formatting, not only scientific knowledge.

Across all calculator conditions, 28 requests executed; 0 contained an arithmetic operation. Executing a numeric literal does not demonstrate programme construction. The saved records distinguish execution from a correct final answer; this study does not establish reliable tool use.

Instrumented training/evaluation timers sum to 1842.4 seconds, excluding model loading and development. Exact model revision, software, device and masking/reference checks are recorded in setup and summary JSON.
