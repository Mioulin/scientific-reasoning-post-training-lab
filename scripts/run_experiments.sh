#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY -m scireason.cli evaluate --name baseline > experiments/baseline.log 2>&1
$PY -m scireason.cli train sft --config configs/sft.yaml > experiments/sft.log 2>&1
$PY -m scireason.cli evaluate --name sft --adapter checkpoints/sft/final > experiments/sft_evaluation.log 2>&1
$PY -m scireason.cli train dpo --config configs/dpo.yaml --smoke > experiments/dpo_smoke.log 2>&1
$PY -m scireason.cli train dpo --config configs/dpo.yaml > experiments/dpo.log 2>&1
$PY -m scireason.cli evaluate --name dpo --adapter checkpoints/dpo/final > experiments/dpo_evaluation.log 2>&1
$PY -m scireason.cli train dpo --config configs/dpo_beta05.yaml > experiments/dpo_beta05.log 2>&1
$PY -m scireason.cli evaluate --name dpo_beta05 --adapter checkpoints/dpo_beta05/final > experiments/dpo_beta05_evaluation.log 2>&1
$PY -m scireason.cli train reward --config configs/reward.yaml --smoke > experiments/reward_smoke.log 2>&1
$PY -m scireason.cli train reward --config configs/reward.yaml > experiments/reward.log 2>&1
$PY -m scireason.cli evaluate-reward > experiments/reward_evaluation.log 2>&1
$PY -m scireason.cli evaluate --name baseline --split final > experiments/baseline_final.log 2>&1
$PY -m scireason.cli evaluate --name sft --adapter checkpoints/sft/final --split final > experiments/sft_final.log 2>&1
$PY -m scireason.cli evaluate --name dpo --adapter checkpoints/dpo/final --split final > experiments/dpo_final.log 2>&1
$PY -m scireason.cli evaluate --name dpo_beta05 --adapter checkpoints/dpo_beta05/final --split final > experiments/dpo_beta05_final.log 2>&1
$PY -m scireason.cli evaluate-reward --split final > experiments/reward_final.log 2>&1
$PY -m scireason.cli analyse
$PY -c 'from scireason.analysis import analyse; analyse("final")'
$PY scripts/format_diagnostic.py

$PY scripts/audit_dataset.py
$PY scripts/validate_tokenization.py
$PY scripts/claim_diagnostic.py
$PY scripts/verify_artifacts.py
$PY scripts/numeric_errors.py
$PY scripts/build_report.py
