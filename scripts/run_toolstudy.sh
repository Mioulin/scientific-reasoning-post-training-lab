#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY -m scireason.toolstudy prepare
$PY -m scireason.toolstudy evaluate --name baseline > experiments/toolstudy/baseline.log 2>&1
for seed in 42 43 44; do
  for kind in sft dpo; do
    if [ ! -f "experiments/toolstudy_${kind}_${seed}_training.json" ] || [ ! -f "checkpoints/toolstudy_${kind}_${seed}/final/adapter_config.json" ]; then
      $PY -m scireason.cli train "$kind" --config "configs/toolstudy/${kind}_${seed}.yaml" > "experiments/toolstudy/${kind}_${seed}_training.log" 2>&1
    fi
    $PY -m scireason.toolstudy evaluate --name "${kind}_${seed}" --adapter "checkpoints/toolstudy_${kind}_${seed}/final" > "experiments/toolstudy/${kind}_${seed}_evaluation.log" 2>&1
  done
done
$PY -m scireason.toolstudy analyse
$PY scripts/analyse_toolstudy_diagnostics.py
