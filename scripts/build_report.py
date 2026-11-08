"""Write the report exclusively from completed run artefacts."""

import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def records(name, split="test"):
    return [json.loads(x) for x in Path(f"experiments/{name}_{split}.jsonl").read_text().splitlines()]


names = ["baseline", "sft", "dpo", "dpo_beta05"]
metrics = {n: read(f"experiments/{n}_test_summary.json")["metrics"] for n in names}
final = {n: read(f"experiments/{n}_final_summary.json")["metrics"] for n in names}
training = {n: read(f"experiments/{n}_training.json") for n in ["sft", "dpo", "dpo_beta05", "reward"]}
reward = read("experiments/reward_test_summary.json")
reward_final = read("experiments/reward_final_summary.json")
format_diag = read("reports/format_diagnostic.json")
comparisons = read("reports/comparisons_test.json")
report = [
    "# Research report: scientific reasoning post-training lab",
    "\n## Abstract",
    (
        "We ran a small, single-seed controlled study of completion-only SFT, offline DPO and independent "
        "scalar reward modelling on Qwen2.5-0.5B-Instruct. The original synthetic dataset contains 288 training, "
        "48 validation and two 120-example held-out splits. Four of ten evaluation families are absent from training. "
        "All numbers below come from saved predictions and trainer logs; pilot v1 is excluded."
    ),
    "\n## Measured generative results",
    Path("reports/results_test.md").read_text(),
    "\n### Reserved final replication",
    Path("reports/results_final.md").read_text(),
    "\n![Primary accuracy](accuracy_test.png)\n\n![Family results](families_test.png)",
    "\n## Interpretation and uncertainty",
]
for name, comp in comparisons.items():
    lo, hi = comp["ci95"]
    report.append(
        f"- {name}: paired accuracy delta {100 * comp['delta']:.1f} percentage points; "
        f"95% example-bootstrap interval [{100 * lo:.1f}, {100 * hi:.1f}]."
    )
report.append(
    "Intervals use 10,000 paired resamples, seed 2026. They are conditional on these templates and "
    "one training seed; they do not measure family-sampling, training-seed or external-domain uncertainty. "
    "The beta comparison is a sensitivity analysis, not a test-selected best model."
)
report.append("\n### Separating answer format from numerical competence")
for name in names:
    d = format_diag[f"{name}_test"]
    report.append(
        f"- {name}: after allowing numeric strings, {d['correct_after_numeric_string_conversion']}/"
        f"{d['numeric_n']} numerical answers are correct; {d['parseable_numeric_or_string']} parseable."
    )
report.append(
    "This diagnostic was added after observing pilot format failures; it is explicitly post hoc. "
    "It does not replace the strict preregistered endpoint. Improvements in the aggregate score "
    "must not be presented as equivalent improvements in mathematical reasoning."
)
report.append("\n### In-distribution and held-out families")
report.append("| Model | ID accuracy (72 tasks) | OOD accuracy (48 tasks) |\n|---|---:|---:|")
for name in names:
    rows = records(name)
    vals = []
    for dist in ["id", "ood"]:
        selected = [r for r in rows if r["distribution"] == dist]
        vals.append(sum(r["grade"]["correct"] for r in selected) / len(selected))
    report.append(f"| {name} | {vals[0]:.3f} | {vals[1]:.3f} |")
report.append("\n### Abstention and explicit unsupported claims")
report.append(
    "| Model | Abstention precision | Recall | False support rate | Numeric MAE (parseable only) | Coverage |\n|---|---:|---:|---:|---:|---:|"
)
for name, m in metrics.items():
    mae = m["numeric_mae_parseable_only"]
    report.append(
        f"| {name} | {m['abstention_precision']} | {m['abstention_recall']:.3f} | "
        f"{m['unsupported_assertion_rate']:.3f} | {mae if mae is not None else 'undefined'} | "
        f"{m['numeric_parseable_n']}/{m['numeric_n']} |"
    )
report.append(
    "False support is measured only where the supplied claim is unsupported. It is not an open-ended "
    "hallucination rate. Unparseable answers remain primary failures even if they cannot be categorised "
    "as explicit false support. No model confidence probability or calibration score is elicited."
)
claim_diag=read('reports/claim_diagnostic.json')
report.append('\n### Post-hoc claim-class diagnostic')
for name, d in claim_diag.items():
    report.append(f"- {name}: supported-claim recall {d['supported_claim_recall']:.1%}; claim balanced accuracy {d['claim_balanced_accuracy']:.1%}.")
report.append('There are only six supported and thirty unsupported claims per split. A policy that always says unsupported can achieve zero false support while failing every supported claim. Interpret the unsupported-assertion metric together with this diagnostic and family accuracy.')

report.append("\n## Reward modelling and shortcut probes")
report.append(
    f"Pairwise test accuracy is {reward['accuracy']:.1%}; reserved final accuracy is "
    f"{reward_final['accuracy']:.1%}. Test mean margin is {reward['mean_margin']:.4f}. "
    f"ID accuracy is {reward['by_distribution']['id']:.1%}, OOD accuracy "
    f"{reward['by_distribution']['ood']:.1%}. Ties count as errors."
)
report.append(
    f"With rhetorical text appended to the rejected answer, the incorrect answer wins "
    f"{reward['wrong_verbose_win_rate']:.1%} of comparisons. With semantics-preserving JSON indentation "
    f"of the rejected answer, it wins {reward['wrong_spaced_win_rate']:.1%}. "
    f"Accuracy against opposite-direction numerical errors is {reward['opposite_numeric_error_accuracy']:.1%}."
)
report.append(
    "The rhetorical probe conflates verbosity with invalid JSON; indentation preserves the structured "
    "answer. Opposite-sign corruptions test the one-sided numeric errors used during training. These "
    "are fixed probes, not an adaptive adversarial search. No policy was optimised against this reward, "
    "and preference accuracy is not objective scientific truth."
)
report.append("\n## Methods and reproducibility")
report.append(
    "Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision "
    "`7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0. It already underwent instruction tuning; "
    "this project does not study pretraining. Data: `scireason-synthetic-v2`, original procedural CC0 "
    "problems. Exact split hashes are in `experiments/dataset_manifest.json`. All preferences are "
    "programmatic oracle labels, not human annotations. Models use the same greedy decoding, 48 new "
    "tokens, seed 42 and chat template. No inference-time answer constraints are imposed."
)
report.append(
    "Hardware: Apple M4 Pro, 12 CPU/16 GPU cores, 24 GB unified memory, macOS 15.7.4. "
    "Python 3.12.13; PyTorch 2.15.0; Transformers 5.22.2; TRL 1.17.1; PEFT 0.21.2; "
    "Accelerate 1.15.0; Datasets 5.1.0. The full dependency graph is locked in uv.lock. "
    "Float32 LoRA targets q/v projections, rank 8, alpha 16, dropout .05; maximum training length "
    "256; microbatch 1, gradient accumulation 8. MPS seeds do not guarantee bitwise reproducibility "
    "across devices or library versions."
)
report.append(
    "| Stage | Epochs | Trainable parameters | Trainer seconds | End-to-end training wrapper seconds | Max RSS GiB | Validation loss |\n|---|---:|---:|---:|---:|---:|---:|"
)
for n, t in training.items():
    setup = read(f"experiments/{n}_setup.json")
    report.append(
        f"| {n} | {setup['config']['epochs']} | {setup['trainable_parameters']:,} | "
        f"{t['metrics']['train_runtime']:.1f} | {t['elapsed_seconds']:.1f} | "
        f"{t['memory']['process_max_rss_bytes'] / 2**30:.2f} | {t['validation']['eval_loss']:.4f} |"
    )
report.append(
    "Wrapper timing includes trainer.train, final validation and checkpoint saving, but excludes model "
    "loading and trainer construction. DPO reference precomputation occurs inside trainer.train. "
    "RSS is the process high-water mark and does not include all Metal/OS memory; logged MPS allocation "
    "is an end-of-run measurement, not peak GPU memory. No paid infrastructure was used."
)
report.append("\n![Training curves](training_curves.png)")
report.append(
    "SFT minimises completion-only cross-entropy. DPO minimises −log σ(β[Δlogπ−Δlogπref]); "
    "the reference is a verified frozen copy of SFT, not the adapter-disabled baseline. The beta .1 "
    "and .5 runs start from the same SFT checkpoint and each train one epoch. RewardTrainer uses "
    "−log σ(r_chosen−r_rejected), with centering coefficient .01; a separate scalar head and LoRA "
    "are optimised. Reduced objective loss is not proof of improved free-generation correctness."
)
report.append("\n## Representative successes, failures and regressions")
for name in names:
    rows = records(name)
    success = next((r for r in rows if r["grade"]["correct"]), None)
    failure = next((r for r in rows if not r["grade"]["correct"]), None)
    report.append(f"\n### {name}")
    for label, r in [("Success", success), ("Failure", failure)]:
        if r:
            report.append(
                f"- {label} ({r['id']}, {r['family']}): {r['prompt']} "
                f"Expected `{r['answer']}`; generated `{r['prediction']}`."
            )
base = records("baseline")
for name in names[1:]:
    reg = [r for a, r in zip(base, records(name)) if a["grade"]["correct"] and not r["grade"]["correct"]]
    report.append(
        f"- {name}: {len(reg)} previously correct baseline examples became incorrect; all such "
        "cases are retained in `regressions_test.json`."
    )
sft = records("sft")
dpo = records("dpo")
reg = [(a, b) for a, b in zip(sft, dpo) if a["grade"]["correct"] and not b["grade"]["correct"]]
report.append(f"- DPO versus SFT: {len(reg)} correct SFT examples became incorrect.")
for a, b in reg[:3]:
    report.append(f"  - `{a['id']}` {a['family']}: SFT `{a['prediction']}` → DPO `{b['prediction']}`.")
reward_failures = [r for r in records("reward") if not r["correct"]]
for r in reward_failures[:3]:
    report.append(
        f"- Reward failure `{r['id']}` ({r['family']}): margin {r['margin']:.4f}, "
        f"chosen `{r['chosen']}`, rejected `{r['rejected']}`."
    )
report.append("\n## Limitations, failures and scope")
report.append(
    "Pilot v1 failed a stronger split audit: distinct wording hid some repeated numerical conditions. "
    "Version 2 rejects these globally. All primary training/evaluation was repeated; pilot artefacts "
    "remain visible. Operations in ID remain similar by design; OOD has only four simple families, "
    "and final replication reuses OOD templates. Pretraining contamination cannot be excluded. "
    "This is elementary structured-answer reasoning, not multi-step research, clinical validation or "
    "general intelligence. Synthetic labels, format demands, one-sided rejected numbers and one "
    "training seed limit interpretation. There are only 12 examples per family per held-out split."
)
report.append(
    "PPO status: **NOT RUN**. The blocker is scientific scope and an unvalidated online reward signal, "
    "not a demonstrated hardware impossibility. A defensible extension needs a compatible maintained "
    "online trainer, rollout/old/reference policies, value/advantage validation, KL diagnostics, a "
    "stronger reward model and separate compute budget. No GRPO substitute is claimed."
)
report.append("\n## Verification and attribution")
report.append(
    "Local CPU tests cover schemas, disjoint splits, oracle preferences, metric tolerances, LoRA "
    "updates, frozen DPO reference and checkpoint resume. Saved full-size adapters were reloaded for "
    "all evaluations. `artifact_verification.json` checks predictions, metrics and weight hashes. "
    "Plots are derived from the same saved rows. GitHub Actions is configured; remote CI is NOT RUN "
    "until publication. Code and experiment execution are AI-assisted; personal mastery requires "
    "independent reproduction. See the Russian learning and interview guides."
)
report.append(
    "\nReferences: [TRL](https://huggingface.co/docs/trl/index), "
    "[Alignment Handbook](https://github.com/huggingface/alignment-handbook), "
    "[DPO](https://arxiv.org/abs/2305.18290), [LoRA](https://arxiv.org/abs/2106.09685)."
)
Path("reports/research_report.md").write_text("\n\n".join(report) + "\n")
Path("reports/results_summary.md").write_text(
    "# Measured results\n\n"
    + Path("reports/results_test.md").read_text()
    + "\nReserved replication:\n\n"
    + Path("reports/results_final.md").read_text()
    + f"\nReward test accuracy: {reward['accuracy']:.1%}; final: {reward_final['accuracy']:.1%}.\n\nSee research_report.md for limitations and paired intervals.\n"
)
Path("reports/cv_bullets.md").write_text(f"""# Evidence-based CV draft

Use after personally reproducing and defending the work; implementation was AI-assisted.

- Built and executed an AI-assisted, reproducible post-training study using PyTorch, Hugging Face TRL and LoRA, training SFT, DPO and a scalar reward model on a 0.5B-parameter language model with Apple Silicon.
- Developed a programmatically verified synthetic scientific benchmark with family-held-out evaluation, leakage checks, deterministic grading and paired confidence intervals; compared four policies across two 120-example evaluation splits.
- Investigated reward-model generalisation and superficial-feature sensitivity, reporting {reward["accuracy"]:.1%} held-out pairwise accuracy alongside counterexamples and numerical-error-direction probes; implemented CPU integration tests for masking, frozen references and checkpoint recovery.

Do not claim PPO, large-scale RLHF, unaided authorship, or general scientific reliability.
""")
# Add measured experiment cards to the Russian guide without duplicating them on rerun.
p = Path("docs/learning_guide_ru.md")
s = p.read_text().split("\n## 9. Карточки выполненных экспериментов")[0]
s += "\n## 9. Карточки выполненных экспериментов\n"
for n in names:
    s += f"\n### {n}\n\n"
    s += "1. Вопрос: как фиксированная policy отвечает на одинаковые проверяемые задачи?\n"
    objective = (
        "без дополнительного обучения"
        if n == "baseline"
        else ("completion cross-entropy" if n == "sft" else "DPO logistic loss относительно SFT reference")
    )
    s += f"2. Модель: Qwen2.5-0.5B-Instruct; objective: {objective}.\n"
    s += "3. Код: `runtime.py`, `metrics.py`; обучение — `training.py`, настройки — `configs/`.\n"
    s += (
        "4. Процедура: greedy inference на 120 test и 120 final задачах.\n"
        if n == "baseline"
        else f"4. Процедура: LoRA, {2 if n == 'sft' else 1} эпохи, batch accumulation 8, seed 42, reload checkpoint.\n"
    )
    s += f"5. Наблюдение: test accuracy {metrics[n]['accuracy']:.1%}, final {final[n]['accuracy']:.1%}; numerical test accuracy {metrics[n]['numeric_accuracy']:.1%}.\n"
    s += "6. Что сработало/не сработало: выполненный pipeline даёт проверяемые ответы и сохраняет ошибки; конкретные успехи и регрессии приведены в research report.\n"
    s += "7. Ограничение: маленькие искусственные семейства, один seed; aggregate score смешивает формат и корректность.\n"
    s += "8. Упражнение: вручную проверьте по одному успеху и провалу и объясните связь с objective.\n"
s += f"""\n### Reward model

1. Вопрос: переносится ли предпочтение правильного ответа на новые семейства и поверхностные изменения?
2. Модель: Qwen backbone + scalar head + LoRA; pairwise logistic loss.
3. Код: `training.py`, `reward_evaluation.py`, `configs/reward.yaml`.
4. Процедура: две эпохи на 288 автоматически размеченных парах, validation, reload, test/final.
5. Результат: test {reward["accuracy"]:.1%}, final {reward_final["accuracy"]:.1%}; противоположный знак числовой ошибки {reward["opposite_numeric_error_accuracy"]:.1%}.
6. Проверки многословия и форматирования выполнены; margin и провалы сохранены, устойчивость к активному reward hacking не доказана.
7. Ограничение: простые synthetic rejected и узкий набор adversarial probes.
8. Упражнение: найдите пару с отрицательным margin и предложите исправление данных без доступа к final labels при подборе модели.
"""
p.write_text(s)
print(
    "Research report, results summary, CV draft and Russian experiment cards generated from measured artefacts."
)
