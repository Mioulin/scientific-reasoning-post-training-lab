# Техническая защита проекта

Числа и статусы берите из research report и сохранённых experiment JSON. Репозиторий
создан при помощи AI-агента; «самостоятельно реализовала» нельзя подменять наличием
сгенерированного кода. Честная формулировка — провела и проверила AI-assisted проект,
если вы действительно воспроизвели команды, проверили результаты и понимаете код.

## Why did you choose DPO rather than PPO?

**Short:** “DPO let me isolate the effect of offline preferences with a frozen SFT
reference and a modest compute budget. I did not need to confound the study with
online sampling, value estimation and reward-model errors.”

**Detailed:** “PPO would require rollouts, an old policy for importance ratios, a
reference policy for regularisation, and normally a value estimator. DPO directly
optimises pairwise log-probability differences. I trained the reward model separately
to study preference generalisation rather than using it as an unvalidated training
signal. This comparison does not establish that DPO is universally superior.”

По-русски: выбор обусловлен исследовательским контролем и бюджетом. PPO здесь
не является выполненным методом. Отдельно обученный RM — не полный RLHF pipeline.

## What is the mathematical difference between SFT and DPO?

**Short:** “SFT maximises the likelihood of a target completion. DPO optimises the
chosen-versus-rejected log-likelihood ratio relative to a fixed reference policy.”

**Detailed:** “SFT minimises token-level cross-entropy on completion tokens only.
DPO minimises −log σ(β[(log πθ(y+|x)−log πθ(y−|x))−(log πref(y+|x)−log πref(y−|x))]).
The latter is a pairwise objective, not a direct target for absolute correctness.
Both update the policy LoRA parameters in this project.”

По-русски: объясните teacher forcing, causal shift и маску −100. DPO может повысить
относительное предпочтение, даже если абсолютная вероятность chosen снизилась.

## How does DPO depend on the reference policy?

**Short:** “The reference defines the baseline likelihood ratio. I used a frozen
copy of the SFT adapter, verified equal before optimisation.”

**Detailed:** “Disabling the SFT adapter would recover the original Qwen policy,
which would be a different reference. The pinned TRL implementation creates a
reference adapter; setup assertions and a CPU integration test check equality and
immutability. Reference log probabilities are precomputed because they are constant.”

По-русски: найдите `reference_check` в setup JSON и тест `test_dpo_preserves_sft_reference`.
Нельзя утверждать, что beta гарантирует фактический KL budget на всех задачах.

## How do you detect reward hacking?

**Short:** “I test whether a wrong answer can win by changing superficial presentation,
and whether accuracy survives a different error-generating rule.”

**Detailed:** “The reward evaluation adds identical rhetorical text to correct and
incorrect responses in separate interventions, tests semantics-preserving JSON
indentation, and reverses the sign of numerical corruptions. I inspect margins and
OOD errors. These are probes of exploitable shortcuts, not proof that an online
optimiser cannot exploit the model. No policy was optimised against this reward.”

По-русски: риторическое дополнение нарушает JSON, поэтому не позволяет изолировать
чистую любовь к многословию. Indentation сохраняет семантику. Ни один конечный набор
таких проверок не доказывает отсутствие reward hacking.

## How do you know an improvement is real?

**Short:** “I compare the same frozen examples with the same decoding settings and
report paired uncertainty, numerical errors and regressions, not training loss alone.”

**Detailed:** “The primary endpoint counts malformed answers as failures. I also
report format validity and a labelled post-hoc numeric-string diagnostic, because
format learning can masquerade as reasoning improvement. Paired bootstrap intervals
are conditional on these templates and one training seed. Replication on reserved
instances helps, but does not measure training-seed or domain uncertainty.”

По-русски: различайте статистическую неопределённость и систематические ограничения.
Положительная delta на этом benchmark не означает универсальный научный прогресс.

## What are the main threats to benchmark validity?

**Short:** “Simple synthetic templates, easy rejected responses, limited families,
format sensitivity and possible pretraining familiarity.”

**Detailed:** “An early audit found numerical-instance overlap despite distinct prompt
wording. I corrected the generator, retained the pilot artefacts and repeated primary
runs. ID tasks still share mathematical operations by design. Four OOD families are
unseen in training, but final replication reuses their templates. Labels are programmatic,
not human judgements, and unsupported-claim metrics are narrow binary decisions.”

По-русски: исправление утечки — конкретный инженерный эпизод для интервью. Не называйте
все строки независимыми научными задачами: множество строк происходит из одного шаблона.

## How would you scale this experiment?

**Short:** “I would improve evaluation diversity and preference quality before simply
increasing parameter count.”

**Detailed:** “I would add independently authored task families, multiple training seeds,
external benchmarks with suitable licences, harder model-generated rejected answers,
and human adjudication on ambiguous cases. On CUDA I would benchmark bf16 LoRA and
supported quantisation rather than assuming MPS settings transfer unchanged.”

По-русски: масштабирование — не только больше GPU. Нужны data lineage, immutable config,
checkpoint recovery, throughput/memory measurements и проверка масок на новом tokenizer.

## How would you improve preference data?

**Short:** “I would match superficial features and collect realistic, diverse errors
rather than deterministic one-sided numerical perturbations.”

**Detailed:** “I would generate candidate responses from several policies, independently
verify numerical answers, balance length and answer magnitude, include negative and
positive perturbations, and split by derivation or family. I would record uncertainty
and disagreement where the preference is not objectively determined.”

По-русски: автоматическая пара не является человеческим предпочтением. Более сложный
rejected полезен только при корректной метке; ошибочные метки делают objective вредным.

## What would you change if you had eight GPUs?

**Short:** “I would spend part of the budget on replication and stronger evaluation,
then scale model size and consider an explicitly controlled online RL experiment.”

**Detailed:** “I would profile before selecting DDP or FSDP, run several seeds and
reference/beta comparisons, and keep a sealed final benchmark. PPO would need a
validated reward signal, rollout accounting, KL monitoring, value-loss diagnostics
and reward-hacking probes. Eight GPUs do not remove data-quality limitations.”

По-русски: это будущий план, не выполненная distributed training реализация.

## What did you personally implement and verify?

**Short:** “This is an AI-assisted research engineering project. The repository records
what was executed; I distinguish that from the parts I have independently reproduced
and can explain.”

**Detailed:** “Executed evidence includes procedural data, baseline inference, LoRA
training through maintained trainers, checkpoint reloads, deterministic grading,
paired comparisons and CPU integration tests, where their status is COMPLETED.
I would demonstrate my own understanding by tracing masks and gradients, reproducing
a run and explaining a failure. I would not claim unaided implementation of TRL or PPO.”

По-русски: перед интервью самостоятельно выполните упражнения learning guide.
Заполните личный журнал: что запустили своими руками, какой баг исправили и почему.
Не приписывайте себе прежний опыт PPO/RLHF на основании этого репозитория.

## CV bullets

Финальные, подтверждённые числа и формулировки находятся в `reports/cv_bullets.md`.
Используйте их лишь после личной проверки; не удаляйте оговорку об AI-assisted работе,
если иначе создаётся впечатление самостоятельной реализации всей инфраструктуры.
