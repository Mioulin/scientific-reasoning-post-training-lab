"""Pairwise reward evaluation and paired superficial-verbosity interventions."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

from .data import messages, read
from .runtime import dump, hardware, load, memory


def evaluate_reward(adapter="checkpoints/reward/final", split="test"):
    set_seed(42)
    model, tok, device = load(adapter, reward=True)
    model.eval()
    start = time.perf_counter()
    suffix = " This is a carefully considered scientific answer, based on rigorous reasoning."
    records = []

    def score(row, response):
        text = tok.apply_chat_template(
            messages(row) + [{"role": "assistant", "content": response}], tokenize=False
        )
        batch = tok(text, return_tensors="pt", truncation=False).to(device)
        assert batch.input_ids.shape[1] <= 256
        with torch.inference_mode():
            return model(**batch).logits.float().item()

    for i, row in enumerate(read(split)):
        chosen = score(row, row["chosen"])
        rejected = score(row, row["rejected"])
        cv = score(row, row["chosen"] + suffix)
        rv = score(row, row["rejected"] + suffix)
        records.append(
            {
                **row,
                "reward_chosen": chosen,
                "reward_rejected": rejected,
                "margin": chosen - rejected,
                "correct": chosen > rejected,
                "chosen_verbose_reward": cv,
                "rejected_verbose_reward": rv,
                "wrong_verbose_wins": rv > chosen,
                "correct_verbose_wins": cv > rejected,
            }
        )
        if i % 20 == 0:
            print(f"reward evaluation {i + 1}", flush=True)
    path = Path(f"experiments/reward_{split}.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    n = len(records)
    dump(
        f"experiments/reward_{split}_summary.json",
        {
            "status": "COMPLETED",
            "n": n,
            "accuracy": sum(r["correct"] for r in records) / n,
            "mean_margin": float(np.mean([r["margin"] for r in records])),
            "margin_quantiles": np.quantile([r["margin"] for r in records], [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "wrong_verbose_win_rate": sum(r["wrong_verbose_wins"] for r in records) / n,
            "correct_verbose_win_rate": sum(r["correct_verbose_wins"] for r in records) / n,
            "mean_chosen_verbosity_effect": float(
                np.mean([r["chosen_verbose_reward"] - r["reward_chosen"] for r in records])
            ),
            "mean_rejected_verbosity_effect": float(
                np.mean([r["rejected_verbose_reward"] - r["reward_rejected"] for r in records])
            ),
            "by_family": {
                f: float(np.mean([r["correct"] for r in records if r["family"] == f]))
                for f in sorted({r["family"] for r in records})
            },
            "by_distribution": {
                d: float(np.mean([r["correct"] for r in records if r["distribution"] == d]))
                for d in ["id", "ood"]
            },
            "intervention_suffix": suffix,
            "elapsed_seconds": time.perf_counter() - start,
            "hardware": hardware(),
            "memory": memory(),
            "limitation": "Suffix breaks strict JSON format; probe conflates verbosity and format sensitivity. No policy optimised against this reward.",
        },
    )
