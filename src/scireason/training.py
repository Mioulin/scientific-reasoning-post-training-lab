"""Maintained TRL trainers with explicit, testable objectives and frozen DPO reference."""

import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import set_seed
from trl import DPOConfig, DPOTrainer, RewardConfig, RewardTrainer, SFTConfig, SFTTrainer

from .data import messages, read
from .runtime import config, dump, hardware, load, memory


def dataset(split, kind, tok, limit=None):
    rows = read(split)
    if limit:
        rows = rows[:limit]
    if kind == "sft":
        return Dataset.from_list(
            [
                {"prompt": messages(r), "completion": [{"role": "assistant", "content": r["chosen"]}]}
                for r in rows
            ]
        )
    if kind == "dpo":
        return Dataset.from_list(
            [
                {
                    "prompt": messages(r),
                    "chosen": [{"role": "assistant", "content": r["chosen"]}],
                    "rejected": [{"role": "assistant", "content": r["rejected"]}],
                }
                for r in rows
            ]
        )
    return Dataset.from_list(
        [
            {
                "chosen": messages(r) + [{"role": "assistant", "content": r["chosen"]}],
                "rejected": messages(r) + [{"role": "assistant", "content": r["rejected"]}],
            }
            for r in rows
        ]
    )


def train(kind, config_path, resume=None, smoke=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg["seed"])
    name = cfg["name"] + ("_smoke" if smoke else "")
    output = f"checkpoints/{name}"
    model, tok, device = load(reward=kind == "reward")
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_CLS" if kind == "reward" else "CAUSAL_LM",
        modules_to_save=["score"] if kind == "reward" else None,
    )
    if kind == "dpo":
        model = PeftModel.from_pretrained(model, cfg["sft_checkpoint"], is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args = {
        "output_dir": output,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "num_train_epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "max_steps": 2 if smoke else -1,
        "bf16": False,
        "fp16": False,
        "gradient_checkpointing": False,
        "max_length": 256,
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": 18,
        "save_strategy": "steps",
        "save_steps": 18,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
        "dataloader_pin_memory": False,
        "optim": "adamw_torch",
        "use_cpu": device == "cpu",
        "disable_tqdm": True,
    }
    if kind == "sft":
        args = SFTConfig(**args, completion_only_loss=True, packing=False)
        cls = SFTTrainer
    elif kind == "dpo":
        args = DPOConfig(**args, beta=cfg["beta"], precompute_ref_log_probs=True)
        cls = DPOTrainer
    else:
        args = RewardConfig(**args, center_rewards_coefficient=0.01)
        cls = RewardTrainer
    trainer = cls(
        model=model,
        args=args,
        processing_class=tok,
        train_dataset=dataset("train", kind, tok, 8 if smoke else None),
        eval_dataset=dataset("validation", kind, tok, 4 if smoke else None),
    )
    reference_check = None
    if kind == "dpo":
        # TRL 1.17.1 clones the supplied SFT adapter as frozen "ref".
        pairs = [(n, p) for n, p in model.named_parameters() if ".default." in n]
        assert pairs and "ref" in model.peft_config
        for n, p in pairs:
            ref = model.get_parameter(n.replace(".default.", ".ref."))
            assert torch.equal(p.detach(), ref.detach()) and not ref.requires_grad
        reference_check = {
            "same_as_initial_sft": True,
            "frozen": True,
            "adapter": "ref",
            "parameters_checked": len(pairs),
        }
    # Verify completion masking on a real collated batch before optimisation.
    mask_check = None
    if kind == "sft":
        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch["labels"]
        assert (labels == -100).any() and (labels != -100).any()
        mask_check = {
            "masked_tokens": int((labels == -100).sum()),
            "supervised_tokens": int((labels != -100).sum()),
        }
    dump(
        f"experiments/{name}_setup.json",
        {
            "config": cfg,
            "model": config(),
            "device": device,
            "trainable_parameters": trainable,
            "reference_check": reference_check,
            "mask_check": mask_check,
            "hardware": hardware(),
        },
    )
    start = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        validation = trainer.evaluate()
        trainer.save_model(output + "/final")
        tok.save_pretrained(output + "/final")
        trainer.save_state()
        dump(
            f"experiments/{name}_training.json",
            {
                "status": "COMPLETED",
                "metrics": result.metrics,
                "validation": validation,
                "log_history": trainer.state.log_history,
                "elapsed_seconds": time.perf_counter() - start,
                "memory": memory(),
                "checkpoint": output + "/final",
            },
        )
    except Exception as exc:
        dump(
            f"experiments/{name}_failure.json",
            {
                "status": "FAILED",
                "type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        raise
