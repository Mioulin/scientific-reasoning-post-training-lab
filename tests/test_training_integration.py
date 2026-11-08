"""CPU-only integration tests with random tiny models; no Hub downloads."""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def tiny():
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "Question": 3, "right": 4, "wrong": 5}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    return model, tok


def test_sft_masks_prompt_and_updates_adapter(tmp_path):
    torch.set_num_threads(1)
    model, tok = tiny()
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        report_to="none",
        save_strategy="no",
        completion_only_loss=True,
        max_length=32,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": "Question ", "completion": "right [EOS]"}]),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["labels"][0, 0] == -100
    assert 4 in batch["labels"]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trainer.train()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)


def test_dpo_preserves_sft_reference(tmp_path):
    torch.set_num_threads(1)
    model, tok = tiny()
    # Nonzero adapter stands in for SFT; disabling it would be the wrong reference.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.fill_(0.1)
    args = DPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        report_to="none",
        save_strategy="no",
        max_length=32,
        beta=0.1,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
    )
    trainer = DPOTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=Dataset.from_list([{"prompt": "Question ", "chosen": "right", "rejected": "wrong"}]),
    )
    refs = {n: p.detach().clone() for n, p in model.named_parameters() if ".ref." in n}
    assert refs
    for n, p in model.named_parameters():
        if ".default." in n:
            assert torch.equal(p, refs[n.replace(".default.", ".ref.")])
    trainer.train()
    assert all(
        torch.equal(p, refs[n]) and not p.requires_grad for n, p in model.named_parameters() if n in refs
    )


def test_pairwise_loss_direction():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
    loss.backward()
    assert chosen.grad.item() < 0 and rejected.grad.item() > 0
    assert abs(loss.item() - 0.693147) < 1e-5
