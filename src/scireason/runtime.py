import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed


def config():
    return json.loads(Path("configs/model.json").read_text())


def hardware():
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy"]
        },
    }


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(adapter=None, reward=False):
    cfg = config()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], cache_dir=cfg["cache_dir"], local_files_only=True
    )
    tok.pad_token = tok.eos_token
    cls = AutoModelForSequenceClassification if reward else AutoModelForCausalLM
    kwargs = {"num_labels": 1} if reward else {}
    model = cls.from_pretrained(
        cfg["model_id"],
        revision=cfg["revision"],
        cache_dir=cfg["cache_dir"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        **kwargs,
    )
    model.config.pad_token_id = tok.pad_token_id
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model.to(device), tok, device


def memory():
    return {
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_allocated_bytes": torch.mps.current_allocated_memory()
        if torch.backends.mps.is_available()
        else None,
        "note": "RSS is process high-water mark; MPS value is current allocation, not peak.",
    }


def evaluate(name, adapter=None, split="test", limit=None):
    from .data import messages, read
    from .metrics import grade, summarise

    set_seed(42)
    model, tok, device = load(adapter)
    model.eval()
    data = read(split)
    if limit:
        data = data[:limit]
    path = Path(f"experiments/{name}_{split}.jsonl")
    path.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    records = []
    with path.open("w") as handle:
        for i, row in enumerate(data):
            text = tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id
                )
            prediction = tok.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            result = {**row, "prediction": prediction, "grade": grade(prediction, row["answer"])}
            records.append(result)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if i % 10 == 0:
                print(f"{name}: {i + 1}/{len(data)} {prediction}", flush=True)
    dump(
        f"experiments/{name}_{split}_summary.json",
        {
            "status": "COMPLETED",
            "model": config(),
            "adapter": adapter,
            "split": split,
            "limit": limit,
            "seed": 42,
            "generation": {"max_new_tokens": 48, "do_sample": False},
            "hardware": hardware(),
            "elapsed_seconds": time.perf_counter() - start,
            "memory": memory(),
            "metrics": summarise(records),
        },
    )
