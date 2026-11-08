import json
from pathlib import Path

from transformers import AutoTokenizer

from scireason.data import messages, read

cfg = json.loads(Path("configs/model.json").read_text())
tok = AutoTokenizer.from_pretrained(
    cfg["model_id"], revision=cfg["revision"], cache_dir=cfg["cache_dir"], local_files_only=True
)
report = {}
for split in ["train", "validation", "test", "final"]:
    lengths = []
    for row in read(split):
        for answer in [row["chosen"], row["rejected"]]:
            ids = tok.apply_chat_template(
                messages(row) + [{"role": "assistant", "content": answer}], tokenize=True
            )
            lengths.append(len(ids))
    assert max(lengths) <= 256
    report[split] = {
        "pairs": len(lengths) // 2,
        "max_sequence_tokens": max(lengths),
        "min_sequence_tokens": min(lengths),
        "truncated": 0,
    }
Path("experiments/tokenization_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
