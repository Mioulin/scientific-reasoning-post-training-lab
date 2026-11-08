import json
import re
from collections import Counter
from pathlib import Path

from scireason.data import read
from scireason.metrics import grade

report = {}
seen_prompts = set()
seen_conditions = set()
for split in ["train", "validation", "test", "final"]:
    rows = read(split)
    for row in rows:
        assert row["prompt"] not in seen_prompts
        seen_prompts.add(row["prompt"])
        key = (row["family"], tuple(sorted(re.findall(r"\d+", row["prompt"]))))
        assert key not in seen_conditions
        seen_conditions.add(key)
        assert grade(row["chosen"], row["answer"])["correct"]
        assert not grade(row["rejected"], row["answer"])["correct"]
    report[split] = {
        "n": len(rows),
        "families": dict(Counter(r["family"] for r in rows)),
        "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "answer_classes": dict(
            Counter(
                "numeric"
                if type(r["answer"]) in (int, float)
                else "abstain"
                if r["answer"] is None
                else r["answer"]
                for r in rows
            )
        ),
    }
report["checks"] = {
    "exact_prompt_duplicates": 0,
    "numeric_condition_duplicates": 0,
    "invalid_preferences": 0,
    "scope": "all four generated splits; does not exclude semantic or pretraining overlap",
}
Path("experiments/dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print("Dataset balance and all preference labels audited.")
