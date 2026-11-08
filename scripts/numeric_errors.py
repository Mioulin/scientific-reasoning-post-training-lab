import json
from pathlib import Path
from statistics import mean

result = {}
for split in ["test", "final"]:
    result[split] = {}
    for name in ["baseline", "sft", "dpo", "dpo_beta05"]:
        rows = [json.loads(x) for x in Path(f"experiments/{name}_{split}.jsonl").read_text().splitlines()]
        groups = {}
        for family in ["rate", "mean", "difference", "bayes", "decay"]:
            selected = [r for r in rows if r["family"] == family]
            errors = [
                r["grade"]["absolute_error"] for r in selected if r["grade"]["absolute_error"] is not None
            ]
            groups[family] = {
                "n": len(selected),
                "parseable_n": len(errors),
                "mae_parseable_only": mean(errors) if errors else None,
                "accuracy": sum(r["grade"]["correct"] for r in selected) / len(selected),
            }
        result[split][name] = groups
Path("reports/numeric_errors_by_family.json").write_text(json.dumps(result, indent=2) + "\n")
