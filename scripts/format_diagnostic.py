"""Post-hoc diagnostic: numeric strings are converted; primary scores unchanged."""

import json
import math
from pathlib import Path

results = {}
for path in sorted(Path("experiments").glob("*_test.jsonl")):
    if path.name.startswith("reward_"):
        continue
    if not path.with_name(path.stem + "_summary.json").exists():
        continue
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    numeric = [r for r in rows if type(r["answer"]) in (int, float)]
    parsed = correct = 0
    for row in numeric:
        try:
            obj = json.loads(row["prediction"])
            value = obj["answer"]
            if isinstance(value, bool) or value is None or obj.get("abstain") is not False:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            parsed += 1
            correct += math.isclose(value, row["answer"], rel_tol=1e-3, abs_tol=1e-3)
        except (ValueError, TypeError, KeyError):
            pass
    results[path.stem] = {
        "numeric_n": len(numeric),
        "parseable_numeric_or_string": parsed,
        "correct_after_numeric_string_conversion": correct,
        "accuracy": correct / len(numeric),
        "status": "Post-hoc format sensitivity diagnostic; not the preregistered primary endpoint.",
    }
Path("reports/format_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
