"""Post-hoc check that low false support is not achieved by denying every claim."""

import json
from pathlib import Path

report = {}
for name in ["baseline", "sft", "dpo", "dpo_beta05"]:
    path = Path(f"experiments/{name}_test_summary.json")
    if not path.exists():
        continue
    rows = [json.loads(x) for x in path.with_name(f"{name}_test.jsonl").read_text().splitlines()]
    supported = [r for r in rows if r["answer"] == "supported"]
    unsupported = [r for r in rows if r["answer"] == "unsupported"]
    sr = sum(r["grade"]["correct"] for r in supported) / len(supported)
    ur = sum(r["grade"]["correct"] for r in unsupported) / len(unsupported)
    report[name] = {
        "supported_n": len(supported),
        "unsupported_n": len(unsupported),
        "supported_claim_recall": sr,
        "unsupported_claim_correct_rate": ur,
        "claim_balanced_accuracy": (sr + ur) / 2,
        "status": "Post-hoc class-balance diagnostic, not a new primary endpoint",
    }
Path("reports/claim_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
