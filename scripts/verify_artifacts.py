"""Check measured artefacts against data, metrics and local adapter files."""

import hashlib
import json
from pathlib import Path

from scireason.data import read
from scireason.metrics import grade, summarise

manifest = json.loads(Path("data/manifest.json").read_text())
checked = []
for split in ["test", "final"]:
    expected = read(split)
    for name in ["baseline", "sft", "dpo", "dpo_beta05"]:
        p = Path(f"experiments/{name}_{split}.jsonl")
        summary = json.loads(p.with_name(p.stem + "_summary.json").read_text())
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        assert len(rows) == len(expected) == 120
        assert [r["id"] for r in rows] == [r["id"] for r in expected]
        assert summary["dataset_manifest"] == manifest
        assert summary["metrics"] == summarise(rows)
        assert all(r["grade"] == grade(r["prediction"], r["answer"]) for r in rows)
        checked.append(str(p))
    rewards = [
        json.loads(line) for line in Path(f"experiments/reward_{split}.jsonl").read_text().splitlines()
    ]
    summary = json.loads(Path(f"experiments/reward_{split}_summary.json").read_text())
    assert [r["id"] for r in rewards] == [r["id"] for r in expected]
    assert summary["accuracy"] == sum(r["reward_chosen"] > r["reward_rejected"] for r in rewards) / len(
        rewards
    )
    checked.append(f"experiments/reward_{split}.jsonl")
checkpoints = {}
for name in ["sft", "dpo", "dpo_beta05", "reward"]:
    train = json.loads(Path(f"experiments/{name}_training.json").read_text())
    assert train["status"] == "COMPLETED"
    p = Path(train["checkpoint"]) / "adapter_model.safetensors"
    assert p.is_file()
    checkpoints[name] = {
        "path": str(p),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }
Path("reports/artifact_verification.json").write_text(
    json.dumps({"status": "COMPLETED", "checked_predictions": checked, "checkpoints": checkpoints}, indent=2)
    + "\n"
)
print("Verified 8 generative evaluations, 2 reward evaluations, and 4 saved adapters.")
