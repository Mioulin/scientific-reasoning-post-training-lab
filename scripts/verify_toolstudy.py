"""Audit every primary prediction, summary and dataset digest without rerunning inference."""

import hashlib
import json
from pathlib import Path

from scireason.metrics import grade, summarise
from scireason.runtime import dump
from scireason.toolstudy import resolve, semantic_key


def main():
    root = Path("experiments/toolstudy")
    manifest = json.loads(Path("data/toolstudy/manifest.json").read_text())
    seen = set()
    for split, item in manifest["splits"].items():
        p = Path(f"data/toolstudy/{split}.jsonl")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        keys = {semantic_key(r) for r in rows}
        assert len(keys) == len(rows) and not keys.intersection(seen)
        seen.update(keys)
    names = ["baseline"] + [f"{k}_{seed}" for seed in [42, 43, 44] for k in ["sft", "dpo"]]
    count = 0
    hashes = {}
    for name in names:
        for split in ["test", "final"]:
            expected = [json.loads(s) for s in Path(f"data/toolstudy/{split}.jsonl").read_text().splitlines()]
            for mode in ["direct", "calculator"]:
                p = root / f"{name}_{split}_{mode}.jsonl"
                rows = [json.loads(s) for s in p.read_text().splitlines()]
                assert len(rows) == 144 and [r["id"] for r in rows] == [r["id"] for r in expected]
                for r, e in zip(rows, expected):
                    assert r["prompt"] == e["prompt"] and r["answer"] == e["answer"]
                    resolved, used, error = resolve(r["prediction"], mode)
                    assert r["resolved"] == resolved and r["tool_used"] == used and r["tool_error"] == error
                    assert r["grade"] == grade(resolved, e["answer"])
                summary = json.loads(p.with_suffix(".summary.json").read_text())
                assert summary["status"] == "COMPLETED" and summary["batch_size"] == 8
                assert summary["metrics"] == summarise(rows) and summary["dataset"] == manifest
                count += len(rows)
                hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for kind in ["sft", "dpo"]:
        for seed in [42, 43, 44]:
            setup = json.loads(Path(f"experiments/toolstudy_{kind}_{seed}_setup.json").read_text())
            assert setup["dataset_manifest"] == manifest
            if kind == "dpo":
                assert setup["reference_check"]["frozen"] and setup["reference_check"]["same_as_initial_sft"]
            else:
                assert (
                    setup["mask_check"]["masked_tokens"] > 0 and setup["mask_check"]["supervised_tokens"] > 0
                )
    dump(
        "reports/toolstudy/verification.json",
        {
            "status": "COMPLETED",
            "verified_predictions": count,
            "files_sha256": hashes,
            "dataset_hashes_verified": True,
            "reference_and_mask_checks_verified": True,
        },
    )
    print(f"Verified {count} predictions, 28 summaries, dataset digests and six training setups.")


if __name__ == "__main__":
    main()
