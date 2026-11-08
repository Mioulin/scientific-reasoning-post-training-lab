"""Post-hoc diagnostics, explicitly separate from the frozen strict primary score."""

import ast
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scireason.metrics import paired_ci
from scireason.runtime import dump
from scireason.toolstudy import calculate


def main():
    root = Path("experiments/toolstudy")
    report = Path("reports/toolstudy")
    table = []
    examples = []
    names = ["baseline"] + [f"{kind}_{seed}" for seed in [42, 43, 44] for kind in ["sft", "dpo"]]
    for name in names:
        for split in ["test", "final"]:
            for mode in ["direct", "calculator"]:
                rows = [
                    json.loads(s) for s in (root / f"{name}_{split}_{mode}.jsonl").read_text().splitlines()
                ]
                numerical = [r for r in rows if type(r["answer"]) in (int, float)]
                value_correct = 0
                for r in numerical:
                    try:
                        obj = json.loads(r["prediction"])
                        answer = obj["answer"]
                        if type(answer) not in (str, int, float):
                            continue
                        value = calculate(str(answer)) if mode == "calculator" else float(answer)
                        value_correct += math.isclose(value, r["answer"], rel_tol=1e-3, abs_tol=1e-3)
                    except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError):
                        pass
                table.append(
                    {
                        "policy": name,
                        "split": split,
                        "mode": mode,
                        "strict_accuracy": np.mean([r["grade"]["correct"] for r in rows]),
                        "format_valid_rate": np.mean([r["grade"]["format_valid"] for r in rows]),
                        "numeric_value_only_accuracy_posthoc": value_correct / len(numerical),
                        "id_accuracy": np.mean(
                            [r["grade"]["correct"] for r in rows if r["distribution"] == "id"]
                        ),
                        "ood_accuracy": np.mean(
                            [r["grade"]["correct"] for r in rows if r["distribution"] == "ood"]
                        ),
                        "successful_tool_calls": sum(r["tool_used"] for r in rows),
                    }
                )
                if split == "final":
                    for status in [True, False]:
                        candidates = [r for r in rows if r["grade"]["correct"] == status]
                        if candidates:
                            examples.append({"policy": name, "mode": mode, **candidates[0]})
    comparisons = {}
    for split in ["test", "final"]:
        for mode in ["direct", "calculator"]:

            def records(name, split=split, mode=mode):
                return [
                    json.loads(s) for s in (root / f"{name}_{split}_{mode}.jsonl").read_text().splitlines()
                ]

            base = records("baseline")
            for seed in [42, 43, 44]:
                sft = records(f"sft_{seed}")
                dpo = records(f"dpo_{seed}")
                for label, a, b in [("sft_minus_baseline", base, sft), ("dpo_minus_sft", sft, dpo)]:
                    comparisons[f"{split}_{mode}_{seed}_{label}"] = paired_ci(a, b)
    dump(report / "model_comparisons.json", comparisons)
    dump(report / "diagnostics.json", table)
    dump(report / "examples.json", examples)
    fig, ax = plt.subplots(figsize=(9, 4))
    for kind in ["sft", "dpo"]:
        for seed in [42, 43, 44]:
            hist = json.loads(Path(f"experiments/toolstudy_{kind}_{seed}_training.json").read_text())[
                "log_history"
            ]
            vals = [r for r in hist if "eval_loss" in r]
            ax.plot([r["epoch"] for r in vals], [r["eval_loss"] for r in vals], label=f"{kind} {seed}")
    ax.set(
        xlabel="Training epoch",
        ylabel="Validation objective",
        title="Different objectives: SFT cross-entropy and DPO preference loss",
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(report / "validation_losses.png", dpi=160)
    plt.close(fig)
    seed_summary = []
    for split in ["test", "final"]:
        for mode in ["direct", "calculator"]:
            for kind in ["sft", "dpo"]:
                rs = [
                    r
                    for r in table
                    if r["split"] == split and r["mode"] == mode and r["policy"].startswith(kind)
                ]
                vals = np.array([r["strict_accuracy"] for r in rs])
                seed_summary.append(
                    {
                        "split": split,
                        "mode": mode,
                        "kind": kind,
                        "mean_accuracy": float(vals.mean()),
                        "sample_sd_across_three_seeds": float(vals.std(ddof=1)),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    }
                )
    dump(report / "seed_summary.json", seed_summary)
    compute = []
    for kind in ["sft", "dpo"]:
        for seed in [42, 43, 44]:
            obj = json.loads(Path(f"experiments/toolstudy_{kind}_{seed}_training.json").read_text())
            compute.append(
                {
                    "stage": f"{kind}_{seed}",
                    "type": "training_and_validation",
                    "elapsed_seconds": obj["elapsed_seconds"],
                    "memory": obj["memory"],
                }
            )
    for path in sorted(root.glob("*.summary.json")):
        obj = json.loads(path.read_text())
        compute.append(
            {
                "stage": path.stem,
                "type": "inference",
                "elapsed_seconds": obj["elapsed_seconds"],
                "memory": obj["memory"],
            }
        )
    dump(
        report / "compute.json",
        {
            "stages": compute,
            "sum_instrumented_seconds": sum(r["elapsed_seconds"] for r in compute),
            "note": "Sum of stage timers, not end-to-end wall time. Excludes model loading and code preparation; memory fields distinguish RSS high-water from current MPS allocation.",
        },
    )
    path = report / "research_report.md"
    text = (
        path.read_text()
        + "\n## Diagnostic decomposition (post hoc)\n\nThe strict primary outcome combines schema compliance and task correctness. After observing baseline format failures, we added numerical value-only grading as a diagnostic: it ignores missing/extra keys and abstention flags, converts numeric strings, and executes calculator expressions independently of schema. This is not the preregistered primary outcome and is not evidence of reliable instruction following. All predictions are unchanged.\n\n| Policy | Mode | Final strict | Format valid | Numeric value only | ID | OOD |\n|---|---|---:|---:|---:|---:|---:|\n"
    )
    for r in table:
        if r["split"] == "final":
            text += f"| {r['policy']} | {r['mode']} | {r['strict_accuracy']:.3f} | {r['format_valid_rate']:.3f} | {r['numeric_value_only_accuracy_posthoc']:.3f} | {r['id_accuracy']:.3f} | {r['ood_accuracy']:.3f} |\n"
    text += "\nExploratory paired model comparisons are in model_comparisons.json (no multiplicity correction; intervals do not capture family uncertainty). Training seed ranges are in seed_summary.json; representative first-in-file successes/failures are in examples.json, with no claim that these are a random qualitative sample. Compute timers and memory definitions are in compute.json. Loss curves compare different objectives and their absolute values are not directly comparable. The tool-output schema differs from SFT numerical-output training; failures may reflect instruction distribution shift. No conclusion of intrinsic tool ineffectiveness is warranted.\n"
    final_sft = next(
        r for r in seed_summary if r["split"] == "final" and r["mode"] == "direct" and r["kind"] == "sft"
    )
    final_dpo = next(
        r for r in seed_summary if r["split"] == "final" and r["mode"] == "direct" and r["kind"] == "dpo"
    )
    executions = []
    for p in root.glob("*calculator.jsonl"):
        executions.extend(r for r in map(json.loads, p.read_text().splitlines()) if r["tool_used"])
    operations = sum(
        any(
            isinstance(n, (ast.BinOp, ast.Call))
            for n in ast.walk(ast.parse(json.loads(r["prediction"])["answer"], mode="eval"))
        )
        for r in executions
    )
    base = next(
        r for r in table if r["policy"] == "baseline" and r["split"] == "final" and r["mode"] == "direct"
    )
    text += f"\n## Observed conclusion\n\nFinal strict accuracy: baseline {base['strict_accuracy']:.2%}; SFT mean {final_sft['mean_accuracy']:.2%} (seed SD {100 * final_sft['sample_sd_across_three_seeds']:.2f} percentage points); DPO mean {final_dpo['mean_accuracy']:.2%} (SD {100 * final_dpo['sample_sd_across_three_seeds']:.2f} points). Compare individual seeds before generalising: the strict score includes formatting, not only scientific knowledge.\n\nAcross all calculator conditions, {len(executions)} requests executed; {operations} contained an arithmetic operation. Executing a numeric literal does not demonstrate programme construction. The saved records distinguish execution from a correct final answer; this study does not establish reliable tool use.\n"
    text += f"\nInstrumented training/evaluation timers sum to {sum(r['elapsed_seconds'] for r in compute):.1f} seconds, excluding model loading and development. Exact model revision, software, device and masking/reference checks are recorded in setup and summary JSON.\n"
    path.write_text(text)


if __name__ == "__main__":
    main()
